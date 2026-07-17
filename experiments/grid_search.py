"""
Grid search for optimal chunking parameters.

For every chunker listed in grid_search_config.yaml, this script:
  1. Uses ALL preprocessors from grid_search_config.yaml (applied in order).
  2. Iterates over every (chunk_size, overlap) combination defined in the config.
  3. After chunking, computes the average chunk length **in tokens** (using the
     generator model's tokenizer) across all chunks.
  4. Derives retrieval parameters dynamically:
       top_n = floor(2000 / avg_chunk_size_tokens)   (minimum 1)
       top_k = 3 * top_n
  5. Runs the full offline + online pipeline.
  6. Evaluates with minimal set.
  7. Saves per-configuration results and a summary CSV.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import List, Any

import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.factory import (
    load_yaml_config,
    build_offline_pipeline,
    build_online_pipeline,
    run_queries,
    write_summary_csv,
)
from src.config import OfflineConfig, OnlineConfig
from src.evaluation import Evaluator

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_avg_chunk_size(chunks: List[Any], tokenizer: AutoTokenizer) -> float:
    """Return the average token length of a list of Chunk objects using the generator tokenizer."""
    if not chunks:
        return 1.0
    total_tokens = sum(len(tokenizer.encode(c.text, add_special_tokens=False)) for c in chunks)
    return total_tokens / len(chunks)


def _derive_retrieval_params(avg_chunk_size_tokens: float) -> tuple[int, int]:
    """
    Derive top_n and top_k from the average chunk size in tokens.

    top_n = floor(2000 / avg_chunk_size_tokens)  (min 1)
    top_k = 3 * top_n
    """
    top_n = max(1, math.floor(2000.0 / avg_chunk_size_tokens))
    top_k = 3 * top_n
    return top_k, top_n


def _make_run_name(preprocessor_names: list[str], chunker_name: str, **params) -> str:
    """Build a short, filesystem-safe name for this grid run."""
    prep_part = "_".join(p[:6] for p in preprocessor_names)
    param_part = "_".join(str(v) for v in params.values())
    return f"{prep_part}_{chunker_name[:10]}_{param_part}"


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

def chunking_grid_search():
    # 1. Load configs
    grid_cfg = load_yaml_config("config/grid_search_config.yaml")
    base_cfg = load_yaml_config("config/config.yaml")

    preprocessor_names: list[str] = grid_cfg["preprocessing"]
    chunkers: list[dict] = grid_cfg["chunking"]
    document_paths: list[str] = base_cfg["documents"]

    offline_config = OfflineConfig(**base_cfg.get("offline_config", {}))
    online_config = OnlineConfig(**base_cfg.get("online_config", {}))
    online_pipeline_cfg: dict = base_cfg["online_pipeline"]
    offline_pipeline_cfg: dict = base_cfg["offline_pipeline"]

    # Load the generator tokenizer once — chunk sizes are measured in its token space
    logger.info("Loading tokenizer for '%s' ...", online_config.generation_model)
    tokenizer = AutoTokenizer.from_pretrained(online_config.generation_model)

    # QA evaluation files
    qa_eval_file = "storage/evaluation/qa_pairs_grid.json"
    with open(qa_eval_file) as f:
        qa_pairs_template = json.load(f)
    queries = [item["user_input"] for item in qa_pairs_template]

    # Output directory for grid search results
    results_dir = Path("storage/grid_search_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    # 2. Iterate over every chunker configuration
    for chunker_cfg in chunkers:
        chunker_name: str = chunker_cfg["name"]
        options: dict = chunker_cfg["options"]

        is_maxmin = chunker_name == "MaxMinChunker"

        if is_maxmin:
            c_values: list[float] = options["c"]
            fixed_threshold_values: list[float] = options["fixed_threshold"]
            embedding_model_name: str = options.get(
                "embedding_model_name", offline_config.embedding_model
            )
            param_combinations = [
                {"c": c_val, "fixed_threshold": ft_val}
                for c_val in c_values
                for ft_val in fixed_threshold_values
            ]
            logger.info(
                "=== Grid search for chunker '%s' | c=%s | fixed_threshold=%s ===",
                chunker_name,
                c_values,
                fixed_threshold_values,
            )
        else:
            chunk_sizes: list[int] = options["chunk_sizes"]
            overlaps: list[int] = options["overlaps"]
            max_total: int | None = options.get("max_paragraphs_per_chunk")
            param_combinations = [
                {"chunk_size": cs, "overlap": ov}
                for cs in chunk_sizes
                for ov in overlaps
                if ov < cs
                and (max_total is None or cs + ov <= max_total)
            ]
            logger.info(
                "=== Grid search for chunker '%s' | chunk_sizes=%s | overlaps=%s ===",
                chunker_name,
                chunk_sizes,
                overlaps,
            )

        for params in param_combinations:
            run_name = _make_run_name(preprocessor_names, chunker_name, **params)
            logger.info("--- Run: %s ---", run_name)

            index_path = Path("storage/grid_search_index") / run_name

            # Build and run offline pipeline
            chunker_kwargs = (
                {
                    "embedding_model_name": embedding_model_name,
                    **params,
                }
                if is_maxmin
                else params
            )
            offline_pipeline = build_offline_pipeline(
                preprocessor_names=preprocessor_names,
                chunker_name=chunker_name,
                index_builder_name=offline_pipeline_cfg["index_builder"],
                storage_path=index_path,
                embedding_model=offline_config.embedding_model,
                **chunker_kwargs,
            )
            offline_result = offline_pipeline.run(document_paths)
            chunks = offline_result.chunks

            # Derive dynamic retrieval parameters
            avg_chunk_size_tokens = _compute_avg_chunk_size(chunks, tokenizer)
            top_k, top_n = _derive_retrieval_params(avg_chunk_size_tokens)
            logger.info(
                "avg_chunk_size=%.1f tokens | top_n=%d | top_k=%d",
                avg_chunk_size_tokens, top_n, top_k,
            )

            # Build online pipeline, run queries, save raw results
            online_pipeline = build_online_pipeline(
                cfg=online_pipeline_cfg,
                index_builder=offline_pipeline.index_builder,
                top_k=top_k,
                top_n=top_n,
                generation_model=online_config.generation_model,
                reranking_score_threshold=online_config.reranking_score_threshold,
            )
            qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)

            qa_save = results_dir / f"{run_name}.json"
            with open(qa_save, "w") as f:
                json.dump(qa_pairs, f, indent=4)

            # Evaluate
            evaluator = Evaluator(str(qa_save))
            eval_df = evaluator.evaluate()
            metrics = eval_df.mean(numeric_only=True).to_dict()

            row = {
                "run_name": run_name,
                "preprocessors": "+".join(preprocessor_names),
                "chunker": chunker_name,
                # MaxMinChunker-specific columns (empty for other chunkers)
                "c": params.get("c", ""),
                "fixed_threshold": params.get("fixed_threshold", ""),
                # Standard chunker columns (empty for MaxMinChunker)
                "chunk_size": params.get("chunk_size", ""),
                "overlap": params.get("overlap", ""),
                "num_chunks": len(chunks),
                "avg_chunk_size_tokens": round(avg_chunk_size_tokens, 1),
                "top_k": top_k,
                "top_n": top_n,
                "reranking_threshold": online_config.reranking_score_threshold,
                **{f"{k}": v for k, v in metrics.items()},
            }
            summary_rows.append(row)
            logger.info("Run '%s' complete.", run_name)

    # 3. Write summary CSV
    if summary_rows:
        summary_path = results_dir / "grid_search_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Grid search complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed. Check your grid_search_config.yaml.")


if __name__ == "__main__":
    chunking_grid_search()