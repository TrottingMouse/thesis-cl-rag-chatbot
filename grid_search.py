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
import csv
from pathlib import Path
from typing import List, Any

import yaml
from dotenv import load_dotenv
from transformers import AutoTokenizer

from factory import load_yaml_config
from registry import get_class
from src.config import OfflineConfig, OnlineConfig
from src.offline.pipeline import OfflinePipeline
from src.online.pipeline import OnlinePipeline
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

    # Using encode() lets us accurately count the tokens for each chunk text
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
    grid_cfg = load_yaml_config("grid_search_config.yaml")
    base_cfg = load_yaml_config("config.yaml")

    preprocessor_names: list[str] = grid_cfg["preprocessing"]
    chunkers: list[dict] = grid_cfg["chunking"]
    document_paths: list[str] = base_cfg["documents"]

    # Shared pipeline settings from base config
    offline_kwargs = base_cfg.get("offline_config", {})
    online_kwargs = base_cfg.get("online_config", {})
    offline_config = OfflineConfig(**offline_kwargs)
    online_config = OnlineConfig(**online_kwargs)
    offline_cfg = base_cfg["offline_pipeline"]
    online_cfg = base_cfg["online_pipeline"]

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

    preprocessors = [get_class(name)() for name in preprocessor_names]

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
            param_combinations = [
                {"chunk_size": cs, "overlap": ov}
                for cs in chunk_sizes
                for ov in overlaps
                if ov < cs
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

            # Unique index path per run to avoid cross-run contamination
            index_path = Path("storage/grid_search_index") / run_name

            # ----------------------------------------------------------
            # Phase 1: Build preprocessors and chunker
            # ----------------------------------------------------------
            ChunkerClass = get_class(chunker_name)
            if is_maxmin:
                chunker = ChunkerClass(
                    embedding_model_name=embedding_model_name,
                    c=params["c"],
                    fixed_threshold=params["fixed_threshold"],
                )
            else:
                chunker = ChunkerClass(
                    chunk_size=params["chunk_size"],
                    overlap=params["overlap"],
                )

            IndexBuilderClass = get_class(offline_cfg["index_builder"])
            index_builder = IndexBuilderClass(
                storage_path=index_path,
                model_name=offline_config.embedding_model,
            )

            offline_pipeline = OfflinePipeline(preprocessors, chunker, index_builder)

            # ----------------------------------------------------------
            # Phase 2: Run offline pipeline (preprocess + chunk + index)
            # ----------------------------------------------------------
            offline_result = offline_pipeline.run(document_paths)
            chunks = offline_result.chunks

            # ----------------------------------------------------------
            # Phase 3: Derive dynamic retrieval parameters from chunk sizes
            # ----------------------------------------------------------
            avg_chunk_size_tokens = _compute_avg_chunk_size(chunks, tokenizer)
            top_k, top_n = _derive_retrieval_params(avg_chunk_size_tokens)

            logger.info(
                "avg_chunk_size=%.1f tokens | top_n=%d | top_k=%d",
                avg_chunk_size_tokens, top_n, top_k,
            )

            # ----------------------------------------------------------
            # Phase 4: Build online pipeline, reusing the already-loaded
            # index_builder instance from the offline pipeline.
            # (index_builder.index and .chunks are populated in memory.)
            # ----------------------------------------------------------
            query_processor = get_class(online_cfg["query_processor"])()

            RetrieverClass = get_class(online_cfg["retriever"])
            retriever = RetrieverClass(index_builder, top_k=top_k)

            RerankerClass = get_class(online_cfg["reranker"])
            reranker = RerankerClass(top_n=top_n)

            GeneratorClass = get_class(online_cfg["generator"])
            if online_cfg["generator"] == "HuggingfaceGenerator":
                generator = GeneratorClass(model_name=online_config.generation_model)
            else:
                generator = GeneratorClass()

            online_pipeline = OnlinePipeline(query_processor, retriever, reranker, generator)

            # ----------------------------------------------------------
            # Phase 5: Run online queries
            # ----------------------------------------------------------
            qa_online_results = online_pipeline.multiple_queries(queries)

            # Attach results to QA pairs (deep copy to avoid mutating the template)
            qa_pairs = copy.deepcopy(qa_pairs_template)

            for i, pipeline_result in enumerate(qa_online_results):
                qa_pairs[i]["response"] = pipeline_result.generation_result
                qa_pairs[i]["retrieved_contexts"] = [
                    chunk.text for chunk in pipeline_result.reranked_results
                ]

            # ----------------------------------------------------------
            # Phase 6: Save raw results
            # ----------------------------------------------------------
            qa_save = results_dir / f"{run_name}.json"
            with open(qa_save, "w") as f:
                json.dump(qa_pairs, f, indent=4)

            # ----------------------------------------------------------
            # Phase 7: Evaluate
            # ----------------------------------------------------------
            evaluator = Evaluator(str(qa_save))
            eval_df = evaluator.evaluate()

            #logger.info("Evaluation:\n%s", eval_df.to_string())

            # Aggregate numeric metrics (mean across QA pairs)
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
                **{f"{k}": v for k, v in metrics.items()},
            }
            summary_rows.append(row)
            logger.info("Run '%s' complete.", run_name)

    # 3. Write summary CSV
    if summary_rows:
        summary_path = results_dir / "grid_search_summary.csv"
        # Collect all fieldnames across all rows (different metrics may appear)
        all_fields: list[str] = []
        seen: set[str] = set()
        for row in summary_rows:
            for k in row:
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("Grid search complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed. Check your grid_search_config.yaml.")


if __name__ == "__main__":
    chunking_grid_search()