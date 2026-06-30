"""
Grid search for optimal chunking parameters.

For every chunker listed in grid_search_config.yaml, this script:
  1. Uses ALL preprocessors from grid_search_config.yaml (applied in order).
  2. Iterates over every (chunk_size, overlap) combination defined in the config.
  3. After chunking, computes the average chunk length (characters) across all chunks.
  4. Derives retrieval parameters dynamically:
       top_n = floor(1000 / avg_chunk_size)   (minimum 1)
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
import os
import csv
from pathlib import Path

import yaml
from dotenv import load_dotenv

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

def _compute_avg_chunk_size(chunks) -> float:
    """Return the average character length of a list of Chunk objects."""
    if not chunks:
        return 1.0
    return sum(len(c.text) for c in chunks) / len(chunks)


def _derive_retrieval_params(avg_chunk_size: float) -> tuple[int, int]:
    """
    Derive top_n and top_k from the average chunk size.

    top_n = floor(1000 / avg_chunk_size)  (min 1)
    top_k = 3 * top_n
    """
    top_n = max(1, math.floor(1000.0 / avg_chunk_size))
    top_k = 3 * top_n
    return top_k, top_n


def _make_run_name(preprocessor_names: list[str], chunker_name: str, chunk_size: int, overlap: int) -> str:
    """Build a short, filesystem-safe name for this grid run."""
    prep_part = "_".join(p[:6] for p in preprocessor_names)
    return f"{prep_part}_{chunker_name[:10]}_{chunk_size}_{overlap}"


# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

def chunking_grid_search():
    # 1. Load configs
    grid_cfg = load_yaml_config("grid_search_config.yaml")
    base_cfg = load_yaml_config("config.yaml")

    preprocessor_names: list[str] = grid_cfg["preprocessing"]
    chunkers: list[dict] = grid_cfg["chunking"]
    document_paths: list[str] = base_cfg["data"]["documents"]

    # Shared pipeline settings from base config
    offline_kwargs = base_cfg.get("offline_config", {})
    offline_config = OfflineConfig(**offline_kwargs)
    offline_cfg = base_cfg["offline_pipeline"]
    online_cfg = base_cfg["online_pipeline"]

    # QA evaluation files
    qa_eval_file = "storage/evaluation/qa_pairs.json"
    with open(qa_eval_file) as f:
        qa_pairs_template = json.load(f)

    qa_pairs = [item["user_input"] for item in qa_pairs_template]

    # Output directory for grid search results
    results_dir = Path("storage/grid_search_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    # 2. Iterate over every chunker configuration
    for chunker_cfg in chunkers:
        chunker_name: str = chunker_cfg["name"]
        chunk_sizes: list[int] = chunker_cfg["chunk_sizes"]
        overlaps: list[int] = chunker_cfg["overlaps"]

        logger.info(
            "=== Grid search for chunker '%s' | chunk_sizes=%s | overlaps=%s ===",
            chunker_name,
            chunk_sizes,
            overlaps,
        )

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                # Skip invalid overlap values (overlap must be < chunk_size)
                if overlap >= chunk_size:
                    logger.warning(
                        "Skipping invalid combination: chunker=%s, chunk_size=%d, overlap=%d "
                        "(overlap must be < chunk_size)",
                        chunker_name, chunk_size, overlap,
                    )
                    continue

                run_name = _make_run_name(preprocessor_names, chunker_name, chunk_size, overlap)
                logger.info("--- Run: %s ---", run_name)

                # Unique index path per run to avoid cross-run contamination
                index_path = Path("storage/grid_search_index") / run_name

                # ----------------------------------------------------------
                # Phase 1: Build preprocessors and chunker
                # ----------------------------------------------------------
                preprocessors = [get_class(name)() for name in preprocessor_names]

                ChunkerClass = get_class(chunker_name)
                chunker = ChunkerClass(chunk_size=chunk_size, overlap=overlap)

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
                avg_chunk_size = _compute_avg_chunk_size(chunks)
                top_k, top_n = _derive_retrieval_params(avg_chunk_size)

                logger.info(
                    "avg_chunk_size=%.1f chars | top_n=%d | top_k=%d",
                    avg_chunk_size, top_n, top_k,
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

                generator = get_class(online_cfg["generator"])()

                online_pipeline = OnlinePipeline(query_processor, retriever, reranker, generator)

                # ----------------------------------------------------------
                # Phase 5: Run online queries
                # ----------------------------------------------------------
                qa_online_results = online_pipeline.multiple_queries(qa_pairs)

                # Attach results to QA pairs (deep copy to avoid mutating the template)
                qa_pairs = copy.deepcopy(qa_pairs_template)

                for i, pipeline_result in enumerate(qa_online_results):
                    qa_pairs[i]["response"] = pipeline_result.generation_result
                    qa_pairs[i]["retrieved_contexts"] = [
                        r.chunk.text for r in pipeline_result.reranked_results
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
                eval_df = evaluator.evaluate_retrieval()

                logger.info("Evaluation:\n%s", eval_df.to_string())

                # Aggregate numeric metrics (mean across QA pairs)
                metrics = eval_df.mean(numeric_only=True).to_dict()

                row = {
                    "run_name": run_name,
                    "preprocessors": "+".join(preprocessor_names),
                    "chunker": chunker_name,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "num_chunks": len(chunks),
                    "avg_chunk_size_chars": round(avg_chunk_size, 1),
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