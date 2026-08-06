"""
Grid search for optimal chunking parameters.

All experiment parameters are hardcoded below — no external YAML config is
needed.  The script covers four chunkers:

  FixedParagraphChunker
    chunk_size ∈ {1, 2, 3}
    overlap    : 1 for sizes 1 and 2; 0 for size 3

  FixedCharacterChunker
    chunk_size ∈ {500, 1_000, 1_500}
    overlap    : 0 or 10 % of chunk_size (rounded)

  DynamicTokenChunker
    chunk_size ∈ {150, 250, 500}
    overlap    : 0 or 10 % of chunk_size (rounded)

  MaxMinChunker
    fixed_threshold ∈ {0.6, 0.7, 0.8}   (≥ default; thematically homogenous corpus)
    c               ∈ {0.8, 0.9, 0.95}  (≥ default)

For every run the script:
  1. Builds the offline index for the given chunker + params.
  2. Derives retrieval parameters dynamically from avg chunk size:
       top_n = floor(2000 / avg_chunk_size_tokens)  (min 1)
       top_k = 3 * top_n
  3. Runs all queries from qa_pairs_grid.json.
  4. Evaluates with evaluate_minimal() and collects mean metrics.

A summary CSV is written to storage/grid_search_results/grid_search_summary.csv.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.factory import (
    build_offline_pipeline,
    build_online_pipeline,
    run_queries,
    write_summary_csv,
)
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
# Hardcoded experiment parameters
# ---------------------------------------------------------------------------

# Documents to index
DOCUMENT_PATHS: list[str] = [
    "documents/PO.pdf",
    "documents/MHB.pdf",
    "documents/POges.pdf",
]

# Models
EMBEDDING_MODEL  = "jinaai/jina-embeddings-v5-text-nano"
GENERATION_MODEL = "Qwen/Qwen3.5-2B"

# Preprocessor chain (applied to every chunker)
PREPROCESSOR_NAMES: list[str] = ["GeminiMarkdownProcessor"]

# Online components (fixed across all runs)
INDEX_BUILDER_NAME = "FaissIndexBuilder"
RETRIEVER_NAME     = "FaissRetriever"
RERANKER_NAME      = "JinaReranker"
GENERATOR_NAME     = "HuggingfaceGenerator"

# Reranking threshold (fixed; only top_k / top_n vary per run)
RERANKING_THRESHOLD = 0.1

# QA evaluation file
QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"

# Output directories
RESULTS_DIR = Path("storage/grid_search_results")
INDEX_BASE  = Path("storage/grid_search_index")

# ---------------------------------------------------------------------------
# Per-chunker parameter grids
# ---------------------------------------------------------------------------

def _paragraph_configs() -> list[dict]:
    """
    FixedParagraphChunker grid.
      sizes   : 1, 2, 3
      overlap : 1 only for sizes 1 and 2 (overlap must be < chunk_size)
    """
    configs = []
    for size in (1, 2, 3):
        overlaps = [0, 1] if size <= 2 else [0]
        for ov in overlaps:
            configs.append({"chunk_size": size, "overlap": ov})
    return configs


def _character_configs() -> list[dict]:
    """
    FixedCharacterChunker grid.
      sizes   : 500, 1_000, 1_500
      overlap : 0  or  10 % of chunk_size (rounded to nearest int)
    """
    configs = []
    for size in (500, 1_000, 1_500):
        ten_pct = round(size * 0.10)
        for ov in sorted({0, ten_pct}):          # deduplicate if 10 % rounds to 0
            configs.append({"chunk_size": size, "overlap": ov})
    return configs


def _dynamic_configs() -> list[dict]:
    """
    DynamicTokenChunker grid.
      sizes   : 150, 250, 500
      overlap : 0  or  10 % of chunk_size (rounded)
    """
    configs = []
    for size in (150, 250, 500):
        ten_pct = round(size * 0.10)
        for ov in sorted({0, ten_pct}):
            configs.append({"chunk_size": size, "overlap": ov})
    return configs


def _maxmin_configs() -> list[dict]:
    """
    MaxMinChunker grid.
      fixed_threshold : 0.6, 0.7, 0.8  (≥ algorithm default of 0.6)
      c               : 0.8, 0.9, 0.95 (≥ algorithm default of 0.9 — note 0.8
                                         is included to explore a slightly
                                         looser damping that may still outperform
                                         the default on this homogenous corpus)
    """
    configs = []
    for ft, c in itertools.product((0.6, 0.7, 0.8), (0.85, 0.9, 0.95)):
        configs.append({"fixed_threshold": ft, "c": c})
    return configs


# Registry of all chunkers with their grids
#   (label, registry_name, param_configs, extra_kwargs_for_build)
CHUNKER_SPECS: list[tuple[str, str, list[dict], dict]] = [
    ("paragraph", "FixedParagraphChunker",  _paragraph_configs(), {}),
    ("character", "FixedCharacterChunker",  _character_configs(), {}),
    ("dynamic",   "DynamicTokenChunker",    _dynamic_configs(),   {}),
    # MaxMinChunker needs the embedding model injected; factory handles this
    # automatically when chunker_name == "MaxMinChunker" and
    # "embedding_model_name" is absent from chunker_kwargs.
    ("maxmin",    "MaxMinChunker",          _maxmin_configs(),    {}),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_avg_chunk_size(chunks, tokenizer: AutoTokenizer) -> float:
    """Return the average token length of a list of Chunk objects."""
    if not chunks:
        return 1.0
    total = sum(
        len(tokenizer.encode(c.text, add_special_tokens=False)) for c in chunks
    )
    return total / len(chunks)


def _derive_retrieval_params(avg_tokens: float) -> tuple[int, int]:
    """
    Derive top_n and top_k from the average chunk size in tokens.

    top_n = floor(2000 / avg_tokens)   (min 1)
    top_k = 3 * top_n
    """
    top_n = max(1, math.floor(2000.0 / avg_tokens))
    top_k = 3 * top_n
    return top_k, top_n


def _run_name(chunker_label: str, params: dict) -> str:
    """Build a short, filesystem-safe run name."""
    param_part = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{chunker_label}__{param_part}"


# ---------------------------------------------------------------------------
# Online pipeline config dict (mirrors the YAML online_pipeline section)
# ---------------------------------------------------------------------------

ONLINE_PIPELINE_CFG = {
    "query_processor": "NoProcessingProcessor",
    "retriever":       RETRIEVER_NAME,
    "reranker":        RERANKER_NAME,
    "generator":       GENERATOR_NAME,
}

# ---------------------------------------------------------------------------
# Main grid search
# ---------------------------------------------------------------------------

def chunking_grid_search() -> None:
    logger.info("Loading tokenizer for '%s' ...", GENERATION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)

    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for chunker_label, chunker_name, param_grid, _extra in CHUNKER_SPECS:
        logger.info("=" * 70)
        logger.info(
            "CHUNKER: %s (%s)  |  %d parameter combination(s)",
            chunker_label, chunker_name, len(param_grid),
        )
        logger.info("=" * 70)

        for params in param_grid:
            run_name = _run_name(chunker_label, params)
            logger.info("--- Run: %s ---", run_name)

            index_path = INDEX_BASE / run_name

            # ------------------------------------------------------------------
            # Offline pipeline
            # ------------------------------------------------------------------
            offline_pipeline = build_offline_pipeline(
                preprocessor_names=PREPROCESSOR_NAMES,
                chunker_name=chunker_name,
                index_builder_name=INDEX_BUILDER_NAME,
                storage_path=index_path,
                embedding_model=EMBEDDING_MODEL,
                **params,
                # factory auto-injects embedding_model_name for MaxMinChunker
            )
            offline_result = offline_pipeline.run(DOCUMENT_PATHS)
            chunks = offline_result.chunks

            # ------------------------------------------------------------------
            # Derive dynamic retrieval parameters from avg chunk size
            # ------------------------------------------------------------------
            avg_tokens = _compute_avg_chunk_size(chunks, tokenizer)
            top_k, top_n = _derive_retrieval_params(avg_tokens)
            logger.info(
                "  num_chunks=%d | avg_chunk_tokens=%.1f | top_n=%d | top_k=%d",
                len(chunks), avg_tokens, top_n, top_k,
            )

            # ------------------------------------------------------------------
            # Online pipeline + query execution
            # ------------------------------------------------------------------
            online_pipeline = build_online_pipeline(
                cfg=ONLINE_PIPELINE_CFG,
                index_builder=offline_pipeline.index_builder,
                top_k=top_k,
                top_n=top_n,
                generation_model=GENERATION_MODEL,
                reranking_score_threshold=RERANKING_THRESHOLD,
            )
            qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)

            qa_save = RESULTS_DIR / f"{run_name}.json"
            with open(qa_save, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
            logger.info("  Raw QA results saved to '%s'.", qa_save)

            # ------------------------------------------------------------------
            # Evaluation
            # ------------------------------------------------------------------
            evaluator = Evaluator(str(qa_save))
            eval_df   = evaluator.evaluate_minimal()
            metrics   = eval_df.mean(numeric_only=True).to_dict()

            row = {
                "run_name":              run_name,
                "preprocessors":         "+".join(PREPROCESSOR_NAMES),
                "chunker":               chunker_name,
                # MaxMinChunker-specific (empty for other chunkers)
                "fixed_threshold":       params.get("fixed_threshold", ""),
                "c":                     params.get("c", ""),
                # Fixed-size chunker columns (empty for MaxMinChunker)
                "chunk_size":            params.get("chunk_size", ""),
                "overlap":               params.get("overlap", ""),
                # Retrieval info
                "num_chunks":            len(chunks),
                "avg_chunk_tokens":      round(avg_tokens, 1),
                "top_k":                 top_k,
                "top_n":                 top_n,
                "reranking_threshold":   RERANKING_THRESHOLD,
                **metrics,
            }
            summary_rows.append(row)
            logger.info("  Run '%s' complete. Metrics: %s", run_name, metrics)

    # --------------------------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------------------------
    if summary_rows:
        summary_path = RESULTS_DIR / "grid_search_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info(
            "Grid search complete. Summary written to '%s'.", summary_path
        )
    else:
        logger.warning("No runs completed.")


if __name__ == "__main__":
    chunking_grid_search()