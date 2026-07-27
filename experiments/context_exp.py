"""
Context size experiment.

Investigates two strategies for setting the number of retrieved / reranked
documents fed to the LLM:

Strategy A – Fixed token cap
    top_n = max(1, floor(2000 / avg_chunk_size_tokens))
    top_k = 3 * top_n

    The token limit of 2000 is chosen because SLMs around 8 B parameters
    already show degraded performance between 4000–8000 context tokens;
    for the 2 B model used here the budget is set to less than half of that.

Strategy B – Similarity threshold
    Same hard top_n as in Strategy A, but chunks whose cross-encoder score
    falls below a configurable threshold are dropped before generation.
    This lets chunkers with small chunks be selective: if a query only needs
    a few chunks the pipeline can use fewer tokens without a fixed hard cut.

The experiment is run across four configurations that cover both
document-representation styles and granularity levels:

  1. Markdown,    FixedParagraphChunker  (chunk_size=1, overlap=0)
  2. Linearized,  FixedParagraphChunker  (chunk_size=1, overlap=0)
  3. Markdown,    SplitTableParagraphChunker
  4. Linearized,  SplitTableParagraphChunker

"Linearized" means GeminiMarkdownProcessor + DirectLLMProcessor (converts
markdown to dense, entity-rich prose).

For Strategy B the thresholds [0.0, 0.1, 0.2, 0.3, 0.4] are tested.
Threshold 0.0 is equivalent to Strategy A (no filtering).

Results are persisted per run and collected in a summary CSV.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

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
# Fixed experiment parameters
# ---------------------------------------------------------------------------

TOKEN_LIMIT = 2000          # target context-window budget in tokens
QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"
RESULTS_DIR  = Path("storage/context_exp_results")
INDEX_BASE   = Path("storage/context_exp_index")

# Paragraph chunker settings (chunk_size=1 ≈ one paragraph per chunk)
PARA_CHUNK_SIZE = 1
PARA_OVERLAP    = 0

# Similarity thresholds tested for Strategy B.
# 0.0 is included as a baseline (equivalent to Strategy A with no filtering).
THRESHOLDS = [0.0, 0.05, 0.1]

# Preprocessing configurations:
#   "markdown"    → GeminiMarkdownProcessor only
#   "linearized"  → GeminiMarkdownProcessor + DirectLLMProcessor
PREPROCESSING_CONFIGS: list[tuple[str, list[str]]] = [
    ("markdown",   ["GeminiMarkdownProcessor"]),
    ("linearized", ["GeminiMarkdownProcessor", "DirectLLMProcessor"]),
]

# Chunker configurations:
#   "paragraph"   → FixedParagraphChunker (chunk_size=1, overlap=0)
#   "split_table" → SplitTableParagraphChunker (default params)
CHUNKER_CONFIGS: list[tuple[str, str, dict]] = [
    ("paragraph",   "FixedParagraphChunker",       {"chunk_size": PARA_CHUNK_SIZE, "overlap": PARA_OVERLAP}),
    ("split_table", "SplitTableParagraphChunker",  {}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_avg_chunk_tokens(chunks, tokenizer: AutoTokenizer) -> float:
    """Return the mean token length across *chunks*."""
    if not chunks:
        return 1.0
    total = sum(len(tokenizer.encode(c.text, add_special_tokens=False)) for c in chunks)
    return total / len(chunks)


def _derive_retrieval_params(avg_tokens: float) -> tuple[int, int]:
    """Derive top_n and top_k from the average chunk size in tokens.

    top_n = max(1, floor(TOKEN_LIMIT / avg_tokens))
    top_k = 3 * top_n
    """
    top_n = max(1, math.floor(TOKEN_LIMIT / avg_tokens))
    top_k = 3 * top_n
    return top_k, top_n


def _run_pipeline_with_threshold(
    *,
    run_name: str,
    offline_pipeline,
    top_k: int,
    top_n: int,
    threshold: float,
    online_pipeline_cfg: dict,
    generation_model: str,
    reranking_score_threshold_override: float,
    queries: list[str],
    qa_pairs_template: list[dict],
    avg_tokens: float,
    num_chunks: int,
    chunker_label: str,
    preprocessing_label: str,
    extra_cols: dict,
) -> dict:
    """Build, run and evaluate one online pipeline, returning a summary row."""
    logger.info("--- Run: %s (threshold=%.2f) ---", run_name, threshold)

    online_pipeline = build_online_pipeline(
        cfg=online_pipeline_cfg,
        index_builder=offline_pipeline.index_builder,
        top_k=top_k,
        top_n=top_n,
        generation_model=generation_model,
        reranking_score_threshold=reranking_score_threshold_override,
    )

    qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)

    # Persist raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qa_save = RESULTS_DIR / f"{run_name}.json"
    with open(qa_save, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info("Raw QA results saved to '%s'.", qa_save)

    # Evaluate (minimal set: AnswerCorrectness only)
    evaluator = Evaluator(str(qa_save))
    eval_df = evaluator.evaluate_minimal()
    metrics = eval_df.mean(numeric_only=True).to_dict()

    row = {
        "run_name":            run_name,
        "preprocessing":       preprocessing_label,
        "chunker":             chunker_label,
        "num_chunks":          num_chunks,
        "avg_chunk_tokens":    round(avg_tokens, 1),
        "top_k":               top_k,
        "top_n":               top_n,
        "reranking_threshold": threshold,
        **extra_cols,
        **metrics,
    }
    logger.info("Run '%s' complete. Metrics: %s", run_name, metrics)
    return row


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def context_experiment() -> None:
    # Load configs
    base_cfg = load_yaml_config("config/config.yaml")
    document_paths: list[str] = base_cfg["documents"]
    online_pipeline_cfg: dict = base_cfg["online_pipeline"]

    offline_config = OfflineConfig(**base_cfg.get("offline_config", {}))
    online_config  = OnlineConfig(**base_cfg.get("online_config", {}))

    embedding_model  = offline_config.embedding_model
    generation_model = online_config.generation_model

    # Load the generator tokenizer once — chunk sizes are measured in its
    # token space so that top_n/top_k mirror what the model actually sees.
    logger.info("Loading tokenizer for '%s' …", generation_model)
    tokenizer = AutoTokenizer.from_pretrained(generation_model)

    # QA evaluation dataset
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    summary_rows: list[dict] = []

    # ======================================================================
    # Outer loop: (preprocessing_config × chunker_config)
    # Each combination builds its offline index once, then all threshold
    # variants share that index.
    # ======================================================================
    for preprocessing_label, preprocessor_names in PREPROCESSING_CONFIGS:
        for chunker_label, chunker_name, chunker_kwargs in CHUNKER_CONFIGS:

            config_label = f"{preprocessing_label}__{chunker_label}"
            logger.info("=" * 70)
            logger.info(
                "CONFIG: %s  (preprocessors=%s, chunker=%s, kwargs=%s)",
                config_label, preprocessor_names, chunker_name, chunker_kwargs,
            )
            logger.info("=" * 70)

            # Build offline pipeline once per (preprocessing, chunker) pair
            offline_pipeline = build_offline_pipeline(
                preprocessor_names=preprocessor_names,
                chunker_name=chunker_name,
                index_builder_name="FaissIndexBuilder",
                storage_path=INDEX_BASE / config_label,
                embedding_model=embedding_model,
                **chunker_kwargs,
            )
            offline_result = offline_pipeline.run(document_paths)
            chunks = offline_result.chunks
            num_chunks = len(chunks)
            logger.info("Offline index built. %d chunk(s) produced.", num_chunks)

            # Derive fixed retrieval parameters from average chunk size
            avg_tokens = _compute_avg_chunk_tokens(chunks, tokenizer)
            top_k, top_n = _derive_retrieval_params(avg_tokens)
            logger.info(
                "avg_chunk_tokens=%.1f → top_n=%d, top_k=%d",
                avg_tokens, top_n, top_k,
            )

            # Common metadata for every threshold run in this config
            extra_cols = {
                "chunk_size": chunker_kwargs.get("chunk_size", ""),
                "overlap":    chunker_kwargs.get("overlap", ""),
            }

            # ------------------------------------------------------------------
            # Inner loop: similarity thresholds (Strategy B)
            # Threshold 0.0 is equivalent to Strategy A (no filtering).
            # ------------------------------------------------------------------
            for threshold in THRESHOLDS:
                run_name = f"{config_label}__thr{str(threshold).replace('.', '')}"

                row = _run_pipeline_with_threshold(
                    run_name=run_name,
                    offline_pipeline=offline_pipeline,
                    top_k=top_k,
                    top_n=top_n,
                    threshold=threshold,
                    online_pipeline_cfg=online_pipeline_cfg,
                    generation_model=generation_model,
                    reranking_score_threshold_override=threshold,
                    queries=queries,
                    qa_pairs_template=qa_pairs_template,
                    avg_tokens=avg_tokens,
                    num_chunks=num_chunks,
                    chunker_label=chunker_label,
                    preprocessing_label=preprocessing_label,
                    extra_cols=extra_cols,
                )
                summary_rows.append(row)

    # Write summary CSV
    if summary_rows:
        summary_path = RESULTS_DIR / "context_exp_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Experiment complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed.")


if __name__ == "__main__":
    context_experiment()
