"""
Query processor experiment.

Compares three query processors
  - NoProcessingProcessor
  - HyDEQueryProcessor
  - CoTQueryProcessor
across all combinations of the following preprocessor and chunker
configurations:

  Preprocessors:
    markdown   → GeminiMarkdownProcessor
    direct     → GeminiMarkdownProcessor + DirectLLMProcessor

  Chunkers (applied per preprocessor as listed in PIPELINE_CONFIGS):
    paragraph  → FixedParagraphChunker  (optimal params per preprocessor, see PIPELINE_CONFIGS)
    character  → FixedCharacterChunker
    wholetable → WholeTableParagraphChunker   (markdown only)
    splittable → SplitTableParagraphChunker   (markdown only)
    dynamic    → DynamicTokenChunker          (direct only)
    llmchunker → LumberChunker                (direct only)
    maxmin     → MaxMinChunker                (direct only)

For each (preprocessor × chunker) combination the offline index is built
once and then shared across the three query-processor runs.

All other pipeline components are hardcoded:
  - Index:      FaissIndexBuilder
  - Retriever:  FaissRetriever         (top_k derived dynamically)
  - Generator:  HuggingfaceGenerator
  - Models:     embedding_model and generation_model from config/config.yaml

top_k and top_n are derived per (preprocessor, chunker) config from the
average chunk size in tokens, using a 1500-token budget:
  top_n = max(1, floor(1500 / avg_chunk_tokens))
  top_k = 3 * top_n

For each of the 9×3 = 27 runs the script:
  1. Reuses the offline index built for that (preprocessor, chunker) config.
  2. Runs all queries from qa_pairs_grid.json.
  3. Persists raw QA pairs to storage/query_exp_results/<run_name>.json.
  4. Evaluates with evaluate_minimal() and collects the mean metrics.

A summary CSV is written to storage/query_exp_results/query_exp_summary.csv.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from transformers import AutoTokenizer

from dotenv import load_dotenv

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
# Hardcoded experiment parameters
# ---------------------------------------------------------------------------

QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"
RESULTS_DIR  = Path("storage/query_exp_results")
INDEX_BASE   = Path("storage/query_exp_index")

TOKEN_LIMIT = 1500          # target context-window budget in tokens (matches context_exp)


# Online components (hardcoded, except model names which come from config)
INDEX_BUILDER_NAME  = "FaissIndexBuilder"


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


# ---------------------------------------------------------------------------
# All (preprocessor, chunker) combinations to evaluate.
#
# Each entry is a tuple:
#   (preprocessing_label, preprocessor_names, chunker_label, chunker_name,
#    chunker_kwargs, reranking_threshold)
#
# top_k and top_n are derived dynamically from the average chunk token size
# after the offline index is built (TOKEN_LIMIT = 1500 tokens).
# reranking_threshold is the best value from context_exp_summary.csv.
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS: list[tuple[str, list[str], str, str, dict, float]] = [
    # markdown preprocessor
    # best threshold=0.00 (score 0.627) | chunker best: S=2, O=0 → acc 0.6274
    ("markdown", ["GeminiMarkdownProcessor"], "paragraph",  "FixedParagraphChunker",     {"chunk_size": 2, "overlap": 0},       0.0),
    # best threshold=0.00 (score 0.649) | chunker best: S=1500, O=0 → acc 0.6811
    ("markdown", ["GeminiMarkdownProcessor"], "character",  "FixedCharacterChunker",      {"chunk_size": 1500, "overlap": 0},    0.0),
    # best threshold=0.05 (score 0.647)
    ("markdown", ["GeminiMarkdownProcessor"], "wholetable", "WholeTableParagraphChunker", {},                                    0.05),
    # best threshold=0.10 (score 0.591)
    ("markdown", ["GeminiMarkdownProcessor"], "splittable", "SplitTableParagraphChunker", {},                                    0.1),
    # direct preprocessor (GeminiMarkdown → DirectLLM)
    # best threshold=0.00 (score 0.598) | chunker best: S=1, O=0 → acc 0.6605
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "paragraph",  "FixedParagraphChunker", {"chunk_size": 1, "overlap": 0},         0.0),
    # best threshold=0.05 (score 0.639) | chunker best: S=150, O=15 → acc 0.6386
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "dynamic",    "DynamicTokenChunker",   {"chunk_size": 150, "overlap": 15},       0.05),
    # best threshold=0.00 (score 0.600) | chunker best: S=500, O=50 → acc 0.6184
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "character",  "FixedCharacterChunker", {"chunk_size": 500, "overlap": 50},       0.0),
    # best threshold=0.05 (score 0.526)
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "llmchunker", "LumberChunker",         {},                                        0.05),
    # best threshold=0.05 (score 0.584) | chunker best: τ=0.75, c=1.3 → acc 0.5861
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "maxmin",     "MaxMinChunker",         {"fixed_threshold": 0.75, "c": 1.3},      0.05),
]

# Query processors to compare: (label, registry name)
PROCESSOR_CONFIGS: list[tuple[str, str]] = [
    ("no_processing", "NoProcessingProcessor"),
    ("hyde",          "HyDEQueryProcessor"),
    ("cot",           "CoTQueryProcessor"),
]


# ---------------------------------------------------------------------------
# Helper: run one online pipeline for a given query processor
# ---------------------------------------------------------------------------

def run_pipeline(
    query_processor_name: str,
    run_name: str,
    preprocessing_label: str,
    chunker_label: str,
    chunker_kwargs: dict,
    offline_pipeline,           # already built & populated offline pipeline
    generation_model: str,
    queries: list[str],
    qa_pairs_template: list[dict],
    top_k: int,
    top_n: int,
    reranking_threshold: float,
) -> dict:
    """
    Build and execute one online pipeline for the given query processor.

    The offline pipeline (and its populated index builder) is shared across
    the three query-processor runs for the same (preprocessor, chunker) config.

    Returns a dict with evaluation metrics and bookkeeping columns.
    """
    logger.info("=== Run: %s ===", run_name)

    base_cfg = load_yaml_config("config/config.yaml")
    online_pipeline_cfg_file: dict = base_cfg["online_pipeline"]

    online_pipeline_cfg = {
        "query_processor": query_processor_name,
        "retriever":       online_pipeline_cfg_file["retriever"],
        "reranker":        online_pipeline_cfg_file["reranker"],
        "generator":       online_pipeline_cfg_file["generator"],
    }

    online_pipeline = build_online_pipeline(
        cfg=online_pipeline_cfg,
        index_builder=offline_pipeline.index_builder,
        top_k=top_k,
        top_n=top_n,
        generation_model=generation_model,
        reranking_score_threshold=reranking_threshold,
    )

    qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)
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
        "query_processor":     query_processor_name,
        "chunk_size":          chunker_kwargs.get("chunk_size", ""),
        "overlap":             chunker_kwargs.get("overlap", ""),
        "top_k":               top_k,
        "top_n":               top_n,
        "reranking_threshold": reranking_threshold,
        **metrics,
    }

    logger.info("Run '%s' complete. Metrics: %s", run_name, metrics)
    return row


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def query_experiment() -> None:
    # Load only the model names from config; all component names are hardcoded
    base_cfg = load_yaml_config("config/config.yaml")
    document_paths: list[str] = base_cfg["documents"]

    offline_config = OfflineConfig(**base_cfg.get("offline_config", {}))
    online_config  = OnlineConfig(**base_cfg.get("online_config", {}))

    embedding_model:  str = offline_config.embedding_model
    generation_model: str = online_config.generation_model

    logger.info("Embedding model:   %s", embedding_model)
    logger.info("Generation model:  %s", generation_model)

    # Load the generator tokenizer once — chunk sizes are measured in its
    # token space so that top_n/top_k mirror what the model actually sees.
    logger.info("Loading tokenizer for '%s' …", generation_model)
    tokenizer = AutoTokenizer.from_pretrained(generation_model)

    # Load QA evaluation dataset
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    summary_rows: list[dict] = []

    # Outer loop: (preprocessing_config × chunker_config)
    # Each combination builds its offline index once; the three query
    # processors then share that index.
    for (
        preprocessing_label,
        preprocessor_names,
        chunker_label,
        chunker_name,
        chunker_kwargs,
        reranking_threshold,
    ) in PIPELINE_CONFIGS:

        config_label = f"{preprocessing_label}__{chunker_label}"
        logger.info("=" * 70)
        logger.info(
            "CONFIG: %s  (preprocessors=%s, chunker=%s, kwargs=%s)",
            config_label, preprocessor_names, chunker_name, chunker_kwargs,
        )
        logger.info("=" * 70)

        # Build the offline pipeline once per (preprocessing, chunker) config
        offline_pipeline = build_offline_pipeline(
            preprocessor_names=preprocessor_names,
            chunker_name=chunker_name,
            index_builder_name=INDEX_BUILDER_NAME,
            storage_path=INDEX_BASE / config_label,
            embedding_model=embedding_model,
            **chunker_kwargs,
        )
        offline_result = offline_pipeline.run(document_paths)
        chunks = offline_result.chunks
        logger.info(
            "Offline index built for '%s'. %d chunk(s) produced.",
            config_label,
            len(chunks),
        )

        # Derive retrieval parameters from average chunk size (1500-token budget)
        avg_tokens = _compute_avg_chunk_tokens(chunks, tokenizer)
        top_k, top_n = _derive_retrieval_params(avg_tokens)
        logger.info(
            "avg_chunk_tokens=%.1f → top_n=%d, top_k=%d",
            avg_tokens, top_n, top_k,
        )

        # Inner loop: query processors — all share the index built above
        logger.info(
            "Comparing query processors: %s",
            [name for _, name in PROCESSOR_CONFIGS],
        )

        for processor_label, query_processor_name in PROCESSOR_CONFIGS:
            run_name = f"{config_label}__{processor_label}"
            row = run_pipeline(
                query_processor_name=query_processor_name,
                run_name=run_name,
                preprocessing_label=preprocessing_label,
                chunker_label=chunker_label,
                chunker_kwargs=chunker_kwargs,
                offline_pipeline=offline_pipeline,
                generation_model=generation_model,
                queries=queries,
                qa_pairs_template=qa_pairs_template,
                top_k=top_k,
                top_n=top_n,
                reranking_threshold=reranking_threshold,
            )
            summary_rows.append(row)

    if summary_rows:
        summary_path = RESULTS_DIR / "query_exp_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Experiment complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed.")


if __name__ == "__main__":
    query_experiment()
