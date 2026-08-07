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
  - Retriever:  FaissRetriever         (TOP_K=9)
  - Reranker:   PassthroughReranker    (TOP_N=3)
  - Generator:  HuggingfaceGenerator
  - Models:     embedding_model and generation_model from config/config.yaml

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
from pathlib import Path

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


# Online components (hardcoded, except model names which come from config)
INDEX_BUILDER_NAME  = "FaissIndexBuilder"
RETRIEVER_NAME      = "FaissRetriever"
RERANKER_NAME       = "PassthroughReranker"
GENERATOR_NAME      = "HuggingfaceGenerator"

# ---------------------------------------------------------------------------
# All (preprocessor, chunker) combinations to evaluate.
#
# Each entry is a tuple:
#   (preprocessing_label, preprocessor_names, chunker_label, chunker_name,
#    chunker_kwargs, top_k, top_n, reranking_threshold)
#
# The top_k / top_n / reranking_threshold values are the best-performing
# settings for each pair as determined by context_exp_summary.csv.
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS: list[tuple[str, list[str], str, str, dict, int, int, float]] = [
    # markdown preprocessor
    # best threshold=0.05 (score 0.627), top_k=39, top_n=13
    # chunker best: S=2, O=0 → acc 0.6274
    ("markdown", ["GeminiMarkdownProcessor"], "paragraph",  "FixedParagraphChunker",     {"chunk_size": 2, "overlap": 0}, 39, 13, 0.05),
    # best threshold=0.1  (score 0.635), top_k=15, top_n=5
    # chunker best: S=1500, O=0 → acc 0.6811
    ("markdown", ["GeminiMarkdownProcessor"], "character",  "FixedCharacterChunker",      {"chunk_size": 1500, "overlap": 0}, 15,  5, 0.1),
    # best threshold=0.0  (score 0.640), top_k=45, top_n=15
    ("markdown", ["GeminiMarkdownProcessor"], "wholetable", "WholeTableParagraphChunker", {}, 45, 15, 0.0),
    # best threshold=0.0  (score 0.649), top_k=111, top_n=37
    ("markdown", ["GeminiMarkdownProcessor"], "splittable", "SplitTableParagraphChunker", {}, 111, 37, 0.0),
    # direct preprocessor (GeminiMarkdown → DirectLLM)
    # best threshold=0.05 (score 0.597), top_k=9,  top_n=3
    # chunker best: S=1, O=0 → acc 0.6605
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "paragraph",  "FixedParagraphChunker", {"chunk_size": 1, "overlap": 0},  9,  3, 0.05),
    # best threshold=0.0  (score 0.668), top_k=9,  top_n=3
    # chunker best: S=150, O=15 → acc 0.6386
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "dynamic",    "DynamicTokenChunker",   {"chunk_size": 150, "overlap": 15},  9,  3, 0.0),
    # best threshold=0.0  (score 0.560), top_k=18, top_n=6
    # chunker best: S=500, O=50 → acc 0.6184
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "character",  "FixedCharacterChunker", {"chunk_size": 500, "overlap": 50}, 18,  6, 0.0),
    # best threshold=0.05 (score 0.502), top_k=3,  top_n=1
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "llmchunker", "LumberChunker",         {},  3,  1, 0.05),
    # best threshold=0.0  (score 0.396), top_k=3,  top_n=1
    # chunker best: τ=0.75, c=1.3 → acc 0.5861
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "maxmin",     "MaxMinChunker",         {"fixed_threshold": 0.75, "c": 1.3},  3,  1, 0.0),
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

    # Load QA evaluation dataset
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    summary_rows: list[dict] = []

    # ======================================================================
    # Outer loop: (preprocessing_config × chunker_config)
    # Each combination builds its offline index once; the three query
    # processors then share that index.
    # ======================================================================
    for (
        preprocessing_label,
        preprocessor_names,
        chunker_label,
        chunker_name,
        chunker_kwargs,
        top_k,
        top_n,
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
        logger.info(
            "Offline index built for '%s'. %d chunk(s) produced.",
            config_label,
            len(offline_result.chunks),
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

    # ------------------------------------------------------------------
    # Write summary CSV
    # ------------------------------------------------------------------
    if summary_rows:
        summary_path = RESULTS_DIR / "query_exp_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Experiment complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed.")


if __name__ == "__main__":
    query_experiment()
