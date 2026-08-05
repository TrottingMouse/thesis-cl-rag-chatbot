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
    paragraph  → FixedParagraphChunker  (chunk_size=1, overlap=0)
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

# Paragraph chunker settings (chunk_size=1 ≈ one paragraph per chunk)
PARA_CHUNK_SIZE = 1
PARA_OVERLAP    = 0

# Online components (hardcoded, except model names which come from config)
INDEX_BUILDER_NAME  = "FaissIndexBuilder"
RETRIEVER_NAME      = "FaissRetriever"
RERANKER_NAME       = "PassthroughReranker"
GENERATOR_NAME      = "HuggingfaceGenerator"
TOP_K               = 9
TOP_N               = 3
RERANKING_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# All (preprocessor, chunker) combinations to evaluate.
#
# Each entry is a tuple:
#   (preprocessing_label, preprocessor_names, chunker_label, chunker_name, chunker_kwargs)
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS: list[tuple[str, list[str], str, str, dict]] = [
    # markdown preprocessor
    ("markdown", ["GeminiMarkdownProcessor"], "paragraph",  "FixedParagraphChunker",      {"chunk_size": PARA_CHUNK_SIZE, "overlap": PARA_OVERLAP}),
    ("markdown", ["GeminiMarkdownProcessor"], "character",  "FixedCharacterChunker",       {}),
    ("markdown", ["GeminiMarkdownProcessor"], "wholetable", "WholeTableParagraphChunker",  {}),
    ("markdown", ["GeminiMarkdownProcessor"], "splittable", "SplitTableParagraphChunker",  {}),
    # direct preprocessor (GeminiMarkdown → DirectLLM)
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "paragraph",   "FixedParagraphChunker",  {"chunk_size": PARA_CHUNK_SIZE, "overlap": PARA_OVERLAP}),
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "dynamic",     "DynamicTokenChunker",    {}),
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "character",   "FixedCharacterChunker",  {}),
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "llmchunker",  "LumberChunker",          {}),
    ("direct",   ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "maxmin",      "MaxMinChunker",          {}),
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
        top_k=TOP_K,
        top_n=TOP_N,
        generation_model=generation_model,
        reranking_score_threshold=RERANKING_THRESHOLD,
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
        "top_k":               TOP_K,
        "top_n":               TOP_N,
        "reranking_threshold": RERANKING_THRESHOLD,
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
