"""
Query processor experiment.

Compares three query processors
  - NoProcessingProcessor
  - HyDEQueryProcessor
  - CoTQueryProcessor
across two preprocessing configurations:
  A. GeminiMarkdownProcessor only
  B. GeminiMarkdownProcessor + DirectLLMProcessor

All other pipeline components are hardcoded:
  - Chunker:    FixedParagraphChunker  (CHUNK_SIZE=1, OVERLAP=0)
  - Index:      FaissIndexBuilder
  - Retriever:  FaissRetriever         (TOP_K=9)
  - Reranker:   PassthroughReranker    (TOP_N=3)
  - Generator:  HuggingfaceGenerator
  - Models:     embedding_model and generation_model from config/config.yaml

For each of the 6 runs the script:
  1. Reuses the offline index built for that preprocessing config.
  2. Runs all queries from qa_pairs_grid.json.
  3. Persists raw QA pairs to storage/query_exp_results/<run_name>.json.
  4. Evaluates with RAGAS and collects the mean metrics.

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

# Offline components (hardcoded)
CHUNKER_NAME       = "FixedParagraphChunker"
INDEX_BUILDER_NAME = "FaissIndexBuilder"
CHUNK_SIZE         = 1
OVERLAP            = 0

# Online components (hardcoded, except model names which come from config)
RETRIEVER_NAME = "FaissRetriever"
RERANKER_NAME  = "PassthroughReranker"
GENERATOR_NAME = "HuggingfaceGenerator"
TOP_K          = 9
TOP_N          = 3
RERANKING_THRESHOLD = 0.1

# Preprocessing configurations: (label, ordered list of preprocessor registry names)
PREPROCESSING_CONFIGS: list[tuple[str, list[str]]] = [
    ("gemini",            ["GeminiMarkdownProcessor"]),
    ("gemini_direct_llm", ["GeminiMarkdownProcessor", "DirectLLMProcessor"]),
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
    offline_pipeline,           # already built & populated offline pipeline
    generation_model: str,
    queries: list[str],
    qa_pairs_template: list[dict],
) -> dict:
    """
    Build and execute one online pipeline for the given query processor.

    The offline pipeline (and its populated index builder) is shared across
    the three query-processor runs for the same preprocessing config.

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

    # Evaluate
    evaluator = Evaluator(str(qa_save))
    eval_df = evaluator.evaluate()
    metrics = eval_df.mean(numeric_only=True).to_dict()

    row = {
        "run_name":            run_name,
        "preprocessing":       preprocessing_label,
        "query_processor":     query_processor_name,
        "chunk_size":          CHUNK_SIZE,
        "overlap":             OVERLAP,
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

    embedding_model:   str = offline_config.embedding_model
    generation_model:  str = online_config.generation_model

    logger.info("Embedding model:   %s", embedding_model)
    logger.info("Generation model:  %s", generation_model)

    # Load QA evaluation dataset
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    summary_rows: list[dict] = []

    # ======================================================================
    # Outer loop: preprocessing configurations
    # Each config gets its own offline index built once.
    # ======================================================================
    for preprocessing_label, preprocessor_names in PREPROCESSING_CONFIGS:
        logger.info("=" * 70)
        logger.info(
            "PREPROCESSING CONFIG: %s  (preprocessors=%s)",
            preprocessing_label,
            preprocessor_names,
        )
        logger.info("=" * 70)

        # Build the offline pipeline once per preprocessing config
        offline_pipeline = build_offline_pipeline(
            preprocessor_names=preprocessor_names,
            chunker_name=CHUNKER_NAME,
            index_builder_name=INDEX_BUILDER_NAME,
            storage_path=INDEX_BASE / preprocessing_label,
            embedding_model=embedding_model,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )
        offline_result = offline_pipeline.run(document_paths)
        logger.info(
            "Offline index built for '%s'. %d chunk(s) produced.",
            preprocessing_label,
            len(offline_result.chunks),
        )

        # Inner loop: query processors — all share the index built above
        logger.info(
            "Comparing query processors: %s",
            [name for _, name in PROCESSOR_CONFIGS],
        )

        for processor_label, query_processor_name in PROCESSOR_CONFIGS:
            run_name = f"{preprocessing_label}__{processor_label}"
            row = run_pipeline(
                query_processor_name=query_processor_name,
                run_name=run_name,
                preprocessing_label=preprocessing_label,
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
