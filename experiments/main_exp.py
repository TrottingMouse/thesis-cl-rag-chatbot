"""
Main experiment.

For every (preprocessor × chunker) combination that was evaluated in
query_exp.py this script runs a single pipeline – the one whose query
processor produced the highest AnswerCorrectness score in
storage/query_exp_results/query_exp_summary.csv – against the *full*
evaluation set (storage/evaluation/qa_pairs.json).

Additionally it measures **negative rejection** for each pipeline by
running the negative-question set (storage/evaluation/negative_qa_pairs.json)
through the same pipeline and calling Evaluator.evaluate_rejection().

Pipeline configurations (preprocessor chain, chunker, chunker params,
retrieval params, reranking threshold) are taken verbatim from query_exp.py.

Results are written to storage/main_exp_results/:
  <run_name>.json            – raw QA pairs (positive set)
  <run_name>_negative.json   – raw QA pairs (negative set)

A summary CSV is written to storage/main_exp_results/main_exp_summary.csv.

Best query processors determined from query_exp_summary.csv
(evaluated 2026-08-09):
  markdown  + paragraph   → HyDEQueryProcessor
  markdown  + character   → NoProcessingProcessor
  markdown  + wholetable  → CoTQueryProcessor
  markdown  + splittable  → NoProcessingProcessor
  direct    + paragraph   → CoTQueryProcessor
  direct    + dynamic     → NoProcessingProcessor
  direct    + character   → CoTQueryProcessor
  direct    + llmchunker  → NoProcessingProcessor
  direct    + maxmin      → CoTQueryProcessor
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
# Experiment parameters
# ---------------------------------------------------------------------------

QA_EVAL_FILE          = "storage/evaluation/qa_pairs.json"
NEGATIVE_QA_EVAL_FILE = "storage/evaluation/negative_qa_pairs.json"
RESULTS_DIR           = Path("storage/main_exp_results")
INDEX_BASE            = Path("storage/main_exp_index")

TOKEN_LIMIT = 1500          # target context-window budget in tokens

INDEX_BUILDER_NAME = "FaissIndexBuilder"

# ---------------------------------------------------------------------------
# Helpers (identical to query_exp.py / context_exp.py)
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
# Pipeline configurations
#
# Each entry is a tuple:
#   (preprocessing_label, preprocessor_names, chunker_label, chunker_name,
#    chunker_kwargs, reranking_threshold, best_query_processor_name)
#
# The best_query_processor_name is taken from query_exp_summary.csv:
# the query processor with highest AnswerCorrectness for this combo.
#
# top_k and top_n are re-derived dynamically from average chunk token size.
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS: list[tuple[str, list[str], str, str, dict, float, str]] = [
    # markdown preprocessor
    # best query processor: HyDEQueryProcessor (score 0.5893)
    (
        "markdown", ["GeminiMarkdownProcessor"], "paragraph",
        "FixedParagraphChunker", {"chunk_size": 2, "overlap": 0},
        0.0, "HyDEQueryProcessor",
    ),
    # best query processor: NoProcessingProcessor (score 0.6895)
    (
        "markdown", ["GeminiMarkdownProcessor"], "character",
        "FixedCharacterChunker", {"chunk_size": 1500, "overlap": 0},
        0.0, "NoProcessingProcessor",
    ),
    # best query processor: CoTQueryProcessor (score 0.6470)
    (
        "markdown", ["GeminiMarkdownProcessor"], "wholetable",
        "WholeTableParagraphChunker", {},
        0.05, "CoTQueryProcessor",
    ),
    # best query processor: NoProcessingProcessor (score 0.5513)
    (
        "markdown", ["GeminiMarkdownProcessor"], "splittable",
        "SplitTableParagraphChunker", {},
        0.1, "NoProcessingProcessor",
    ),
    # direct preprocessor (GeminiMarkdown → DirectLLM)
    # best query processor: CoTQueryProcessor (score 0.6461)
    (
        "direct", ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "paragraph",
        "FixedParagraphChunker", {"chunk_size": 1, "overlap": 0},
        0.0, "CoTQueryProcessor",
    ),
    # best query processor: NoProcessingProcessor (score 0.6356)
    (
        "direct", ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "dynamic",
        "DynamicTokenChunker", {"chunk_size": 150, "overlap": 15},
        0.05, "NoProcessingProcessor",
    ),
    # best query processor: CoTQueryProcessor (score 0.6775)
    (
        "direct", ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "character",
        "FixedCharacterChunker", {"chunk_size": 500, "overlap": 50},
        0.0, "CoTQueryProcessor",
    ),
    # best query processor: NoProcessingProcessor (score 0.5070)
    (
        "direct", ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "llmchunker",
        "LumberChunker", {},
        0.05, "NoProcessingProcessor",
    ),
    # best query processor: CoTQueryProcessor (score 0.5955)
    (
        "direct", ["GeminiMarkdownProcessor", "DirectLLMProcessor"], "maxmin",
        "MaxMinChunker", {"fixed_threshold": 0.75, "c": 1.3},
        0.05, "CoTQueryProcessor",
    ),
]


# ---------------------------------------------------------------------------
# Per-run execution helper
# ---------------------------------------------------------------------------

def run_pipeline(
    query_processor_name: str,
    run_name: str,
    preprocessing_label: str,
    chunker_label: str,
    chunker_kwargs: dict,
    offline_pipeline,
    generation_model: str,
    queries: list[str],
    qa_pairs_template: list[dict],
    negative_queries: list[str],
    negative_qa_pairs_template: list[dict],
    top_k: int,
    top_n: int,
    reranking_threshold: float,
) -> dict:
    """Build one online pipeline and run both the positive and negative sets.

    Returns a dict with evaluation metrics merged from both evaluations,
    plus bookkeeping columns.
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

    # ------------------------------------------------------------------
    # Positive evaluation (AnswerCorrectness on full QA set)
    # ------------------------------------------------------------------
    logger.info("Running positive queries for '%s' …", run_name)
    qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pos_save = RESULTS_DIR / f"{run_name}.json"
    with open(pos_save, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info("Positive QA results saved to '%s'.", pos_save)

    evaluator = Evaluator(str(pos_save))
    eval_df = evaluator.evaluate()
    pos_metrics = eval_df.mean(numeric_only=True).to_dict()
    logger.info("Positive metrics for '%s': %s", run_name, pos_metrics)

    # ------------------------------------------------------------------
    # Negative rejection evaluation
    # ------------------------------------------------------------------
    logger.info("Running negative queries for '%s' …", run_name)
    neg_qa_pairs = run_queries(online_pipeline, negative_queries, negative_qa_pairs_template)

    neg_save = RESULTS_DIR / f"{run_name}_negative.json"
    with open(neg_save, "w", encoding="utf-8") as f:
        json.dump(neg_qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info("Negative QA results saved to '%s'.", neg_save)

    neg_evaluator = Evaluator(str(neg_save))
    neg_df = neg_evaluator.evaluate_rejection()
    neg_metrics = {f"negative_{k}": v for k, v in neg_df.mean(numeric_only=True).to_dict().items()}
    logger.info("Negative metrics for '%s': %s", run_name, neg_metrics)

    # ------------------------------------------------------------------
    # Assemble summary row
    # ------------------------------------------------------------------
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
        **pos_metrics,
        **neg_metrics,
    }

    logger.info("Run '%s' complete.", run_name)
    return row


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main_experiment() -> None:
    base_cfg = load_yaml_config("config/config.yaml")
    document_paths: list[str] = base_cfg["documents"]

    offline_config = OfflineConfig(**base_cfg.get("offline_config", {}))
    online_config  = OnlineConfig(**base_cfg.get("online_config", {}))

    embedding_model:  str = offline_config.embedding_model
    generation_model: str = online_config.generation_model

    logger.info("Embedding model:   %s", embedding_model)
    logger.info("Generation model:  %s", generation_model)

    logger.info("Loading tokenizer for '%s' …", generation_model)
    tokenizer = AutoTokenizer.from_pretrained(generation_model)

    # Load positive QA evaluation dataset (full set)
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d positive queries from '%s'.", len(queries), QA_EVAL_FILE)

    # Load negative QA dataset
    with open(NEGATIVE_QA_EVAL_FILE) as f:
        negative_qa_pairs_template = json.load(f)
    negative_queries: list[str] = [item["user_input"] for item in negative_qa_pairs_template]
    logger.info(
        "Loaded %d negative queries from '%s'.",
        len(negative_queries), NEGATIVE_QA_EVAL_FILE,
    )

    summary_rows: list[dict] = []

    for (
        preprocessing_label,
        preprocessor_names,
        chunker_label,
        chunker_name,
        chunker_kwargs,
        reranking_threshold,
        query_processor_name,
    ) in PIPELINE_CONFIGS:

        config_label = f"{preprocessing_label}__{chunker_label}"
        run_name     = f"{config_label}__{query_processor_name.replace('QueryProcessor', '').replace('Processor', '').lower()}"

        logger.info("=" * 70)
        logger.info(
            "CONFIG: %s  (preprocessors=%s, chunker=%s, kwargs=%s, processor=%s)",
            config_label, preprocessor_names, chunker_name, chunker_kwargs,
            query_processor_name,
        )
        logger.info("=" * 70)

        # Build the offline index once per (preprocessing, chunker) config
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
            negative_queries=negative_queries,
            negative_qa_pairs_template=negative_qa_pairs_template,
            top_k=top_k,
            top_n=top_n,
            reranking_threshold=reranking_threshold,
        )
        summary_rows.append(row)

    # ------------------------------------------------------------------
    # Write summary CSV
    # ------------------------------------------------------------------
    if summary_rows:
        summary_path = RESULTS_DIR / "main_exp_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Experiment complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed.")


if __name__ == "__main__":
    main_experiment()
