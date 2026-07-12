"""
This script evaluates the performance of two choices for preprocessing.
1. DoclingMarkdownPreprocessor vs GeminiMarkdownPreprocessor
2. DirectLLMPreprocessor vs PaperLLMPreprocessor
For 2., the best performing preprocessor from 1. is used. The metric here is answer accuracy.
It uses the ParagraphChunker with chunk size 1 and overlap 0. Top_k is 9 and top_n is 3.
The dataset used is qa_pairs_grid.json.
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
# Fixed experiment parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1
OVERLAP = 0
TOP_K = 9
TOP_N = 3
QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"
RESULTS_DIR = Path("storage/preprocessing_exp_results")
INDEX_BASE = Path("storage/preprocessing_exp_index")


# ---------------------------------------------------------------------------
# Helper: build and run one complete offline+online pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    preprocessor_names: list[str],
    run_name: str,
    document_paths: list[str],
    online_pipeline_cfg: dict,
    offline_config: OfflineConfig,
    online_config: OnlineConfig,
    queries: list[str],
    qa_pairs_template: list[dict],
) -> dict:
    """
    Build and execute one offline+online pipeline for the given preprocessors.

    Returns a dict with evaluation metrics and bookkeeping columns.
    """
    logger.info("=== Run: %s ===", run_name)

    # Build and run the offline pipeline
    offline_pipeline = build_offline_pipeline(
        preprocessor_names=preprocessor_names,
        chunker_name="FixedParagraphChunker",
        index_builder_name="FaissIndexBuilder",
        storage_path=INDEX_BASE / run_name,
        embedding_model=offline_config.embedding_model,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    offline_result = offline_pipeline.run(document_paths)
    chunks = offline_result.chunks
    logger.info("Produced %d chunk(s).", len(chunks))

    # Build and run the online pipeline, reusing the populated index builder
    online_pipeline = build_online_pipeline(
        cfg=online_pipeline_cfg,
        index_builder=offline_pipeline.index_builder,
        top_k=TOP_K,
        top_n=TOP_N,
        generation_model=online_config.generation_model,
    )
    qa_pairs = run_queries(online_pipeline, queries, qa_pairs_template)

    # Persist raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qa_save = RESULTS_DIR / f"{run_name}.json"
    with open(qa_save, "w") as f:
        json.dump(qa_pairs, f, indent=4)
    logger.info("Raw QA results saved to '%s'.", qa_save)

    # Evaluate
    evaluator = Evaluator(str(qa_save))
    eval_df = evaluator.evaluate()
    metrics = eval_df.mean(numeric_only=True).to_dict()

    row = {
        "run_name": run_name,
        "preprocessors": "+".join(preprocessor_names),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k": TOP_K,
        "top_n": TOP_N,
        "num_chunks": len(chunks),
        **metrics,
    }

    logger.info("Run '%s' complete. Metrics: %s", run_name, metrics)
    return row


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def preprocessing_experiment():
    # Load base config
    base_cfg = load_yaml_config("config/config.yaml")
    document_paths: list[str] = base_cfg["documents"]
    online_pipeline_cfg: dict = base_cfg["online_pipeline"]

    offline_config = OfflineConfig(**base_cfg.get("offline_config", {}))
    online_config = OnlineConfig(**base_cfg.get("online_config", {}))

    # Load QA evaluation dataset
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries = [item["user_input"] for item in qa_pairs_template]

    summary_rows: list[dict] = []

    # ======================================================================
    # PHASE 1 – Compare markdown preprocessors
    #   Run A: DoclingMarkdownProcessor alone
    #   Run B: GeminiMarkdownProcessor alone
    # ======================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: DoclingMarkdownProcessor vs GeminiMarkdownProcessor")
    logger.info("=" * 70)

    phase1_configs = [
        ("docling_only", ["DoclingMarkdownProcessor"]),
        ("gemini_only",  ["GeminiMarkdownProcessor"]),
    ]

    phase1_results: dict[str, dict] = {}

    for run_name, preprocessor_names in phase1_configs:
        row = run_pipeline(
            preprocessor_names=preprocessor_names,
            run_name=run_name,
            document_paths=document_paths,
            online_pipeline_cfg=online_pipeline_cfg,
            offline_config=offline_config,
            online_config=online_config,
            queries=queries,
            qa_pairs_template=qa_pairs_template,
        )
        row["phase"] = 1
        summary_rows.append(row)
        phase1_results[run_name] = row

    # Determine Phase 1 winner by answer_correctness
    accuracy_key = "answer_correctness"
    best_phase1_name = max(
        phase1_results,
        key=lambda k: float(phase1_results[k].get(accuracy_key, 0.0)),
    )
    best_phase1_preprocessors = dict(phase1_configs)[best_phase1_name]

    logger.info(
        "Phase 1 winner: '%s' (preprocessors=%s, %s=%.4f)",
        best_phase1_name,
        best_phase1_preprocessors,
        accuracy_key,
        float(phase1_results[best_phase1_name].get(accuracy_key, 0.0)),
    )

    # ======================================================================
    # PHASE 2 – Compare LLM post-processors
    #   Best markdown preprocessor from Phase 1 is prepended to both runs.
    #   Run C: <best_phase1> + DirectLLMProcessor
    #   Run D: <best_phase1> + PaperLLMProcessor
    # ======================================================================
    logger.info("=" * 70)
    logger.info(
        "PHASE 2: DirectLLMProcessor vs PaperLLMProcessor (base: %s)",
        best_phase1_preprocessors,
    )
    logger.info("=" * 70)

    phase2_configs = [
        (
            f"{best_phase1_name}_direct_llm",
            best_phase1_preprocessors + ["DirectLLMProcessor"],
        ),
        (
            f"{best_phase1_name}_paper_llm",
            best_phase1_preprocessors + ["PaperLLMProcessor"],
        ),
    ]

    for run_name, preprocessor_names in phase2_configs:
        row = run_pipeline(
            preprocessor_names=preprocessor_names,
            run_name=run_name,
            document_paths=document_paths,
            online_pipeline_cfg=online_pipeline_cfg,
            offline_config=offline_config,
            online_config=online_config,
            queries=queries,
            qa_pairs_template=qa_pairs_template,
        )
        row["phase"] = 2
        summary_rows.append(row)

    # Write summary CSV
    if summary_rows:
        summary_path = RESULTS_DIR / "preprocessing_exp_summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("Experiment complete. Summary written to '%s'.", summary_path)
    else:
        logger.warning("No runs were completed.")


if __name__ == "__main__":
    preprocessing_experiment()