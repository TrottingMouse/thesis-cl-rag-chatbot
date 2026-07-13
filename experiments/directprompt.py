"""
This script evaluates the performance of GeminiMarkdownProcessor chained with
DirectLLMProcessor (i.e. the same preprocessor pair used as the best-phase2
candidate in preprocessing_exp, but run in isolation here).

It uses the same fixed configuration as preprocessing_exp:
  - Chunker:  FixedParagraphChunker  (chunk_size=1, overlap=0)
  - Retrieval: top_k=9, top_n=3
  - Dataset:  qa_pairs_grid.json

The single run converts PDFs to markdown via GeminiMarkdownProcessor and then
passes the result through DirectLLMProcessor before chunking and indexing.
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
# Fixed experiment parameters  (identical to preprocessing_exp)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1
OVERLAP = 0
TOP_K = 9
TOP_N = 3
QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"
RESULTS_DIR = Path("storage/directprompt_results")
INDEX_BASE = Path("storage/directprompt_index")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def directprompt_experiment():
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

    run_name = "gemini_direct_llm"
    preprocessor_names = ["GeminiMarkdownProcessor", "DirectLLMProcessor"]

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

    # Write summary CSV
    summary_path = RESULTS_DIR / "directprompt_summary.csv"
    write_summary_csv(summary_path, [row])
    logger.info("Experiment complete. Summary written to '%s'.", summary_path)


if __name__ == "__main__":
    directprompt_experiment()
