"""
Query processor experiment.

For every query processor (NoProcessingProcessor, HyDEQueryProcessor,
CoTQueryProcessor), this script:
  1. Loads the queries from qa_pairs_grid.json.
  2. Runs each query through the processor's .process() method.
  3. Saves the results (original query + processed/expanded queries) to
     storage/query_exp_results/expanded_queries.json for manual inspection.

No evaluation is performed at this stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.factory import load_yaml_config
from src.online.query.processors import (
    NoProcessingProcessor,
    HyDEQueryProcessor,
    CoTQueryProcessor,
)

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
# Constants
# ---------------------------------------------------------------------------

QA_EVAL_FILE = "storage/evaluation/qa_pairs_grid.json"
RESULTS_DIR = Path("storage/query_exp_results")

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def query_expansion_experiment() -> None:
    # Load base config to get the generation model name
    base_cfg = load_yaml_config("config/config.yaml")
    generation_model: str = base_cfg["online_config"]["generation_model"]
    logger.info("Using generation model: %s", generation_model)

    # Load QA pairs
    with open(QA_EVAL_FILE) as f:
        qa_pairs_template = json.load(f)
    queries: list[str] = [item["user_input"] for item in qa_pairs_template]
    logger.info("Loaded %d queries from '%s'.", len(queries), QA_EVAL_FILE)

    # Instantiate all processors
    processors = [
        NoProcessingProcessor(),
        HyDEQueryProcessor(model_name=generation_model),
        CoTQueryProcessor(model_name=generation_model),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run each processor over all queries and collect results
    all_results: dict[str, list[dict]] = {}

    for processor in processors:
        logger.info("=== Running processor: %s ===", processor.name)
        processor_results: list[dict] = []

        for query in queries:
            logger.info("  Processing: %r", query)
            augmented = processor.process(query)
            processor_results.append(
                {
                    "original_query": augmented.original_query,
                    "processed_queries": augmented.processed_queries,
                    "query_type": augmented.query_type,
                }
            )

        all_results[processor.name] = processor_results
        logger.info("  Done. %d queries processed.", len(processor_results))

    # Save combined results
    output_path = RESULTS_DIR / "expanded_queries.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    logger.info("Results saved to '%s'.", output_path)


if __name__ == "__main__":
    query_expansion_experiment()
