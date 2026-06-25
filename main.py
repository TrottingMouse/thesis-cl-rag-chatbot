import logging
import json
from dotenv import load_dotenv
from factory import build_pipelines_from_config

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

def main():
    # 1. Build everything from the config file
    offline_pipeline, online_pipeline, data_config = build_pipelines_from_config("config.yaml")

    # 2. Run Offline Pipeline
    document_paths = data_config["documents"]
    offline_result = offline_pipeline.run(document_paths)

    # 3. Run Online Pipeline
    eval_file = data_config["evaluation_file"]
    with open(eval_file) as f:
        qa_pairs = json.load(f)

    queries = [item["user_input"] for item in qa_pairs]
    online_results = online_pipeline.multiple_queries(queries)

    # Do something with online_results...

if __name__ == "__main__":
    main()