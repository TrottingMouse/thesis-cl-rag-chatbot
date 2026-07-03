import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
    
from factory import build_pipelines_from_config
from src.evaluation import Evaluator
import logging
import json
import os
from dotenv import load_dotenv
from factory import build_pipelines_from_config

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

def main():
    # 1. Build everything from the config file
    offline_pipeline, online_pipeline, data_config, pipeline_name = build_pipelines_from_config("config.yaml")

    # 2. Run Offline Pipeline
    document_paths = data_config["documents"]
    offline_result = offline_pipeline.run(document_paths)

    # 3. Run Online Pipeline
    positive_eval_file = "storage/evaluation/qa_pairs_grid.json"
    with open(positive_eval_file) as f:
        positive_qa_pairs = json.load(f)

    positive_queries = [item["user_input"] for item in positive_qa_pairs]

    positive_online_results = online_pipeline.batch_query(positive_queries)

    for i, pipeline_result in enumerate(positive_online_results):
        positive_qa_pairs[i]["response"] = pipeline_result.generation_result
        positive_qa_pairs[i]["retrieved_contexts"] = [retrieval_result.chunk.text for retrieval_result in pipeline_result.reranked_results]
    
    positive_save_path = "storage/results/" + data_config["online_config"]["generation_model"] + ".json"
    os.makedirs(os.path.dirname(positive_save_path), exist_ok=True)
    with open(positive_save_path, "w") as f:
        json.dump(positive_qa_pairs, f, indent=4)
    logging.info(f"Offline pipeline completed. Results saved to {positive_save_path}")

    
    
    
    

        
        
        

if __name__ == "__main__":
    main()