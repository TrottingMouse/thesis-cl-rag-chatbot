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
    offline_pipeline, online_pipeline, config, pipeline_name = build_pipelines_from_config("config.yaml")

    # 2. Run Offline Pipeline
    document_paths = config["documents"]
    offline_result = offline_pipeline.run(document_paths)

    # 3. Run Online Pipeline
    positive_eval_file = "storage/evaluation/qa_pairs.json"
    negative_eval_file = "storage/evaluation/negative_qa_pairs.json"
    with open(positive_eval_file) as f:
        positive_qa_pairs = json.load(f)
    with open(negative_eval_file) as f:
        negative_qa_pairs = json.load(f)

    positive_queries = [item["user_input"] for item in positive_qa_pairs]
    negative_queries = [item["user_input"] for item in negative_qa_pairs]

    positive_online_results = online_pipeline.multiple_queries(positive_queries)
    negative_online_results = online_pipeline.multiple_queries(negative_queries)

    for i, pipeline_result in enumerate(positive_online_results):
        positive_qa_pairs[i]["response"] = pipeline_result.generation_result
        positive_qa_pairs[i]["retrieved_contexts"] = [chunk.text for chunk in pipeline_result.reranked_results]
    for i, pipeline_result in enumerate(negative_online_results):
        negative_qa_pairs[i]["response"] = pipeline_result.generation_result
        negative_qa_pairs[i]["retrieved_contexts"] = [chunk.text for chunk in pipeline_result.reranked_results]
    
    positive_save_path = "storage/results/positive/" + pipeline_name + ".json"
    negative_save_path = "storage/results/negative/" + pipeline_name + ".json"
    os.makedirs(os.path.dirname(positive_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(negative_save_path), exist_ok=True)
    with open(positive_save_path, "w") as f:
        json.dump(positive_qa_pairs, f, indent=4)
    with open(negative_save_path, "w") as f:
        json.dump(negative_qa_pairs, f, indent=4)
    logging.info(f"Offline pipeline completed. Results saved to {positive_save_path} and {negative_save_path}")

    evaluator_positive = Evaluator(positive_save_path)
    evaluation_df_positive = evaluator_positive.evaluate()
    print(evaluation_df_positive)
    evaluator_negative = Evaluator(negative_save_path)
    evaluation_df_negative = evaluator_negative.evaluate_rejection()
    print(evaluation_df_negative)
    
    
    
    

        
        
        

if __name__ == "__main__":
    main()