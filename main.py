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
    eval_file = data_config["evaluation_file"]
    with open(eval_file) as f:
        qa_pairs = json.load(f)

    queries = [item["user_input"] for item in qa_pairs]
    online_results = online_pipeline.multiple_queries(queries)

    for i, pipeline_result in enumerate(online_results):
        qa_pairs[i]["generated_answer"] = pipeline_result.generation_result
        qa_pairs[i]["retrieved_contexts"] = [retrieval_result.chunk.text for retrieval_result in pipeline_result.reranked_results]
    
    save_path = "storage/results/" + pipeline_name + ".json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)
    logging.info(f"Offline pipeline completed. Results saved to {save_path}")

    evaluator = Evaluator(save_path)
    evaluation_df = evaluator.evaluate(accept=True)
    print(evaluation_df)
    
    
    
    

        
        
        

if __name__ == "__main__":
    main()