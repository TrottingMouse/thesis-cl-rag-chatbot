import itertools
import copy
import json
import logging
from factory import load_yaml_config, build_pipelines_from_config # or load_xml_config

logging.basicConfig(level=logging.INFO)

# Define the grid of parameters you want to test
SEARCH_SPACE = {
    "top_k": [5, 10, 15],
    "top_n": [3, 5],
    "chunker": ["FixedParagraphChunker", "FixedCharacterChunker"]
}

def run_grid_search(base_config_path: str):
    # 1. Load the base configuration
    base_config = load_yaml_config(base_config_path)
    
    # 2. Extract keys and generate all combinations of values
    keys = list(SEARCH_SPACE.keys())
    value_lists = list(SEARCH_SPACE.values())
    combinations = list(itertools.product(*value_lists))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    best_score = 0
    best_params = None
    all_results = []

    # 3. Iterate through every combination
    for combo in combinations:
        # Create a dictionary mapping the key to the current combination's value
        params = dict(zip(keys, combo))
        logging.info(f"Testing parameters: {params}")
        
        # Create a fresh copy of the base config for this specific run
        current_config = copy.deepcopy(base_config)
        
        # Inject the grid parameters into the config dictionary
        # Note: Ensure the target dictionaries exist first (e.g., online_config)
        if "online_config" not in current_config:
            current_config["online_config"] = {}
            
        current_config["online_config"]["top_k"] = params["top_k"]
        current_config["online_config"]["top_n"] = params["top_n"]
        current_config["offline_pipeline"]["chunker"] = params["chunker"]

        # 4. Build pipelines using the modified dictionary
        # (You will need to slightly adapt your factory to accept a dict instead of a filepath)
        offline_pipeline, online_pipeline, data_config = build_pipelines_from_dict(current_config)

        # 5. Execute pipelines
        offline_pipeline.run(data_config["documents"])
        
        with open(data_config["evaluation_file"]) as f:
            qa_pairs = json.load(f)
        queries = [item["user_input"] for item in qa_pairs]
        
        online_results = online_pipeline.multiple_queries(queries)
        
        # 6. Evaluate (You'll need a custom evaluation function here)
        # e.g., score = calculate_mrr(online_results, qa_pairs)
        score = evaluate_results(online_results, qa_pairs) 
        
        all_results.append({
            "parameters": params,
            "score": score
        })
        
        # Track the best performing setup
        if score > best_score:
            best_score = score
            best_params = params

    print(f"\nGrid Search Complete!")
    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")
    
    return all_results