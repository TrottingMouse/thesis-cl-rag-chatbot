import yaml
from pathlib import Path
from registry import get_class
from src.config import OfflineConfig, OnlineConfig
from src.offline.pipeline import OfflinePipeline
from src.online.pipeline import OnlinePipeline

def load_yaml_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def build_pipelines_from_config(yaml_path: str):
    config = load_yaml_config(yaml_path)
    
    # 1. Initialize configs with overrides from YAML
    # If the section doesn't exist in YAML, it returns {}, falling back to your Python defaults.
    offline_kwargs = config.get("offline_config", {})
    online_kwargs = config.get("online_config", {})
    
    offline_config = OfflineConfig(**offline_kwargs)
    online_config = OnlineConfig(**online_kwargs)

    # 2. Build Offline Pipeline
    offline_cfg = config["offline_pipeline"]
    
    preprocessors = [get_class(name)() for name in offline_cfg["preprocessors"]]
    chunker = get_class(offline_cfg["chunker"])()
    
    IndexBuilderClass = get_class(offline_cfg["index_builder"])
    index_builder = IndexBuilderClass(
        storage_path=Path(config["data"]["index_path"]),
        model_name=offline_config.embedding_model # Uses the overridden value!
    )
    
    offline_pipeline = OfflinePipeline(preprocessors, chunker, index_builder)

    # 3. Build Online Pipeline
    online_cfg = config["online_pipeline"]
    
    query_processor = get_class(online_cfg["query_processor"])()
    
    RetrieverClass = get_class(online_cfg["retriever"])
    retriever = RetrieverClass(index_builder, top_k=online_config.top_k) # Uses overridden top_k
    
    RerankerClass = get_class(online_cfg["reranker"])
    reranker = RerankerClass(top_n=online_config.top_n) # Uses overridden top_n
    
    generator = get_class(online_cfg["generator"])()
    
    online_pipeline = OnlinePipeline(query_processor, retriever, reranker, generator)

    return offline_pipeline, online_pipeline, config["data"]