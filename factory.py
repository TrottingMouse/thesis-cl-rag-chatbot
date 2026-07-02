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
    
    #hier bei BM25 aufpassen weil kein model_name als parameter (Konstruktor überschrieben bei dense)
    IndexBuilderClass = get_class(offline_cfg["index_builder"])
    index_builder = IndexBuilderClass(
        storage_path=Path("storage/index"),
        model_name=offline_config.embedding_model
    )
    
    offline_pipeline = OfflinePipeline(preprocessors, chunker, index_builder)

    # 3. Build Online Pipeline
    online_cfg = config["online_pipeline"]
    
    query_processor = get_class(online_cfg["query_processor"])()
    
    RetrieverClass = get_class(online_cfg["retriever"])
    retriever = RetrieverClass(index_builder, top_k=online_config.top_k) # Uses overridden top_k
    
    RerankerClass = get_class(online_cfg["reranker"])
    reranker = RerankerClass(top_n=online_config.top_n) # Uses overridden top_n

    # Only init with model if generator 
    GeneratorClass = get_class(online_cfg["generator"])
    if online_cfg["generator"] == "HuggingfaceGenerator":
        generator = GeneratorClass(model_name=online_config.generation_model)
    else:
        generator = GeneratorClass()
    
    online_pipeline = OnlinePipeline(query_processor, retriever, reranker, generator)

    # 4. Build a unique pipeline name from the first 6 letters of each component
    component_names = (
        offline_cfg["preprocessors"]          # list of preprocessor names
        + [offline_cfg["chunker"]]
        + [offline_cfg["index_builder"]]
        + [online_cfg["query_processor"]]
        + [online_cfg["retriever"]]
        + [online_cfg["reranker"]]
        + [online_cfg["generator"]]
    )
    pipeline_name = "_".join(name[:6] for name in component_names)

    return offline_pipeline, online_pipeline, config, pipeline_name