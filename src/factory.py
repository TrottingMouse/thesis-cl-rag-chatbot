import yaml
from pathlib import Path
from src.registry import get_class
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
    offline_components = config["offline_pipeline"]
    
    preprocessors = [get_class(name)() for name in offline_components["preprocessors"]]

    # Forward all chunker-specific parameters from the config dict.
    ChunkerClass = get_class(offline_components["chunker"])
    chunker = ChunkerClass(**offline_config.chunking_params)
    
    # Index builders that embed chunks require an embedding model; passthrough does not.
    IndexBuilderClass = get_class(offline_components["index_builder"])
    if offline_components["index_builder"] == "PassthroughIndexBuilder":
        index_builder = IndexBuilderClass(storage_path=Path("storage/index"))
    else:
        #hier bei BM25 aufpassen weil kein model_name als parameter (Konstruktor überschrieben bei dense)
        index_builder = IndexBuilderClass(
            storage_path=Path("storage/index"),
            model_name=offline_config.embedding_model
        )
    
    offline_pipeline = OfflinePipeline(preprocessors, chunker, index_builder)

    # 3. Build Online Pipeline
    online_components = config["online_pipeline"]
    
    query_processor = get_class(online_components["query_processor"])()
    
    RetrieverClass = get_class(online_components["retriever"])
    if online_components["retriever"] == "PassthroughRetriever":
        retriever = RetrieverClass(index_builder)
    else:
        retriever = RetrieverClass(index_builder, top_k=online_config.top_k) # Uses overridden top_k
    
    RerankerClass = get_class(online_components["reranker"])
    reranker = RerankerClass(top_n=online_config.top_n) # Uses overridden top_n

    # Only init with model if generator 
    GeneratorClass = get_class(online_components["generator"])
    if online_components["generator"] == "HuggingfaceGenerator":
        generator = GeneratorClass(model_name=online_config.generation_model)
    else:
        generator = GeneratorClass()
    
    online_pipeline = OnlinePipeline(query_processor, retriever, reranker, generator)

    # 4. Build a unique pipeline name from the first 6 letters of each component
    component_names = (
        offline_components["preprocessors"]          # list of preprocessor names
        + [offline_components["chunker"]]
        + [offline_components["index_builder"]]
        + [online_components["query_processor"]]
        + [online_components["retriever"]]
        + [online_components["reranker"]]
        + [online_components["generator"]]
    )
    pipeline_name = "_".join(name[:6] for name in component_names)

    return offline_pipeline, online_pipeline, config, pipeline_name