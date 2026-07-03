from src.online.generation import HuggingfaceGenerator
from src.offline.chunking.fixed_size import FixedParagraphChunker, FixedCharacterChunker, DynamicTokenChunker
from src.offline.chunking.llm import LumberChunker
from src.offline.chunking.semantic import MaxMinChunker
from src.offline.preprocessing.llm_processors import DirectLLMProcessor, PaperLLMProcessor
from src.offline.preprocessing.preprocessors import GeminiMarkdownProcessor, RawTextProcessor
from src.offline.indexing.indexing import FaissIndexBuilder
from src.online.reranking.rerankers import Qwen3Reranker
from src.online.reranking import PassthroughReranker
from src.online.generation.generators import PassthroughGenerator
from src.online.query.processors import NoProcessingProcessor
from src.online.retrieval.retrievers import FaissRetriever

COMPONENT_REGISTRY = {
    # Preprocessors
    "RawTextProcessor": RawTextProcessor,
    "GeminiMarkdownProcessor": GeminiMarkdownProcessor,
    "DirectLLMProcessor": DirectLLMProcessor,
    "PaperLLMProcessor": PaperLLMProcessor,
    
    # Chunkers
    "FixedParagraphChunker": FixedParagraphChunker,
    "FixedCharacterChunker": FixedCharacterChunker,
    "LumberChunker": LumberChunker,
    "MaxMinChunker": MaxMinChunker,
    "DynamicTokenChunker": DynamicTokenChunker,
    
    # Index Builders
    "FaissIndexBuilder": FaissIndexBuilder,
    
    # Online Components
    "NoProcessingProcessor": NoProcessingProcessor,
    "FaissRetriever": FaissRetriever,
    "PassthroughReranker": PassthroughReranker,
    "Qwen3Reranker": Qwen3Reranker,
    "PassthroughGenerator": PassthroughGenerator,
    "HuggingfaceGenerator": HuggingfaceGenerator,
}

def get_class(name: str):
    if name not in COMPONENT_REGISTRY:
        raise ValueError(f"Component '{name}' not found in registry.")
    return COMPONENT_REGISTRY[name]