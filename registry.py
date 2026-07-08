from src.minirag_pipeline import MiniRAGPipeline
from src.offline.chunking import WholeTableParagraphChunker
from src.online.retrieval import PassthroughRetriever
from src.offline.indexing import PassthroughIndexBuilder
from src.offline.chunking import PassthroughChunker
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
    "PassthroughChunker": PassthroughChunker,
    "WholeTableParagraphChunker": WholeTableParagraphChunker,
    
    # Index Builders
    "FaissIndexBuilder": FaissIndexBuilder,
    "PassthroughIndexBuilder": PassthroughIndexBuilder,
    
    # Online Components
    "NoProcessingProcessor": NoProcessingProcessor,
    "FaissRetriever": FaissRetriever,
    "PassthroughRetriever": PassthroughRetriever,
    "PassthroughReranker": PassthroughReranker,
    "Qwen3Reranker": Qwen3Reranker,
    "PassthroughGenerator": PassthroughGenerator,
    "HuggingfaceGenerator": HuggingfaceGenerator,

    # MiniRAG adapter (whole-pipeline, not a single component)
    "MiniRAGPipeline": MiniRAGPipeline,
}

def get_class(name: str):
    if name not in COMPONENT_REGISTRY:
        raise ValueError(f"Component '{name}' not found in registry.")
    return COMPONENT_REGISTRY[name]