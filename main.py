from src.online.generation.generators import PassthroughGenerator
from src.online.reranking.rerankers import Qwen3Reranker
from src.online.query.processors import NoProcessingProcessor
from src.online.retrieval.retrievers import DenseRetriever
from src.config import OnlineConfig
import logging
from pathlib import Path
from src.config import OfflineConfig
from src.offline.indexing.indexing import FaissIndexBuilder
from src.offline.chunking.fixed_size import FixedCharacterChunker
from src.offline.preprocessing.file_conversions import RawTextProcessor
from src.offline.pipeline import OfflinePipeline
from src.online.pipeline import OnlinePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


offline_config = OfflineConfig()
online_config = OnlineConfig()

processor = RawTextProcessor()
chunker = FixedCharacterChunker()
index_builder = FaissIndexBuilder(storage_path=Path("storage/index"), model_name=offline_config.embedding_model)

offline = OfflinePipeline(processor, chunker, index_builder)

offline_result = offline.run(["documents/PO.pdf", "documents/MHB.pdf"])

query_processor = NoProcessingProcessor()
dense_retriever = DenseRetriever(index_builder, top_k=online_config.top_k)
reranker = Qwen3Reranker(top_n=online_config.top_n)
passthrough_generator = PassthroughGenerator()

online = OnlinePipeline(query_processor, dense_retriever, reranker, passthrough_generator)





