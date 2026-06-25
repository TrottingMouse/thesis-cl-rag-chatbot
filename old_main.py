from src.offline.chunking import FixedParagraphChunker, FixedSentenceChunker, FixedCharacterChunker
from src.offline.preprocessing import DirectLLMProcessor, GeminiMarkdownProcessor, PaperLLMProcessor, RawTextProcessor
from src.offline.indexing import FaissIndexBuilder
from src.offline.pipeline import OfflinePipeline
from src.online.reranking import PassthroughReranker, Qwen3Reranker
from src.online.generation import PassthroughGenerator
from src.online.query import NoProcessingProcessor
from src.online.retrieval import FaissRetriever
from src.online.pipeline import OnlinePipeline, OnlinePipelineResult
from src.config import OnlineConfig, OfflineConfig

import logging
from pathlib import Path
from dotenv import load_dotenv
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()



offline_config = OfflineConfig()
online_config = OnlineConfig()

#processor = RawTextProcessor()
markdownprocessor = GeminiMarkdownProcessor()
llmprocessor = DirectLLMProcessor()
chunker = FixedParagraphChunker()
index_builder = FaissIndexBuilder(storage_path=Path("storage/index"), model_name=offline_config.embedding_model)

offline = OfflinePipeline([markdownprocessor, llmprocessor], chunker, index_builder)

offline_result = offline.run(["documents/PO.pdf", "documents/MHB.pdf"])

query_processor = NoProcessingProcessor()
dense_retriever = FaissRetriever(index_builder, top_k=online_config.top_k)
# reranker = Qwen3Reranker(top_n=online_config.top_n)
reranker = PassthroughReranker(top_n=online_config.top_n)
passthrough_generator = PassthroughGenerator()

online: OnlinePipeline = OnlinePipeline(query_processor, dense_retriever, reranker, passthrough_generator)

queries = json.load(open("storage/evaluation/eval_init.jsonl"))
online_result: OnlinePipelineResult = online.query("Welche Kurse sollte ich im ersten Semester belegen?")
print(online_result.generation_result)





