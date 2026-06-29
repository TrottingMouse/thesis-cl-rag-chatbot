import json
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy, answer_correctness, AspectCritic
# 2. LangChain Module und Ragas Wrapper importieren
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.run_config import RunConfig
from dotenv import load_dotenv
import os


class Evaluator:
    """
    Evaluates the RAG system.
    Inputs: a jsonl file with questions, gold answers, generated answers, retrieved chunks and metadata
    Outputs: metrics for evaluation
    """
    def __init__(self, filepath):
        # read data from json file
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        # 3. Gemini via LangChain initialisieren
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # 4. Die LangChain-Objekte für Ragas verpacken
        self.ragas_llm = LangchainLLMWrapper(gemini_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)
        answer_relevancy.strictness = 1
        

    def evaluate(self, accept: bool):
        eval_dataset = EvaluationDataset.from_list(self.data)

        my_run_config = RunConfig(max_workers=1, timeout=120, max_retries=10)

        if accept:
            result = evaluate(
                dataset=eval_dataset,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                metrics=[context_recall, context_precision, faithfulness, answer_relevancy, answer_correctness],
                raise_exceptions=True,
                run_config=my_run_config
            )
        else:
            negative_rejection = AspectCritic(
                name="negative_rejection",
                definition="Did the model reject the query as not answerable from the given context?",
                llm=self.ragas_llm,
            )
            result = evaluate(
                dataset=eval_dataset,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                metrics=[negative_rejection]
            )

        print(result)
        df = result.to_pandas()
        return df
        


# load_dotenv()
# eval = Evaluator("storage/negative_example.jsonl")
# eval.evaluate(accept=False)
# evaluator = Evaluator("storage/queryeval_example.jsonl")
# evaluator.evaluate(accept=True)
""",
  "metadata": {
    "experiment_id": "run_001",
    "chunking_strategy": "LlamaParse_Semantic",
    "preprocessing": "Query_Rewrite_v2",
    "timestamp": "2026-06-03T14:30:00"
  }"""