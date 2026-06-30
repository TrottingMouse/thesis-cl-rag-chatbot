import json
import pandas as pd
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.run_config import RunConfig

# 1. Bring back the LangChain wrappers to satisfy the legacy embeddings requirement
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 2. Import metrics from the base 'ragas.metrics' module to bypass the evaluate() bug
# Notice these are the capitalized Class names, which we will initialize below.
from ragas.metrics import (
    ContextRecall,
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    AspectCritic
)

class Evaluator:
    """
    Evaluates the RAG system.
    Inputs: a jsonl file with questions, gold answers, generated answers, retrieved chunks and metadata
    Outputs: metrics for evaluation
    """
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 3. Initialize OpenAI via LangChain (with our massive timeouts to prevent crashes)
        openai_llm = ChatOpenAI(model="gpt-4o-mini", timeout=120, max_retries=10)
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", timeout=120, max_retries=10)

        # 4. Wrap them for Ragas
        self.ragas_llm = LangchainLLMWrapper(openai_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(openai_embeddings)

    def evaluate(self, accept: bool):
        eval_dataset = EvaluationDataset.from_list(self.data)
        
        # 5. Keep the massive timeout so rate-limit backoffs succeed
        my_run_config = RunConfig(max_workers=2, timeout=1200, max_retries=10)

        if accept:
            # 6. Initialize the metric objects and pass the LLM/Embeddings
            ans_rel = AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
            ans_rel.strictness = 1
            
            metrics = [
                ContextRecall(llm=self.ragas_llm),
                ContextPrecision(llm=self.ragas_llm),
                Faithfulness(llm=self.ragas_llm),
                ans_rel,
                AnswerCorrectness(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
            ]
            
            result = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                raise_exceptions=False, 
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
                metrics=[negative_rejection],
                raise_exceptions=False,
                run_config=my_run_config
            )

        print(result)
        return result.to_pandas()
        


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