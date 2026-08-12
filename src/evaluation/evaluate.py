import json

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    AspectCritic
)

_RUN_CONFIG = RunConfig(max_workers=2, timeout=1200, max_retries=10)


class Evaluator:
    """
    Evaluates a RAG pipeline using RAGAS metrics.

    Input: a JSON file with questions, gold answers, generated answers,
           retrieved chunks, and metadata.
    Output: a pandas DataFrame with metric scores.
    """

    def __init__(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        openai_llm = ChatOpenAI(model="gpt-4o-mini", timeout=120, max_retries=10)
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", timeout=120, max_retries=10)
        self.ragas_llm = LangchainLLMWrapper(openai_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(openai_embeddings)

    def evaluate(self):
        """Run the full metric suite (ContextRecall, Faithfulness, AnswerRelevancy, AnswerCorrectness)."""
        eval_dataset = EvaluationDataset.from_list(self.data)

        ans_rel = AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
        ans_rel.strictness = 1

        metrics = [
            ContextRecall(llm=self.ragas_llm),
            Faithfulness(llm=self.ragas_llm),
            ans_rel,
            AnswerCorrectness(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
        ]

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            raise_exceptions=False,
            run_config=_RUN_CONFIG,
        )

        print(result)
        return result.to_pandas()

    def evaluate_minimal(self):
        """Lightweight evaluation: only AnswerCorrectness.

        Use in experiment scripts where the full metric suite would be too slow.
        Returns the same DataFrame structure as :meth:`evaluate`.
        """
        eval_dataset = EvaluationDataset.from_list(self.data)

        result = evaluate(
            dataset=eval_dataset,
            metrics=[AnswerCorrectness(llm=self.ragas_llm, embeddings=self.ragas_embeddings)],
            raise_exceptions=False,
            run_config=_RUN_CONFIG,
        )

        print(result)
        return result.to_pandas()

    def evaluate_rejection(self):
        """Evaluate whether the model correctly rejects unanswerable queries."""
        eval_dataset = EvaluationDataset.from_list(self.data)

        negative_rejection = AspectCritic(
            name="negative_rejection",
            definition="Did the model reject the query as not answerable from the given context?",
            llm=self.ragas_llm,
        )

        result = evaluate(
            dataset=eval_dataset,
            metrics=[negative_rejection],
            raise_exceptions=False,
            run_config=_RUN_CONFIG,
        )

        print(result)
        return result.to_pandas()