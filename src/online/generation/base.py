"""
Abstract base class for generators (Online Pipeline – Step 4).

A generator receives the reranked context chunks and the original
(or augmented) query, and returns the final answer text.

Possible other implementations
----------------------------------
* ``PassthroughGenerator``     – concatenates the top chunks without any
                                  LLM call (baseline / ablation)
* ``OpenAIGenerator``          – calls the OpenAI Chat Completions API
* ``OllamaGenerator``          – calls a locally running Ollama model

"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import AugmentedQuery, Chunk, RetrievalResult


class BaseGenerator(ABC):
    """
    Contract for all generators.

    Subclasses **must** implement :meth:`generate`.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this generator.

        Example: ``'openai_gpt4o'``.
        """

    @abstractmethod
    def generate(
        self,
        augmented_query: AugmentedQuery,
        context: list[RetrievalResult],
    ) -> str:
        """
        Generate an answer given the query and retrieved context.

        Parameters
        ----------
        augmented_query:
            The processed query.  Implementations typically use
            ``augmented_query.original_query`` as the question text.
        context:
            Reranked results that form the context window.
            ``context[0]`` is the most relevant result (rank 1).

        Returns
        -------
        str
            The generated answer text.
        """
    def construct_prompt(self, query: str, context: list[RetrievalResult]) -> str:
        """
        Constructs the prompt for the generator.

        Parameters
        ----------
        query:
            The original query.
        context:
            The retrieved chunks.

        Returns
        -------
        str
            The constructed prompt.
        """
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        return f"""
        Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext.
        Wenn die Antwort nicht im Kontext zu finden ist, antworte: "Dazu enthalten die bereitgestellten Dokumente keine Informationen."
        
        Frage: {query}
        Kontext: {context_str}
        
        Antwort:
        """

    def generate_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        contexts: list[list[RetrievalResult]],
    ) -> list[str]:
        """
        Generate answers for a batch of queries.

        The default implementation calls :meth:`generate` for each query
        sequentially.  Subclasses that support true batched inference
        (e.g. :class:`HuggingfaceGenerator`) should override this method
        for better throughput.

        Parameters
        ----------
        augmented_queries:
            One per input query.
        contexts:
            One reranked context list per query, in the same order as
            ``augmented_queries``.

        Returns
        -------
        list[str]
            One answer string per input query, in the same order.
        """
        return [
            self.generate(aq, ctx)
            for aq, ctx in zip(augmented_queries, contexts)
        ]
