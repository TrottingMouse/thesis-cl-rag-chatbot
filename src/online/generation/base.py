"""
Abstract base class for generators (Online Pipeline – Step 4).

A generator receives the reranked context chunks and the original
(or augmented) query, and returns a :class:`~src.models.GenerationResult`
that contains the final answer text.

Possible other implementations
----------------------------------
* ``PassthroughGenerator``     – concatenates the top chunks without any
                                  LLM call (baseline / ablation)
* ``OpenAIGenerator``          – calls the OpenAI Chat Completions API
* ``OllamaGenerator``          – calls a locally running Ollama model

"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import AugmentedQuery, RetrievalResult


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
            Reranked retrieval results that form the context window.
            ``context[0]`` is the most relevant chunk (rank 1).

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
        Wenn die Antwort nicht im Kontext zu finden ist, antworte: "Ich weiß es nicht."
        
        Frage: {query}
        Kontext: {context_str}
        
        Antwort:
        """
    
