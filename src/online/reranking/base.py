"""
Abstract base class for rerankers (Online Pipeline – Step 3).

A reranker receives the candidate list from the retriever and the original
(or augmented) query, and returns a reordered, possibly pruned list of
:class:`~src.models.RetrievalResult` objects.

Possible concrete implementations
----------------------------------
* ``PassthroughReranker``        – returns the retriever's ranking unchanged
                                   (baseline / ablation)
* ``CrossEncoderReranker``       – scores each (query, chunk) pair with a
                                   fine-tuned cross-encoder model
* ``LLMReranker``                – prompts an LLM/SLM to score or rank
                                   candidates (listwise or pointwise)
* ``ColBERTReranker``            – late-interaction reranking via ColBERT v2
* ``ReciprocaRankFusionReranker``– RRF across multiple ranked lists (useful
                                   when the retriever returns results from
                                   several query variants)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import AugmentedQuery, RetrievalResult


class BaseReranker(ABC):
    """
    Contract for all rerankers.

    Subclasses **must** implement :meth:`rerank`.
    """

    def __init__(self, top_n: int = 5) -> None:
        """
        Parameters
        ----------
        top_n:
            Number of results to return after reranking.  Should be ≤ the
            retriever's ``top_k``.
        """
        self.top_n = top_n

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this reranker.

        Example: ``'cross_encoder_ms_marco'``.
        """

    @abstractmethod
    def rerank(
        self,
        augmented_query: AugmentedQuery,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank the retriever's candidate list.

        Parameters
        ----------
        augmented_query:
            The processed query (use ``augmented_query.original_query`` for
            pointwise / listwise reranking).
        candidates:
            Ordered list of candidates from the retriever
            (``candidates[0]`` is the retriever's best hit).

        Returns
        -------
        list[RetrievalResult]
            Up to ``self.top_n`` results, reordered by the reranker's
            scoring.  Each result's ``rank`` field must be set (1-indexed).
        """
