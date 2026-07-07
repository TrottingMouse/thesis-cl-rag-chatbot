"""
Abstract base class for rerankers (Online Pipeline – Step 3).

A reranker receives the candidate list from the retriever and the original
(or augmented) query, and returns a reordered, possibly pruned list of
:class:`~src.models.Chunk` objects.

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

from src.models import AugmentedQuery, Chunk


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
        candidates: list[Chunk],
    ) -> list[Chunk]:
        """
        Rerank the retriever's candidate list.

        Parameters
        ----------
        augmented_query:
            The processed query (use ``augmented_query.original_query`` for
            pointwise / listwise reranking).
        candidates:
            Ordered list of candidate chunks from the retriever
            (``candidates[0]`` is the retriever's best hit).

        Returns
        -------
        list[Chunk]
            Up to ``self.top_n`` chunks, reordered by the reranker's scoring.
        """

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[Chunk]],
    ) -> list[list[Chunk]]:
        """
        Rerank candidates for a batch of queries.

        The default implementation calls :meth:`rerank` for each query
        sequentially.  Subclasses that can exploit batched model inference
        (e.g. a single cross-encoder forward pass for all pairs) should
        override this method.

        Parameters
        ----------
        augmented_queries:
            One :class:`~src.models.AugmentedQuery` per input query.
        candidates_batch:
            One candidate list per query, in the same order as
            ``augmented_queries``.

        Returns
        -------
        list[list[Chunk]]
            One reranked result list per input query, in the same order.
        """
        return [
            self.rerank(aq, candidates)
            for aq, candidates in zip(augmented_queries, candidates_batch)
        ]
