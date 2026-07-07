"""
Abstract base class for retrievers (Online Pipeline – Step 2).

A retriever takes an :class:`~src.models.AugmentedQuery` (which may contain
multiple query variants) and returns a ranked list of
:class:`~src.models.Chunk` objects from the pre-built index.

Possible concrete implementations
----------------------------------
* ``DenseRetriever``   – cosine-similarity search over a FAISS/Chroma index
* ``BM25Retriever``    – sparse lexical retrieval (rank_bm25 / Elasticsearch)
* ``HybridRetriever``  – linear combination of dense + sparse scores (RRF or
                         weighted sum)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import AugmentedQuery, Chunk
from src.offline.indexing import BaseIndexBuilder


class BaseRetriever(ABC):
    """
    Contract for all retrievers.

    A retriever is **read-only** with respect to the index; it only calls
    :meth:`BaseIndexBuilder.load` to get query-time access to the index.

    Subclasses **must** implement :meth:`retrieve`.  For throughput-sensitive
    workloads, subclasses may also override :meth:`retrieve_batch`.
    """

    def __init__(self, index_builder: BaseIndexBuilder, top_k: int = 20) -> None:
        """
        Parameters
        ----------
        index:
            A loaded index builder whose index is ready for querying.
        top_k:
            Maximum number of candidates to return before reranking.
            A generous ``top_k`` (e.g. 20–50) gives the reranker more to
            work with.
        """
        self.index_builder = index_builder
        self.top_k = top_k
        self.index = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this retriever.

        Example: ``'dense_faiss'``.
        """

    @abstractmethod
    def retrieve(self, augmented_query: AugmentedQuery) -> list[Chunk]:
        """
        Retrieve the top-k candidate chunks for the given query.

        If ``augmented_query.processed_queries`` contains multiple variants,
        implementations should merge and deduplicate results across variants
        (e.g. via Reciprocal Rank Fusion).

        Parameters
        ----------
        augmented_query:
            The processed query produced by a :class:`BaseQueryProcessor`.

        Returns
        -------
        list[Chunk]
            Up to ``self.top_k`` chunks, ordered by descending relevance
            score.  Duplicates (same ``chunk_id``) must be removed.
        """

    def retrieve_batch(
        self, augmented_queries: list[AugmentedQuery]
    ) -> list[list[Chunk]]:
        """
        Retrieve candidates for a batch of augmented queries.

        The default implementation calls :meth:`retrieve` for each query
        sequentially.  Subclasses that can exploit batched embedding or
        index operations should override this method.

        Parameters
        ----------
        augmented_queries:
            One :class:`~src.models.AugmentedQuery` per input query.

        Returns
        -------
        list[list[Chunk]]
            One result list per input query, in the same order.
        """
        return [self.retrieve(aq) for aq in augmented_queries]
