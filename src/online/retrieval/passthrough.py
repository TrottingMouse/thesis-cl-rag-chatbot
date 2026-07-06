"""
Passthrough retriever – returns every indexed chunk for any query.

Intended for baseline experiments where the generator receives the whole
preprocessed corpus as context, bypassing any similarity search or ranking.
"""

from __future__ import annotations

from src.models import AugmentedQuery, RetrievalResult
from src.offline.indexing.passthrough import PassthroughIndexBuilder
from src.online.retrieval.base import BaseRetriever


class PassthroughRetriever(BaseRetriever):
    """
    Returns **all** stored chunks as retrieval results, ignoring the query.

    Each chunk is assigned a constant score of ``1.0`` and a rank that
    reflects its original storage order.  No embedding or similarity
    computation is performed.

    This retriever is coupled to :class:`~src.offline.indexing.PassthroughIndexBuilder`
    because it accesses ``index_builder.chunks`` directly.

    Notes
    -----
    * The ``top_k`` parameter of the base class is **ignored** – every chunk
      is always returned.  This ensures the generator sees the complete corpus,
      which is the definition of the passthrough baseline.
    * Downstream rerankers will receive all chunks; a
      :class:`~src.online.reranking.PassthroughReranker` (or any identity
      reranker) is the natural companion to avoid further pruning.
    """

    def __init__(self, index_builder: PassthroughIndexBuilder) -> None:
        """
        Parameters
        ----------
        index_builder:
            A *loaded* :class:`PassthroughIndexBuilder` (i.e. ``load()`` has
            already been called so that ``index_builder.chunks`` is populated).
        """
        # Pass top_k=len(chunks) would require load() first; we set a sentinel
        # value and override retrieve() to always return everything.
        super().__init__(index_builder, top_k=0)

    @property
    def name(self) -> str:
        return "passthrough_retriever"

    def retrieve(self, augmented_query: AugmentedQuery) -> list[RetrievalResult]:
        """
        Return all indexed chunks regardless of the query.

        Parameters
        ----------
        augmented_query:
            The processed query (content is ignored).

        Returns
        -------
        list[RetrievalResult]
            One :class:`~src.models.RetrievalResult` per stored chunk, in
            storage order, each with ``score=1.0``.
        """
        chunks = self.index_builder.chunks
        return [
            RetrievalResult(
                chunk=chunk,
                score=1.0,
                retriever_name=self.name,
                rank=rank,
            )
            for rank, chunk in enumerate(chunks, start=1)
        ]

    def retrieve_batch(
        self, augmented_queries: list[AugmentedQuery]
    ) -> list[list[RetrievalResult]]:
        """
        Return all chunks for every query in the batch.

        Parameters
        ----------
        augmented_queries:
            One :class:`~src.models.AugmentedQuery` per input query.

        Returns
        -------
        list[list[RetrievalResult]]
            The same full-corpus result list repeated once per query.
        """
        results = self.retrieve(augmented_queries[0]) if augmented_queries else []
        return [results for _ in augmented_queries]
