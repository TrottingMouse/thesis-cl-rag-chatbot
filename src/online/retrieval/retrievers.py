import numpy as np
import faiss


from src.online.retrieval import BaseRetriever
from src.models import AugmentedQuery, RetrievalResult
from src.offline.indexing import FaissIndexBuilder


class FaissRetriever(BaseRetriever):
    """
    FAISS-based dense retriever using cosine similarity.

    For a single processed query the retriever embeds it, L2-normalises the
    vector (matching the normalisation applied at index build time), and runs a
    nearest-neighbour search on the ``IndexFlatIP`` index.

    When ``augmented_query.processed_queries`` contains *multiple* variants
    (e.g. from HyDE or query expansion) the per-query ranked lists are merged
    with **Reciprocal Rank Fusion** (RRF) so that chunks appearing highly in
    several lists bubble up in the final ranking.
    """

    # RRF smoothing constant (standard value from Cormack et al., 2009)
    _RRF_K: int = 60

    def __init__(self, index_builder: FaissIndexBuilder, top_k: int = 20) -> None:
        """
        Parameters
        ----------
        index_builder:
            A *loaded* ``FaissIndexBuilder`` (i.e. ``load()`` has already been
            called so that ``index_builder.index`` and ``index_builder.chunks``
            are populated).
        top_k:
            Maximum number of results to return.
        """
        super().__init__(index_builder, top_k)

    @property
    def name(self) -> str:
        return "faiss_retriever"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, augmented_query: AugmentedQuery) -> list[RetrievalResult]:
        """
        Retrieve the top-k candidate chunks for the given (augmented) query.

        Parameters
        ----------
        augmented_query:
            The processed query produced by a ``BaseQueryProcessor``.  If
            ``processed_queries`` contains more than one variant, results are
            merged via Reciprocal Rank Fusion before truncation to ``top_k``.

        Returns
        -------
        list[RetrievalResult]
            Up to ``self.top_k`` results ordered by descending relevance score.
            Duplicates (same ``chunk_id``) are removed.
        """
        queries = augmented_query.processed_queries
        if not queries:
            return []

        query_type = augmented_query.query_type

        if len(queries) == 1:
            return self._search_single(queries[0], query_type)

        # Multiple query variants → per-query ranked lists → RRF fusion
        per_query_results: list[list[RetrievalResult]] = [
            self._search_single(q, query_type) for q in queries
        ]
        return self._reciprocal_rank_fusion(per_query_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_single(self, query: str, query_type: str) -> list[RetrievalResult]:
        """
        Embed *query*, search the FAISS index, and return up to ``top_k``
        ``RetrievalResult`` objects ordered by descending cosine similarity.
        """
        faiss_index: faiss.Index = self.index_builder.index
        chunks = self.index_builder.chunks
        model = self.index_builder.model

        # Encode and normalise to unit length (matching build-time normalisation)
        query_vector: np.ndarray = model.encode(
            [query],
            convert_to_numpy=True,
            task='retrieval',
            prompt_name=query_type
        ).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Search – returns (distances, indices) shaped (1, top_k)
        scores, indices = faiss_index.search(query_vector, self.top_k)
        scores = scores[0]
        indices = indices[0]

        results: list[RetrievalResult] = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                # FAISS returns -1 when fewer than top_k vectors exist
                continue
            results.append(
                RetrievalResult(
                    chunk=chunks[idx],
                    score=float(score),
                    retriever_name=self.name,
                )
            )

        return results

    def retrieve_batch(
        self, augmented_queries: list[AugmentedQuery]
    ) -> list[list[RetrievalResult]]:
        """
        Retrieve top-k candidates for a batch of augmented queries.

        All ``processed_queries`` across every item in the batch are embedded
        in a single :func:`model.encode` call and searched in a single
        :func:`faiss_index.search` call, dramatically reducing per-query
        overhead compared with calling :meth:`retrieve` in a loop.

        Parameters
        ----------
        augmented_queries:
            One :class:`~src.models.AugmentedQuery` per input query.

        Returns
        -------
        list[list[RetrievalResult]]
            One result list per input query, in the same order.  Each list
            contains up to ``self.top_k`` results, duplicates removed, ordered
            by descending relevance score (with RRF when multiple processed
            query variants exist for a single query).
        """
        if not augmented_queries:
            return []

        faiss_index: faiss.Index = self.index_builder.index
        chunks = self.index_builder.chunks
        model = self.index_builder.model

        # Build a flat list of all processed query strings, tracking which
        # augmented-query index each one belongs to.
        # flat_queries[i]  -> the query string to embed
        # query_owners[i]  -> index into augmented_queries
        flat_queries: list[str] = []
        query_owners: list[int] = []
        for aq_idx, aq in enumerate(augmented_queries):
            for pq in aq.processed_queries:
                flat_queries.append(pq)
                query_owners.append(aq_idx)

        if not flat_queries:
            return [[] for _ in augmented_queries]

        # Single batched encode + normalise
        all_vectors: np.ndarray = model.encode(
            flat_queries,
            convert_to_numpy=True,
            task="retrieval",
            prompt_name=augmented_queries[0].query_type
        ).astype(np.float32)
        faiss.normalize_L2(all_vectors)

        # Single batched FAISS search: shape (num_flat, top_k)
        all_scores, all_indices = faiss_index.search(all_vectors, self.top_k)

        # Group per-query ranked lists back by augmented-query index
        # per_query_ranked_lists[aq_idx] accumulates one list per variant
        per_query_ranked_lists: list[list[list[RetrievalResult]]] = [
            [] for _ in augmented_queries
        ]
        for flat_idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            aq_idx = query_owners[flat_idx]
            results: list[RetrievalResult] = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                results.append(
                    RetrievalResult(
                        chunk=chunks[idx],
                        score=float(score),
                        retriever_name=self.name,
                    )
                )
            per_query_ranked_lists[aq_idx].append(results)

        # For each original augmented query, fuse or pass through
        batch_results: list[list[RetrievalResult]] = []
        for aq_idx, ranked_lists in enumerate(per_query_ranked_lists):
            if not ranked_lists:
                batch_results.append([])
            elif len(ranked_lists) == 1:
                batch_results.append(ranked_lists[0])
            else:
                batch_results.append(self._reciprocal_rank_fusion(ranked_lists))

        return batch_results

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

        RRF score for a chunk ``d`` = Σ_i  1 / (k + rank_i(d))

        where ``rank_i(d)`` is the 1-based position of ``d`` in list ``i`` and
        ``k`` is the smoothing constant ``_RRF_K``.  Chunks not present in a
        list are simply not included in that list's contribution.

        The final list is sorted by descending RRF score and truncated to
        ``top_k``.  The ``score`` field of the returned ``RetrievalResult``
        holds the RRF score; ``metadata`` retains the individual per-query
        cosine similarities under the key ``'per_query_scores'``.

        Parameters
        ----------
        ranked_lists:
            One ranked list per processed query variant.

        Returns
        -------
        list[RetrievalResult]
            Merged and deduplicated results, up to ``self.top_k``.
        """
        k = self._RRF_K

        # chunk_id → accumulated RRF score
        rrf_scores: dict[str, float] = {}
        # chunk_id → best RetrievalResult (used to recover the Chunk object)
        best_result: dict[str, RetrievalResult] = {}
        # chunk_id → list of individual cosine similarities
        per_query_scores: dict[str, list[float]] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                cid = result.chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                per_query_scores.setdefault(cid, []).append(result.score)
                # Keep the result with the highest cosine similarity as
                # representative for this chunk.
                if cid not in best_result or result.score > best_result[cid].score:
                    best_result[cid] = result

        # Build final list sorted by descending RRF score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused: list[RetrievalResult] = []
        for cid in sorted_ids[: self.top_k]:
            rep = best_result[cid]
            fused.append(
                RetrievalResult(
                    chunk=rep.chunk,
                    score=rrf_scores[cid],
                    retriever_name=self.name,
                    metadata={"per_query_scores": per_query_scores[cid]},
                )
            )

        return fused