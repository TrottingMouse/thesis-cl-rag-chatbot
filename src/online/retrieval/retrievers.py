import numpy as np
import faiss

from src.online.retrieval.base import BaseRetriever
from src.models.data_models import AugmentedQuery, RetrievalResult
from src.offline.indexing.indexing import FaissIndexBuilder


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
        # BaseRetriever calls index_builder.load() and stores the return value
        # (None) in self.index.  We additionally keep the builder itself so we
        # can access the FAISS index object and the chunks list.
        self._index_builder = index_builder
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

        if len(queries) == 1:
            return self._search_single(queries[0])

        # Multiple query variants → per-query ranked lists → RRF fusion
        per_query_results: list[list[RetrievalResult]] = [
            self._search_single(q) for q in queries
        ]
        return self._reciprocal_rank_fusion(per_query_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_single(self, query: str) -> list[RetrievalResult]:
        """
        Embed *query*, search the FAISS index, and return up to ``top_k``
        ``RetrievalResult`` objects ordered by descending cosine similarity.
        """
        faiss_index: faiss.Index = self._index_builder.index
        chunks = self._index_builder.chunks
        model = self._index_builder.model

        # Encode and normalise to unit length (matching build-time normalisation)
        query_vector: np.ndarray = model.encode(
            [query],
            convert_to_numpy=True,
            task="retrieval",
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