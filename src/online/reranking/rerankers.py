"""
Concrete reranker implementations (Online Pipeline – Step 3).

Currently implemented
---------------------
* :class:`JinaReranker` – uses ``jinaai/jina-reranker-v3`` via
  ``transformers`` ``AutoModel`` with the model's built-in ``.rerank()``
  method.
* :class:`PassthroughReranker` – returns the retriever's ranking unchanged
  (baseline / ablation, no model required).
"""

from __future__ import annotations

import logging

from src.models import AugmentedQuery, Chunk
from src.online.reranking import BaseReranker

logger = logging.getLogger(__name__)


class JinaReranker(BaseReranker):
    """
    Reranker backed by ``jinaai/jina-reranker-v3`` via ``transformers``
    ``AutoModel``.

    Requires ``transformers`` and ``trust_remote_code=True`` (the model ships
    custom modelling code).  The model uses a *last-but-not-late interaction*
    architecture: query and all documents are processed in a single causal
    self-attention pass, making it both accurate and efficient.

    Parameters
    ----------
    top_n:
        Number of results to keep after reranking.
    model_name:
        HuggingFace model ID.  Override only when testing with a
        different checkpoint.
    """

    def __init__(
        self,
        top_n: int = 5,
        model_name: str = "jinaai/jina-reranker-v3",
    ) -> None:
        super().__init__(top_n=top_n)
        self._model_name = model_name

        # Lazy import so that the rest of the codebase can be imported
        # even when transformers is not installed.
        from transformers import AutoModel  # type: ignore[import]

        logger.info("Loading reranker model '%s' …", model_name)

        self._model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Reranker model '%s' loaded.", model_name)

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"jina_reranker({self._model_name})"

    def rerank(
        self,
        augmented_query: AugmentedQuery,
        candidates: list[Chunk],
    ) -> list[Chunk]:
        """
        Score every (query, chunk) pair with the jina-reranker-v3 model and
        return the top-n chunks sorted by descending relevance.

        Parameters
        ----------
        augmented_query:
            The processed query; ``original_query`` is used as the
            question text passed to the reranker.
        candidates:
            Candidate chunks from the retriever.

        Returns
        -------
        list[Chunk]
            Up to ``self.top_n`` chunks, sorted by rerank score (highest first).
        """
        if not candidates:
            return []

        query = augmented_query.original_query
        passages = [chunk.text for chunk in candidates]

        logger.debug(
            "Reranking %d candidate(s) with '%s' …",
            len(candidates),
            self._model_name,
        )

        # model.rerank returns a list of dicts sorted by descending relevance:
        # [{"index": int, "relevance_score": float, "document": str}, …]
        rankings = self._model.rerank(
            query,
            passages,
            top_n=min(self.top_n, len(candidates)),
        )

        reranked: list[Chunk] = [candidates[entry["index"]] for entry in rankings]

        logger.debug("Reranking done – returning %d result(s).", len(reranked))
        return reranked

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[Chunk]],
    ) -> list[list[Chunk]]:
        """
        Rerank candidates for a batch of queries.

        Each query is scored independently via ``model.rerank``.  The
        jina-reranker-v3 model's listwise architecture already processes all
        documents for a single query in one forward pass, so per-query calls
        are efficient.

        Parameters
        ----------
        augmented_queries:
            One per input query.
        candidates_batch:
            One candidate list per query, in the same order.

        Returns
        -------
        list[list[Chunk]]
            One reranked result list per input query, in the same order.
        """
        if not augmented_queries:
            return []

        logger.debug(
            "Batch reranking %d query(ies) with '%s' …",
            len(augmented_queries),
            self._model_name,
        )

        batch_reranked: list[list[Chunk]] = []
        for aq, candidates in zip(augmented_queries, candidates_batch):
            batch_reranked.append(self.rerank(aq, candidates))

        logger.debug("Batch reranking done.")
        return batch_reranked


class PassthroughReranker(BaseReranker):
    """
    No-op reranker that returns the retriever's ranking unchanged.

    Useful as a baseline / ablation when you want to disable reranking
    without changing the pipeline structure.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def rerank(
        self,
        augmented_query: AugmentedQuery,
        candidates: list[Chunk],
    ) -> list[Chunk]:
        return candidates[: self.top_n]

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[Chunk]],
    ) -> list[list[Chunk]]:
        return [self.rerank(aq, candidates) for aq, candidates in zip(augmented_queries, candidates_batch)]
