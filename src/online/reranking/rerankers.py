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
import torch

from src.models import AugmentedQuery, Chunk, RetrievalResult
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
        Hard upper limit on results to return after reranking.
    threshold:
        Minimum ``reranking_score`` a result must reach to be included.
        Results below this value are discarded even if they are within
        ``top_n``.  Set to ``0.0`` to disable threshold filtering.
    model_name:
        HuggingFace model ID.  Override only when testing with a
        different checkpoint.
    """

    def __init__(
        self,
        top_n: int = 5,
        threshold: float = 0.1,
        model_name: str = "jinaai/jina-reranker-v3",
    ) -> None:
        super().__init__(top_n=top_n)
        self.threshold = threshold
        self._model_name = model_name

        # Lazy import so that the rest of the codebase can be imported
        # even when transformers is not installed.
        from transformers import AutoModel  # type: ignore[import]

        logger.info("Loading reranker model '%s' …", model_name)

        if torch.cuda.is_available():
            self._model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
            )
        else:
            self._model = AutoModel.from_pretrained(
                model_name,
                dtype="auto",
                trust_remote_code=True,
            )
        self._model.eval()
        device = next(self._model.parameters()).device
        logger.info("Reranker model '%s' loaded on device: %s", model_name, device)

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"jina_reranker({self._model_name})"

    def rerank(
        self,
        augmented_query: AugmentedQuery,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Score every (query, chunk) pair with the jina-reranker-v3 model and
        return the top-n results sorted by descending relevance.

        Parameters
        ----------
        augmented_query:
            The processed query; ``original_query`` is used as the
            question text passed to the reranker.
        candidates:
            Candidate results from the retriever.

        Returns
        -------
        list[RetrievalResult]
            Up to ``self.top_n`` results, sorted by rerank score (highest
            first), with ``reranking_score`` populated.
        """
        if not candidates:
            return []

        query = augmented_query.original_query
        passages = [r.chunk.text for r in candidates]

        logger.debug(
            "Reranking %d candidate(s) with '%s' …",
            len(candidates),
            self._model_name,
        )

        # Score all candidates so the threshold can be applied across the full
        # candidate pool – not just the first top_n hits.
        rankings = self._model.rerank(
            query,
            passages,
            top_n=len(candidates),
        )

        reranked: list[RetrievalResult] = []
        for entry in rankings:
            score = float(entry["relevance_score"])
            if score < self.threshold:
                break  # rankings are sorted descending; no later entry will pass
            reranked.append(
                RetrievalResult(
                    chunk=candidates[entry["index"]].chunk,
                    retrieval_score=candidates[entry["index"]].retrieval_score,
                    reranking_score=score,
                )
            )
            if len(reranked) == self.top_n:
                break  # hard cap reached

        logger.debug("Reranking done – returning %d result(s).", len(reranked))
        return reranked

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
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
        list[list[RetrievalResult]]
            One reranked result list per input query, in the same order.
        """
        if not augmented_queries:
            return []

        logger.debug(
            "Batch reranking %d query(ies) with '%s' …",
            len(augmented_queries),
            self._model_name,
        )

        batch_reranked: list[list[RetrievalResult]] = []
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
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        # Propagate retrieval_score as reranking_score to signal no reranking
        return [
            RetrievalResult(
                chunk=r.chunk,
                retrieval_score=r.retrieval_score,
                reranking_score=r.retrieval_score,
            )
            for r in candidates[: self.top_n]
        ]

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
        return [self.rerank(aq, candidates) for aq, candidates in zip(augmented_queries, candidates_batch)]
