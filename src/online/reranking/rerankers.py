"""
Concrete reranker implementations (Online Pipeline – Step 3).

Currently implemented
---------------------
* :class:`Qwen3Reranker` – uses ``Qwen/Qwen3-Reranker-0.6B`` via
  ``sentence-transformers`` ``CrossEncoder``.
* :class:`PassthroughReranker` – returns the retriever's ranking unchanged
  (baseline / ablation, no model required).
"""

from __future__ import annotations

import logging

from src.models.data_models import AugmentedQuery, RetrievalResult
from src.online.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class Qwen3Reranker(BaseReranker):
    """
    Reranker backed by ``Qwen/Qwen3-Reranker-0.6B`` via
    ``sentence-transformers`` ``CrossEncoder``.

    Requires ``sentence-transformers >= 5.4.0``.

    The model is a causal LM that scores (query, passage) pairs by
    comparing the logits of the ``yes`` / ``no`` tokens, so higher
    scores mean higher relevance.

    Parameters
    ----------
    top_n:
        Number of results to keep after reranking.
    model_name:
        HuggingFace model ID.  Override only when testing with a
        different checkpoint.
    device:
        PyTorch device string (e.g. ``'cpu'``, ``'cuda'``, ``'mps'``).
        ``None`` lets ``sentence-transformers`` auto-detect.
    """

    def __init__(
        self,
        top_n: int = 5,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str | None = None,
    ) -> None:
        super().__init__(top_n=top_n)
        self._model_name = model_name

        # Lazy import so that the rest of the codebase can be imported
        # even when sentence-transformers is not installed.
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        logger.info("Loading reranker model '%s' …", model_name)
        self._model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded.")

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"qwen3_reranker({self._model_name})"

    def rerank(
        self,
        augmented_query: AugmentedQuery,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Score every (query, chunk) pair with the Qwen3 cross-encoder and
        return the top-n results sorted by descending relevance.

        Parameters
        ----------
        augmented_query:
            The processed query; ``original_query`` is used as the
            question text passed to the cross-encoder.
        candidates:
            Candidate chunks from the retriever.

        Returns
        -------
        list[RetrievalResult]
            Up to ``self.top_n`` results, sorted by rerank score
            (highest first).  Each result's ``rank`` field is set
            (1-indexed) and a ``rerank_score`` key is added to
            ``metadata``.
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

        # sentence-transformers CrossEncoder.rank returns a list of dicts:
        # [{"corpus_id": int, "score": float}, …] sorted by descending score.
        rankings = self._model.rank(
            query,
            passages,
            top_k=min(self.top_n, len(candidates)),
            convert_to_tensor=False,
        )

        reranked: list[RetrievalResult] = []
        for rank_pos, entry in enumerate(rankings, start=1):
            result = candidates[entry["corpus_id"]]
            result.score = float(entry["score"])
            result.rank = rank_pos
            result.metadata["rerank_score"] = result.score
            result.metadata["reranker"] = self._model_name
            reranked.append(result)

        logger.debug("Reranking done – returning %d result(s).", len(reranked))
        return reranked


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
        top = candidates[: self.top_n]
        for rank_pos, result in enumerate(top, start=1):
            result.rank = rank_pos
        return top
