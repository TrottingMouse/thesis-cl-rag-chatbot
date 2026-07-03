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
import torch

from src.models import AugmentedQuery, RetrievalResult
from src.online.reranking import BaseReranker

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
        model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    ) -> None:
        super().__init__(top_n=top_n)
        self._model_name = model_name

        # Lazy import so that the rest of the codebase can be imported
        # even when sentence-transformers is not installed.
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        import torch

        logger.info("Loading reranker model '%s' …", model_name)
        
        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs = {
                "torch_dtype": torch.bfloat16,  # Significantly faster and memory-efficient
                "device_map": "auto",           # Hands scheduling directly to HF Accelerate
                "attn_implementation": "sdpa"   # Standard PyTorch scaled dot-product attention
            }
            device = None  # Ignored by sentence-transformers when device_map is present
        else:
            device = "cpu"

        self._model = CrossEncoder(
            model_name, 
            device=device, 
            model_kwargs=model_kwargs
        )
        logger.info("Reranker model loaded on device: %s", self._model.device)

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

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
        """
        Rerank candidates for a batch of queries in a single model forward pass.

        All (query, passage) pairs from every query in the batch are flattened
        into one list and scored by the cross-encoder in one call to
        ``CrossEncoder.predict``.  The scores are then distributed back to the
        original per-query candidate lists.

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

        # Flatten all (query, passage) pairs
        flat_pairs: list[tuple[str, str]] = []
        # Track slice boundaries so we can redistribute scores later
        slices: list[tuple[int, int]] = []  # (start, end) into flat_pairs
        for aq, candidates in zip(augmented_queries, candidates_batch):
            start = len(flat_pairs)
            query = aq.original_query
            for result in candidates:
                flat_pairs.append((query, result.chunk.text))
            slices.append((start, len(flat_pairs)))

        if not flat_pairs:
            return [[] for _ in augmented_queries]

        logger.debug(
            "Batch reranking %d pair(s) across %d query(ies) with '%s' …",
            len(flat_pairs),
            len(augmented_queries),
            self._model_name,
        )

        # Single batched predict call
        all_scores: list[float] = self._model.predict(flat_pairs, convert_to_numpy=True).tolist()

        # Redistribute scores back to per-query results
        batch_reranked: list[list[RetrievalResult]] = []
        for (start, end), candidates in zip(slices, candidates_batch):
            query_scores = all_scores[start:end]
            # Sort indices by descending score and take top_n
            ranked_indices = sorted(
                range(len(query_scores)), key=lambda i: query_scores[i], reverse=True
            )[: self.top_n]
            reranked: list[RetrievalResult] = []
            for rank_pos, idx in enumerate(ranked_indices, start=1):
                result = candidates[idx]
                result.score = query_scores[idx]
                result.rank = rank_pos
                result.metadata["rerank_score"] = result.score
                result.metadata["reranker"] = self._model_name
                reranked.append(result)
            batch_reranked.append(reranked)

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
        top = candidates[: self.top_n]
        for rank_pos, result in enumerate(top, start=1):
            result.rank = rank_pos
        return top

    def rerank_batch(
        self,
        augmented_queries: list[AugmentedQuery],
        candidates_batch: list[list[RetrievalResult]],
    ) -> list[list[RetrievalResult]]:
        return [self.rerank(aq, candidates) for aq, candidates in zip(augmented_queries, candidates_batch)]
