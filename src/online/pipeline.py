"""
Online pipeline orchestrator.

Wires together a :class:`~src.online.query.BaseQueryProcessor`,
a :class:`~src.online.retrieval.BaseRetriever`, and a
:class:`~src.online.reranking.BaseReranker` into a single callable
query-time pipeline.

Usage example (skeleton – components not yet implemented)::

    pipeline = OnlinePipeline(
        query_processor=MyQueryProcessor(),
        retriever=MyRetriever(index=my_index, top_k=20),
        reranker=MyReranker(top_n=5),
    )
    results = pipeline.query("What are the admission requirements?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models.data_models import AugmentedQuery, RetrievalResult
from src.online.query.base import BaseQueryProcessor
from src.online.retrieval.base import BaseRetriever
from src.online.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


@dataclass
class OnlinePipelineResult:
    """Carries the intermediate and final artefacts of an online query run."""

    augmented_query: AugmentedQuery
    """Query after processing/augmentation."""

    retrieval_candidates: list[RetrievalResult]
    """Raw retriever output before reranking."""

    reranked_results: list[RetrievalResult]
    """Final results after reranking, ordered by rank (rank 1 = best)."""


class OnlinePipeline:
    """
    Runs the three-stage online (query-time) pipeline:

    1. **Query processing** – augment / expand the raw user query.
    2. **Retrieval**         – fetch the top-k candidates from the index.
    3. **Reranking**         – reorder and prune to the top-n most relevant chunks.

    All three stages are injected via the constructor, making each fully
    swappable without touching the pipeline logic.
    """

    def __init__(
        self,
        query_processor: BaseQueryProcessor,
        retriever: BaseRetriever,
        reranker: BaseReranker,
    ) -> None:
        self.query_processor = query_processor
        self.retriever = retriever
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, raw_query: str) -> OnlinePipelineResult:
        """
        Process a user query through the full online pipeline.

        Parameters
        ----------
        raw_query:
            The raw query string from the user.

        Returns
        -------
        OnlinePipelineResult
            Contains the augmented query, raw retrieval candidates, and
            the final reranked results.
        """
        logger.info(
            "Online pipeline query | processor=%s | retriever=%s | reranker=%s",
            self.query_processor.name,
            self.retriever.name,
            self.reranker.name,
        )

        # Stage 1 – Query processing
        logger.debug("Stage 1/3: Processing query…")
        augmented_query: AugmentedQuery = self.query_processor.process(raw_query)
        logger.debug(
            "Stage 1/3: Done – %d query variant(s) produced.",
            len(augmented_query.processed_queries),
        )

        # Stage 2 – Retrieval
        logger.debug("Stage 2/3: Retrieving top-%d candidates…", self.retriever.top_k)
        candidates: list[RetrievalResult] = self.retriever.retrieve(augmented_query)
        logger.debug("Stage 2/3: Done – %d candidate(s) retrieved.", len(candidates))

        # Stage 3 – Reranking
        logger.debug("Stage 3/3: Reranking to top-%d…", self.reranker.top_n)
        reranked: list[RetrievalResult] = self.reranker.rerank(augmented_query, candidates)
        logger.debug("Stage 3/3: Done – %d result(s) returned.", len(reranked))

        return OnlinePipelineResult(
            augmented_query=augmented_query,
            retrieval_candidates=candidates,
            reranked_results=reranked,
        )

    def describe(self) -> dict[str, str]:
        """Return a summary dict of the pipeline's component names."""
        return {
            "query_processor": self.query_processor.name,
            "retriever": self.retriever.name,
            "reranker": self.reranker.name,
        }
