"""
Online pipeline orchestrator.

Wires together a :class:`~src.online.query.BaseQueryProcessor`,
a :class:`~src.online.retrieval.BaseRetriever`,
a :class:`~src.online.reranking.BaseReranker`, and a
:class:`~src.online.generation.BaseGenerator` into a single callable
query-time pipeline.

Usage example (skeleton – components not yet implemented)::

    pipeline = OnlinePipeline(
        query_processor=MyQueryProcessor(),
        retriever=MyRetriever(index=my_index, top_k=20),
        reranker=MyReranker(top_n=5),
        generator=MyGenerator(),
    )
    results = pipeline.query("What are the admission requirements?")
    print(results.generation_result.answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models.data_models import AugmentedQuery, RetrievalResult
from src.online.generation.base import BaseGenerator
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

    generation_result: str
    """Answer generated from the reranked context by the generator."""


class OnlinePipeline:
    """
    Runs the four-stage online (query-time) pipeline:

    1. **Query processing** – augment / expand the raw user query.
    2. **Retrieval**         – fetch the top-k candidates from the index.
    3. **Reranking**         – reorder and prune to the top-n most relevant chunks.
    4. **Generation**        – produce a natural-language answer from the context.

    All four stages are injected via the constructor, making each fully
    swappable without touching the pipeline logic.
    """

    def __init__(
        self,
        query_processor: BaseQueryProcessor,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        generator: BaseGenerator,
    ) -> None:
        self.query_processor = query_processor
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

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
            Contains the augmented query, raw retrieval candidates,
            the final reranked results, and the generated answer string.
        """
        logger.info(
            "Online pipeline query | processor=%s | retriever=%s | reranker=%s | generator=%s",
            self.query_processor.name,
            self.retriever.name,
            self.reranker.name,
            self.generator.name,
        )

        # Stage 1 – Query processing
        logger.debug("Stage 1/4: Processing query…")
        augmented_query: AugmentedQuery = self.query_processor.process(raw_query)
        logger.debug(
            "Stage 1/4: Done – %d query variant(s) produced.",
            len(augmented_query.processed_queries),
        )

        # Stage 2 – Retrieval
        logger.debug("Stage 2/4: Retrieving top-%d candidates…", self.retriever.top_k)
        candidates: list[RetrievalResult] = self.retriever.retrieve(augmented_query)
        logger.debug("Stage 2/4: Done – %d candidate(s) retrieved.", len(candidates))

        # Stage 3 – Reranking
        logger.debug("Stage 3/4: Reranking to top-%d…", self.reranker.top_n)
        reranked: list[RetrievalResult] = self.reranker.rerank(augmented_query, candidates)
        logger.debug("Stage 3/4: Done – %d result(s) returned.", len(reranked))

        # Stage 4 – Generation
        logger.debug("Stage 4/4: Generating answer…")
        answer: str = self.generator.generate(raw_query, reranked)
        logger.debug("Stage 4/4: Done – answer produced by '%s'.", self.generator.name)

        return OnlinePipelineResult(
            augmented_query=augmented_query,
            retrieval_candidates=candidates,
            reranked_results=reranked,
            generation_result=answer,
        )

    def describe(self) -> dict[str, str]:
        """Return a summary dict of the pipeline's component names."""
        return {
            "query_processor": self.query_processor.name,
            "retriever": self.retriever.name,
            "reranker": self.reranker.name,
            "generator": self.generator.name,
        }
