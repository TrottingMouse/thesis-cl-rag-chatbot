"""
Offline pipeline orchestrator.

Wires together a :class:`~src.offline.preprocessing.BasePreprocessor`,
a :class:`~src.offline.chunking.BaseChunker`, and a
:class:`~src.offline.indexing.BaseIndexBuilder` into a single callable
pipeline.

Usage example (skeleton – components not yet implemented)::

    pipeline = OfflinePipeline(
        preprocessor=MyPreprocessor(),
        chunker=MyChunker(),
        index_builder=MyIndexBuilder(storage_path=Path("storage/my_index")),
    )
    pipeline.run(raw_documents)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.models import Document, Chunk
from src.offline.preprocessing.base import BasePreprocessor
from src.offline.chunking.base import BaseChunker
from src.offline.indexing.base import BaseIndexBuilder

logger = logging.getLogger(__name__)


@dataclass
class OfflinePipelineResult:
    """Carries the intermediate artefacts produced by the offline pipeline."""

    processed_documents: list[Document]
    chunks: list[Chunk]


class OfflinePipeline:
    """
    Runs the three-stage offline pipeline:

    1. **Preprocessing** – convert raw documents to plain text.
    2. **Chunking**       – split documents into chunks.
    3. **Indexing**       – build and persist the index.

    All three stages are injected via the constructor, making each fully
    swappable without touching the pipeline logic.
    """

    def __init__(
        self,
        preprocessor: BasePreprocessor,
        chunker: BaseChunker,
        index_builder: BaseIndexBuilder,
    ) -> None:
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.index_builder = index_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raw_documents: list[Document]) -> OfflinePipelineResult:
        """
        Execute the full offline pipeline.

        Parameters
        ----------
        raw_documents:
            Documents loaded from disk (e.g. the two PDFs in ``documents/``).

        Returns
        -------
        OfflinePipelineResult
            Intermediate artefacts (processed docs + chunks) for inspection
            or debugging; the index is persisted to disk as a side effect.
        """
        logger.info(
            "Offline pipeline started | preprocessor=%s | chunker=%s | index=%s",
            self.preprocessor.name,
            self.chunker.name,
            self.index_builder.name,
        )

        # Stage 1 – Preprocessing
        logger.info("Stage 1/3: Preprocessing %d document(s)…", len(raw_documents))
        processed_docs = self.preprocessor.preprocess_batch(raw_documents)
        logger.info("Stage 1/3: Done – %d document(s) processed.", len(processed_docs))

        # Stage 2 – Chunking
        logger.info("Stage 2/3: Chunking…")
        chunks = self.chunker.chunk_batch(processed_docs)
        logger.info("Stage 2/3: Done – %d chunk(s) produced.", len(chunks))

        # Stage 3 – Indexing
        logger.info("Stage 3/3: Building index…")
        self.index_builder.build(chunks)
        logger.info("Stage 3/3: Done – index persisted to '%s'.", self.index_builder.storage_path)

        return OfflinePipelineResult(
            processed_documents=processed_docs,
            chunks=chunks,
        )

    def describe(self) -> dict[str, str]:
        """Return a summary dict of the pipeline's component names."""
        return {
            "preprocessor": self.preprocessor.name,
            "chunker": self.chunker.name,
            "index_builder": self.index_builder.name,
        }
