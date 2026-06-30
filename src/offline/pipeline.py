"""
Offline pipeline orchestrator.

Wires together a sequence of :class:`~src.offline.preprocessing.BasePreprocessor`
instances, a :class:`~src.offline.chunking.BaseChunker`, and a
:class:`~src.offline.indexing.BaseIndexBuilder` into a single callable
pipeline.

Usage example::

    pipeline = OfflinePipeline(
        preprocessors=[RawTextProcessor(), MyLLMProcessor()],
        chunker=MyChunker(),
        index_builder=MyIndexBuilder(storage_path=Path("storage/my_index")),
    )
    pipeline.run(raw_document_paths)
"""
from __future__ import annotations

from src.config import OfflineConfig
from src.offline.indexing import FaissIndexBuilder
from src.offline.chunking import FixedCharacterChunker
from src.offline.preprocessing import RawTextProcessor

import logging
from dataclasses import dataclass
from pathlib import Path

from src.models import Document, Chunk
from src.offline.preprocessing import BasePreprocessor
from src.offline.chunking import BaseChunker
from src.offline.indexing import BaseIndexBuilder

logger = logging.getLogger(__name__)


@dataclass
class OfflinePipelineResult:
    """Carries the intermediate artefacts produced by the offline pipeline."""

    processed_documents: list[Document]
    chunks: list[Chunk]


class OfflinePipeline:
    """
    Runs the three-stage offline pipeline:

    1. **Preprocessing** – pass documents through one or more preprocessors
       in sequence; each preprocessor's output becomes the next one's input.
    2. **Chunking**       – split documents into chunks.
    3. **Indexing**       – build and persist the index.

    All three stages are injected via the constructor, making each fully
    swappable without touching the pipeline logic.
    """

    def __init__(
        self,
        preprocessors: list[BasePreprocessor],
        chunker: BaseChunker,
        index_builder: BaseIndexBuilder,
    ) -> None:
        if not preprocessors:
            raise ValueError("At least one preprocessor must be provided.")
        self.preprocessors = preprocessors
        self.chunker = chunker
        self.index_builder = index_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raw_document_paths: list[str]) -> OfflinePipelineResult:
        """
        Execute the full offline pipeline.

        Parameters
        ----------
        raw_document_paths:
            Paths to the raw documents on disk (e.g. PDFs in ``documents/``).

        Returns
        -------
        OfflinePipelineResult
            Intermediate artefacts (processed docs + chunks) for inspection
            or debugging; the index is persisted to disk as a side effect.
        """
        preprocessor_names = " -> ".join(p.name for p in self.preprocessors)
        logger.info(
            "Offline pipeline started | preprocessors=[%s] | chunker=%s | index=%s",
            preprocessor_names,
            self.chunker.name,
            self.index_builder.name,
        )

        # Stage 1 – Preprocessing (chained)
        logger.info(
            "Stage 1/3: Preprocessing %d document(s) through %d preprocessor(s)…",
            len(raw_document_paths),
            len(self.preprocessors),
        )
        processed_docs = self.preprocessors[0].preprocess_from_paths(raw_document_paths)
        for i, preprocessor in enumerate(self.preprocessors[1:], start=2):
            logger.info(
                "Stage 1/3: Applying preprocessor %d/%d (%s)…",
                i,
                len(self.preprocessors),
                preprocessor.name,
            )
            processed_docs = [preprocessor.preprocess(doc) for doc in processed_docs]
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

    def describe(self) -> dict[str, str | list[str]]:
        """Return a summary dict of the pipeline's component names."""
        return {
            "preprocessors": [p.name for p in self.preprocessors],
            "chunker": self.chunker.name,
            "index_builder": self.index_builder.name,
        }

