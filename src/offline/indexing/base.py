"""
Abstract base class for index builders (Offline Pipeline – Step 3).

An index builder takes the flat list of :class:`~src.models.Chunk` objects
produced by a chunker and builds a persistent, queryable index.

Possible concrete implementations
----------------------------------
* ``FaissIndexBuilder``      – dense vector index (FAISS flat / IVF / HNSW)
* ``BM25IndexBuilder``       – sparse lexical index (rank_bm25 / Elasticsearch)
* ``HybridIndexBuilder``     – stores both a dense and a sparse index and
                               exposes both to a hybrid retriever
* ``ChromaIndexBuilder``     – ChromaDB-backed vector store (persistent on disk)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models import Chunk


class BaseIndexBuilder(ABC):
    """
    Contract for all index builders.

    Subclasses **must** implement :meth:`build` and :meth:`load`.

    The index is expected to be persisted to / loaded from :attr:`storage_path`
    so that the offline pipeline need not re-run when the online pipeline starts.
    """

    def __init__(self, storage_path: Path) -> None:
        """
        Parameters
        ----------
        storage_path:
            Directory under which the index artefacts are stored.
            Subclasses may create sub-directories within this path.
        """
        self.storage_path = storage_path

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this index builder.

        Example: ``'faiss_cosine'``.
        """

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """
        Build the index from a list of chunks and persist it to disk.

        This is the *write* path, called once during the offline pipeline.

        Parameters
        ----------
        chunks:
            All chunks to be indexed.  The builder is responsible for
            embedding them (if needed) and persisting the result.
        """

    @abstractmethod
    def load(self) -> None:
        """
        Load a previously built index from :attr:`storage_path` into memory.

        Called at the start of the online pipeline before any queries arrive.
        Raises :exc:`FileNotFoundError` if the index has not been built yet.
        """

    @abstractmethod
    def is_built(self) -> bool:
        """
        Return ``True`` if a persisted index exists at :attr:`storage_path`
        and can be loaded without rebuilding.
        """
