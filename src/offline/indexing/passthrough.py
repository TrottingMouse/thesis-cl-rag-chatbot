"""
Passthrough index builder – persists chunks with no embedding.

Intended for baseline experiments where the :class:`PassthroughRetriever`
returns every stored chunk verbatim, giving the generator the whole corpus
as context.

No vector index is created; the chunks are simply pickled to disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.models import Chunk
from .base import BaseIndexBuilder

class PassthroughIndexBuilder(BaseIndexBuilder):
    """
    Stores a list of :class:`~src.models.Chunk` objects on disk without
    embedding them.

    The companion :class:`~src.online.retrieval.PassthroughRetriever` loads
    these chunks and returns all of them in response to any query, so the
    generator receives the entire indexed corpus as context.

    File layout under :attr:`storage_path`::

        chunks.pkl   – pickled ``list[Chunk]``
    """

    # Filename for the persisted chunks list
    _CHUNKS_FILE = "chunks.pkl"

    def __init__(self, storage_path: Path) -> None:
        super().__init__(storage_path)
        self.chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return "passthrough_index"

    def build(self, chunks: list[Chunk]) -> None:
        """
        Persist *chunks* to disk without embedding them.

        Parameters
        ----------
        chunks:
            All chunks to be stored.

        Raises
        ------
        ValueError
            If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Cannot build a passthrough index with an empty chunks list.")

        self.chunks = chunks
        self.storage_path.mkdir(parents=True, exist_ok=True)

        chunks_path = self.storage_path / self._CHUNKS_FILE
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    # def load(self) -> None:
    #     """
    #     Load previously persisted chunks from disk into memory.

    #     Raises
    #     ------
    #     FileNotFoundError
    #         If :meth:`build` has not been called yet.
    #     """
    #     if not self.is_built():
    #         raise FileNotFoundError(
    #             f"Passthrough index not found at {self.storage_path}. "
    #             "Run the offline pipeline first."
    #         )

    #     chunks_path = self.storage_path / self._CHUNKS_FILE
    #     with open(chunks_path, "rb") as f:
    #         self.chunks = pickle.load(f)

    # def is_built(self) -> bool:
    #     """Return ``True`` if the chunks file exists on disk."""
    #     return (self.storage_path / self._CHUNKS_FILE).exists()
