"""
Passthrough chunker – keeps each document as a single chunk.

Intended for baseline experiments where the generator receives the entire
preprocessed document as context, with no splitting applied.
"""

from __future__ import annotations

from src.models import Document, Chunk
from .base import BaseChunker


class PassthroughChunker(BaseChunker):
    """
    Returns each document as a single :class:`~src.models.Chunk`.

    No splitting is performed.  This is useful as a baseline to measure
    whether chunking itself adds value, or to feed whole documents directly
    to the generator.

    The produced ``chunk_id`` follows the same ``'{doc_id}_chunk_0'``
    convention used by the other chunkers so that downstream components
    remain compatible.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Wrap the entire document text in a single chunk.

        Parameters
        ----------
        document:
            A preprocessed document whose ``text`` is ready for use.

        Returns
        -------
        list[Chunk]
            A list containing exactly one chunk (or an empty list if the
            document text is empty).
        """
        if not document.text:
            return []

        return [
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_0",
                text=document.text,
                chunker_name=self.name,
                metadata={"source_path": str(document.source_path)},
            )
        ]
