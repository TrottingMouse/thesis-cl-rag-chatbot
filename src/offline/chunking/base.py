"""
Abstract base class for chunkers (Offline Pipeline – Step 2).

A chunker splits a :class:`~src.models.Document` into a list of
:class:`~src.models.Chunk` objects.  Chunking strategy has a large impact on
retrieval quality, especially with small context windows (SLMs).

Possible concrete implementations
----------------------------------
* ``FixedSizeChunker``           – fixed token/character window with overlap
* ``SentenceChunker``            – sentence-boundary-aware splitting (spaCy / NLTK)
* ``RecursiveCharChunker``       – LangChain-style recursive splitting on
                                   ``['\n\n', '\n', ' ', '']``
* ``SemanticChunker``            – embedding-similarity-based boundary detection
* ``PropositionChunker``         – one proposition per chunk (pairs with the
                                   ``PropositionPreprocessor``)
* ``MarkdownHeaderChunker``      – splits on Markdown heading hierarchy
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Document, Chunk


class BaseChunker(ABC):
    """
    Contract for all chunking strategies.

    Subclasses **must** implement :meth:`chunk`.  They may override
    :meth:`chunk_batch` for efficiency.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this chunker.

        Used to populate :attr:`Chunk.chunker_name` and for logging /
        experiment tracking.  Example: ``'fixed_size_128'``.
        """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single processed document into chunks.

        Parameters
        ----------
        document:
            A preprocessed document whose ``text`` is ready for splitting.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks; each chunk's ``chunk_id`` must be unique
            within the document.
        """

    def chunk_batch(
        self, documents: list[Document]
    ) -> list[Chunk]:
        """
        Split a list of documents into chunks.

        Returns a flat list containing the chunks of all documents in order.
        Override for batched / parallel implementations.

        Parameters
        ----------
        documents:
            Preprocessed documents to chunk.

        Returns
        -------
        list[Chunk]
        """
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks
