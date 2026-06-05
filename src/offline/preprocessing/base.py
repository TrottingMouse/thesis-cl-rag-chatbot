"""
Abstract base class for document preprocessors (Offline Pipeline – Step 1).

A preprocessor transforms a :class:`~src.models.Document` into another
:class:`~src.models.Document` whose ``text`` field is ready for
chunking.

Possible concrete implementations
----------------------------------
* ``PlainTextPreprocessor``   – extracts raw text from PDF without modification
* ``MarkdownTablePreprocessor`` – converts detected tables to Markdown
* ``PropositionPreprocessor`` – uses an LLM/SLM to decompose content into
  atomic propositions (Dense X Retrieval style)
* ``StructuredSectionPreprocessor`` – preserves document structure (headings,
  lists) as annotated Markdown
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Document


class BasePreprocessor(ABC):
    """
    Contract for all document preprocessors.

    Subclasses **must** implement :meth:`preprocess`.  They may override
    :meth:`preprocess_batch` for efficiency (the default implementation is a
    simple loop).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this preprocessor.

        Used to populate :attr:`Document.preprocessor_name` and for
        logging / experiment tracking.  Example: ``'markdown_table'``.
        """

    @abstractmethod
    def preprocess(self, document: Document) -> Document:
        """
        Transform a single raw document into a processed document.

        Parameters
        ----------
        document:
            The raw document loaded from disk.

        Returns
        -------
        Document
            The cleaned/converted document ready for chunking.
        """

    def preprocess_batch(
        self, documents: list[Document]
    ) -> list[Document]:
        """
        Transform a list of raw documents.

        Override for batched / async implementations.  The default
        implementation calls :meth:`preprocess` sequentially.

        Parameters
        ----------
        documents:
            Raw documents to process.

        Returns
        -------
        list[Document]
        """
        return [self.preprocess(doc) for doc in documents]
