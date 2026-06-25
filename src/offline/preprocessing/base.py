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

import os
import re

from src.models import Document


class BasePreprocessor(ABC):
    """
    Contract for all document preprocessors.

    Subclasses **must** implement :meth:`name` and :meth:`process_document`.
    :meth:`preprocess` is a template method that handles cache lookup, calls
    :meth:`process_document`, writes the result to cache, and returns the
    processed :class:`Document`.
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
    def process_document(self, source_path: str) -> str:
        """
        Convert the raw document at *source_path* into processed text.

        This method contains the actual conversion logic and is called by
        :meth:`preprocess` only when no cached result exists yet.

        Parameters
        ----------
        source_path:
            Path to the raw source file (e.g. a PDF).

        Returns
        -------
        str
            The processed text ready for chunking.
        """

    def load_file(self, doc_id: str) -> str | None:
        """
        Load the cached document text for *doc_id* from
        ``storage/cached_documents/<doc_id>.txt``.

        Returns
        -------
        str | None
            The file contents, or ``None`` if the cache file does not exist.
        """
        cached_path = "storage/cached_documents/" + doc_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                return f.read()
        return None

    def preprocess(self, document: Document) -> Document:
        """
        Transform a single raw document into a processed document.

        Checks the cache first; on a miss delegates to :meth:`process_document`,
        writes the result to cache, and returns a new :class:`Document`.

        Parameters
        ----------
        document:
            The raw document loaded from disk.

        Returns
        -------
        Document
            The cleaned/converted document ready for chunking.
        """
        new_id = document.doc_id + "_" + self.name
        cached_path = "storage/cached_documents/" + new_id + ".txt"

        content = self.load_file(new_id)
        if content is None:
            content = self.process_document(document.source_path)
            with open(cached_path, "w") as f:
                f.write(content)

        return Document(
            source_path=document.source_path,
            text=content,
            preprocessor_name=self.name,
            doc_id=new_id,
        )

    def preprocess_from_path(self, path: str) -> Document:
        document = Document(path, "", "", re.search(r".+\/(.+)\.[^.]+$", path).group(1))
        return self.preprocess(document)

    def preprocess_from_paths(self, paths: list[str]) -> list[Document]:
        """
        Transform a list of raw documents.

        Parameters
        ----------
        paths:
            Paths to the raw documents to process.

        Returns
        -------
        list[Document]
        """
        return [self.preprocess_from_path(doc) for doc in paths]
