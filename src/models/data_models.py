"""
Shared data models used across the offline and online pipelines.

All pipeline components communicate via these dataclasses so that
swapping one component (e.g. a chunker) never requires changing the
interface of any neighbouring component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Offline pipeline models
# ---------------------------------------------------------------------------


@dataclass
class RawDocument:
    """A document exactly as loaded from disk, before any preprocessing."""

    doc_id: str
    """Unique identifier derived from the file path (e.g. the stem)."""

    source_path: Path
    """Absolute path to the source file."""

    content: bytes
    """Raw binary content of the file."""

    mime_type: str = "application/octet-stream"
    """MIME type of the source file (e.g. 'application/pdf')."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value pairs attached by the loader (page count, author, …)."""


@dataclass
class ProcessedDocument:
    """
    A document after preprocessing.

    Preprocessing may convert tables to Markdown, extract propositions via
    an LLM/SLM, clean whitespace, etc.  The result is always plain text that
    a chunker can consume.
    """

    doc_id: str
    """Same identifier as the originating :class:`RawDocument`."""

    source_path: Path
    """Path to the originating file (preserved for provenance)."""

    text: str
    """Preprocessed plain-text content ready for chunking."""

    preprocessor_name: str = ""
    """Name of the :class:`BasePreprocessor` that produced this document."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value pairs forwarded from the raw document plus any
    metadata added during preprocessing."""


@dataclass
class Chunk:
    """
    A single chunk produced by a chunker.

    Chunks are the atomic units that get embedded and stored in the index.
    """

    chunk_id: str
    """Unique identifier for this chunk (e.g. ``'{doc_id}_chunk_{n}'``)."""

    doc_id: str
    """Identifier of the :class:`ProcessedDocument` this chunk originates from."""

    text: str
    """The actual text content of the chunk."""

    chunker_name: str = ""
    """Name of the :class:`BaseChunker` that produced this chunk."""

    char_start: int | None = None
    """Character offset of the chunk's start within the source document text."""

    char_end: int | None = None
    """Character offset of the chunk's end within the source document text."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value pairs (page number, section heading, …)."""


# ---------------------------------------------------------------------------
# Online pipeline models
# ---------------------------------------------------------------------------


@dataclass
class AugmentedQuery:
    """
    A query after processing/augmentation.

    Query processors may expand the query, rewrite it, decompose it into
    sub-queries, etc.
    """

    original_query: str
    """The raw query string provided by the user."""

    processed_queries: list[str]
    """
    One or more query strings to retrieve against.  A simple passthrough
    processor returns ``[original_query]``; HyDE or query expansion may
    return multiple variants.
    """

    processor_name: str = ""
    """Name of the :class:`BaseQueryProcessor` that produced this object."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra information (embeddings, decomposition tree, …)."""


@dataclass
class RetrievalResult:
    """
    A single candidate chunk returned by a retriever.

    Before reranking this holds the retrieval score; after reranking a
    reranker may overwrite :attr:`score` or add a ``rerank_score`` entry
    to :attr:`metadata`.
    """

    chunk: Chunk
    """The retrieved chunk."""

    score: float
    """Retrieval score (higher is more relevant; exact semantics depend on
    the retriever, e.g. cosine similarity or BM25 score)."""

    retriever_name: str = ""
    """Name of the :class:`BaseRetriever` that produced this result."""

    rank: int | None = None
    """Rank assigned after reranking (1 = most relevant)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra information (retrieved embedding, BM25 term scores, …)."""
