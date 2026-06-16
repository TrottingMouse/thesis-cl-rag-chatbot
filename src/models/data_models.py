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
class Document:
    """A single document flowing through the pipeline."""

    # original nennen/entfernen?
    source_path: Path
    """Absolute path to the source file."""

    text: str = ""
    """Preprocessed plain-text content ready for chunking."""

    # evtl entfernen
    preprocessor_name: str = ""
    """Name of the :class:`BasePreprocessor` that produced this document."""

    doc_id: str = ""
    """Unique identifier for this document (e.g. MHB_markdown).
    Contains the raw document name and the preprocessing steps."""


@dataclass
class Chunk:
    """
    A single chunk produced by a chunker.

    Chunks are the atomic units that get embedded and stored in the index.
    """

    chunk_id: str
    """Unique identifier for this chunk (e.g. ``'{doc_id}_chunk_{n}'``)."""

    text: str
    """The actual text content of the chunk."""

    chunker_name: str = ""
    """Name of the :class:`BaseChunker` that produced this chunk."""

    # char_start: int | None = None
    # """Character offset of the chunk's start within the source document text."""

    # char_end: int | None = None
    # """Character offset of the chunk's end within the source document text."""

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
