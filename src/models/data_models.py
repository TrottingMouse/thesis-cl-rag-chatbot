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

    query_type: str = "query"
    """Depends on the format which the QueryProcessor provides. Either query or document."""


@dataclass
class RetrievalResult:
    """
    A single retrieval result returned by the retriever (and optionally reranker).

    Wraps a :class:`Chunk` together with the scores assigned during retrieval
    and, if applicable, reranking.
    """

    chunk: Chunk
    """The retrieved chunk."""

    retrieval_score: float
    """Similarity / relevance score assigned by the retriever (e.g. cosine similarity)."""

    reranking_score: float | None = None
    """Score assigned by a reranker.  ``None`` if no reranker was applied."""
