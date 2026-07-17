"""
models – Shared data models

Pydantic / dataclass models used by both the offline and online pipelines.

Exports:
    Document         – raw ingested document (text + metadata)
    Chunk            – text chunk produced by a chunker
    AugmentedQuery   – query after processing/augmentation
    RetrievalResult  – retrieved chunk together with retrieval and reranking scores
"""

from .data_models import Document, Chunk, AugmentedQuery, RetrievalResult

__all__ = ["Document", "Chunk", "AugmentedQuery", "RetrievalResult"]
