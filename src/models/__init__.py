"""
models – Shared data models

Pydantic / dataclass models used by both the offline and online pipelines.

Exports:
    Document        – raw ingested document (text + metadata)
    Chunk           – text chunk produced by a chunker
    RetrievalResult – ranked retrieval hit (chunk + score)
    AugmentedQuery  – query enriched with retrieved context, ready for generation
"""

from .data_models import Document, Chunk, RetrievalResult, AugmentedQuery

__all__ = ["Document", "Chunk", "RetrievalResult", "AugmentedQuery"]
