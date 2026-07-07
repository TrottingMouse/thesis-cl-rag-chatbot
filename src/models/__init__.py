"""
models – Shared data models

Pydantic / dataclass models used by both the offline and online pipelines.

Exports:
    Document        – raw ingested document (text + metadata)
    Chunk           – text chunk produced by a chunker
    AugmentedQuery  – query after processing/augmentation
"""

from .data_models import Document, Chunk, AugmentedQuery

__all__ = ["Document", "Chunk", "AugmentedQuery"]
