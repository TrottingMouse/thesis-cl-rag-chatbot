"""
offline.indexing – Index construction

Embeds text chunks and builds/persists a queryable index for
similarity search during the online pipeline.

Exports:
    BaseIndexBuilder        – abstract interface all index builders must implement
    FaissIndexBuilder       – builds and persists a FAISS flat-L2 index
    PassthroughIndexBuilder – stores chunks without embedding (baseline)
"""

from .base import BaseIndexBuilder
from .indexing import FaissIndexBuilder
from .passthrough import PassthroughIndexBuilder

__all__ = ["BaseIndexBuilder", "FaissIndexBuilder", "PassthroughIndexBuilder"]
