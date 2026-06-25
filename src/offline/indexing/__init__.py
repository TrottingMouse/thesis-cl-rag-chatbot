"""
offline.indexing – FAISS index construction

Embeds text chunks and builds/persists a FAISS vector index for
similarity search during the online pipeline.

Exports:
    BaseIndexBuilder  – abstract interface all index builders must implement
    FaissIndexBuilder – builds and persists a FAISS flat-L2 index
"""

from .base import BaseIndexBuilder
from .indexing import FaissIndexBuilder

__all__ = ["BaseIndexBuilder", "FaissIndexBuilder"]
