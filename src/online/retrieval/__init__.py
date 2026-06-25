"""
online.retrieval – Chunk retrieval

Queries the FAISS index to retrieve the top-k most relevant chunks for
a processed user query.

Exports:
    BaseRetriever  – abstract interface all retrievers must implement
    FaissRetriever – retrieves chunks via nearest-neighbour search on a FAISS index
"""

from .base import BaseRetriever
from .retrievers import FaissRetriever

__all__ = ["BaseRetriever", "FaissRetriever"]
