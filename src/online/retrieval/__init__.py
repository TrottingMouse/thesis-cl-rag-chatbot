"""
online.retrieval – Chunk retrieval

Queries the index to retrieve the most relevant chunks for a processed
user query.

Exports:
    BaseRetriever         – abstract interface all retrievers must implement
    FaissRetriever        – retrieves chunks via nearest-neighbour search on a FAISS index
    PassthroughRetriever  – returns all indexed chunks for any query (baseline)
"""

from .base import BaseRetriever
from .retrievers import FaissRetriever
from .passthrough import PassthroughRetriever

__all__ = ["BaseRetriever", "FaissRetriever", "PassthroughRetriever"]
