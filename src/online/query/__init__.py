"""
online.query – Query processing

Normalises and optionally expands the raw user query before retrieval.

Exports:
    BaseQueryProcessor    – abstract interface all query processors must implement
    NoProcessingProcessor – passes the query through unchanged (identity processor)
    HyDEQueryProcessor    – Hypothetical Document Embeddings via LLM
"""

from .base import BaseQueryProcessor
from .processors import NoProcessingProcessor, HyDEQueryProcessor

__all__ = ["BaseQueryProcessor", "NoProcessingProcessor", "HyDEQueryProcessor"]
