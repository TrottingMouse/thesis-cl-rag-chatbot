"""
online.query – Query processing

Normalises and optionally expands the raw user query before retrieval.

Exports:
    BaseQueryProcessor    – abstract interface all query processors must implement
    NoProcessingProcessor – passes the query through unchanged (identity processor)
"""

from .base import BaseQueryProcessor
from .processors import NoProcessingProcessor

__all__ = ["BaseQueryProcessor", "NoProcessingProcessor"]
