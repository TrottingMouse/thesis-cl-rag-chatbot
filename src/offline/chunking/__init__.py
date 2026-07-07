"""
offline.chunking – Text chunking strategies

Splits preprocessed documents into overlapping text chunks ready for
embedding and indexing.

Exports:
    BaseChunker             – abstract interface all chunkers must implement
    FixedCharacterChunker   – splits text into fixed-size character windows
    FixedParagraphChunker   – splits text into fixed-size paragraph windows
    WholeTableParagraphChunker – keeps whole tables as single chunks
    LumberChunker           – LLM-based semantic chunker (LumberChunker paper)
    MaxMinChunker           – embedding-based semantic chunker (MaxMin algorithm)
    PassthroughChunker      – keeps each document as a single chunk (baseline)
"""

from .base import BaseChunker
from .fixed_size import FixedCharacterChunker, FixedParagraphChunker
from .tables import WholeTableParagraphChunker
from .llm import LumberChunker
from .semantic import MaxMinChunker
from .passthrough import PassthroughChunker

__all__ = [
    "BaseChunker",
    "FixedCharacterChunker",
    "FixedParagraphChunker",
    "WholeTableParagraphChunker",
    "LumberChunker",
    "MaxMinChunker",
    "PassthroughChunker",
]
