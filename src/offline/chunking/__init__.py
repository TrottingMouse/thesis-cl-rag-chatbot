"""
offline.chunking – Text chunking strategies

Splits preprocessed documents into overlapping text chunks ready for
embedding and indexing.

Exports:
    BaseChunker             – abstract interface all chunkers must implement
    FixedCharacterChunker   – splits text into fixed-size character windows
    FixedSentenceChunker    – splits text into fixed-size sentence windows
    FixedParagraphChunker   – splits text into fixed-size paragraph windows
    WholeTableParagraphChunker – keeps whole tables as single chunks
    LumberChunker           – LLM-based semantic chunker (LumberChunker paper)
    MaxMinChunker           – embedding-based semantic chunker (MaxMin algorithm)
"""

from .base import BaseChunker
from .fixed_size import FixedCharacterChunker, FixedSentenceChunker, FixedParagraphChunker
from .tables import WholeTableParagraphChunker
from .llm import LumberChunker
from .semantic import MaxMinChunker

__all__ = [
    "BaseChunker",
    "FixedCharacterChunker",
    "FixedSentenceChunker",
    "FixedParagraphChunker",
    "WholeTableParagraphChunker",
    "LumberChunker",
    "MaxMinChunker",
]
