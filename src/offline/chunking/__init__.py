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
"""

from .base import BaseChunker
from .fixed_size import FixedCharacterChunker, FixedSentenceChunker, FixedParagraphChunker
from .tables import WholeTableParagraphChunker

__all__ = [
    "BaseChunker",
    "FixedCharacterChunker",
    "FixedSentenceChunker",
    "FixedParagraphChunker",
    "WholeTableParagraphChunker",
]
