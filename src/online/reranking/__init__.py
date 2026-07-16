"""
online.reranking – Retrieval reranking

Re-scores and filters the list of retrieved chunks before answer generation.

Exports:
    BaseReranker         – abstract interface all rerankers must implement
    PassthroughReranker  – returns retrieved chunks unchanged (identity reranker)
    JinaReranker         – reranker powered by jinaai/jina-reranker-v3 (transformers)
"""

from .base import BaseReranker
from .rerankers import PassthroughReranker, JinaReranker

__all__ = ["BaseReranker", "PassthroughReranker", "JinaReranker"]
