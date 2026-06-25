"""
online – Online (inference) pipeline

This sub-package implements the online query pipeline that answers user
questions using the FAISS index built by the offline pipeline.  The four
main stages are:

    1. query      – normalise / expand the raw user query
    2. retrieval  – retrieve candidate chunks from the FAISS index
    3. reranking  – re-score and filter retrieved chunks
    4. generation – produce the final answer from the augmented context

Exports:
    OnlinePipeline – orchestrates the full online pipeline end-to-end
"""

from .pipeline import OnlinePipeline

__all__ = ["OnlinePipeline"]
