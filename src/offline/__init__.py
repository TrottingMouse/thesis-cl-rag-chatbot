"""
offline – Offline (indexing) pipeline

This sub-package implements the offline processing pipeline that converts raw
documents into a searchable FAISS index.  The three main stages are:

    1. preprocessing  – convert raw files to clean text (Docling, Gemini, LLM)
    2. chunking       – split documents into overlapping text chunks
    3. indexing       – embed chunks and build/persist a FAISS index

Exports:
    OfflinePipeline – orchestrates the full offline pipeline end-to-end
"""

from .pipeline import OfflinePipeline

__all__ = ["OfflinePipeline"]
