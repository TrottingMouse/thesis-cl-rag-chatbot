"""
offline.preprocessing – Document preprocessing

Converts raw input files (PDF, HTML, plain text, …) into clean text
suitable for downstream chunking.

Exports:
    BasePreprocessor        – abstract interface all preprocessors must implement
    GeminiMarkdownProcessor – converts documents to Markdown using the Gemini API
    DoclingMarkdownProcessor – converts documents to Markdown using Docling
    RawTextProcessor        – passes raw text through without transformation
    PaperLLMProcessor       – LLM-based structured extraction for academic papers
    DirectLLMProcessor      – direct LLM call for general-purpose text extraction
"""

from .base import BasePreprocessor
from .preprocessors import GeminiMarkdownProcessor, DoclingMarkdownProcessor, RawTextProcessor
from .llm_processors import PaperLLMProcessor, DirectLLMProcessor

__all__ = [
    "BasePreprocessor",
    "GeminiMarkdownProcessor",
    "DoclingMarkdownProcessor",
    "RawTextProcessor",
    "PaperLLMProcessor",
    "DirectLLMProcessor",
]
