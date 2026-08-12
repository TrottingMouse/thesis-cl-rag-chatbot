from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
import pdfplumber
import re

from src.models import Document
from .base import BasePreprocessor


class GeminiMarkdownProcessor(BasePreprocessor):
    """
    Loads documents pre-converted to Markdown by Gemini (external conversion step).

    Does not perform any conversion itself – :meth:`process_document` raises
    ``FileNotFoundError`` to signal that the document must be in the cache.
    """

    @property
    def name(self) -> str:
        return "markdown_gemini"

    def process_document(self, source_path: str) -> str:
        raise FileNotFoundError(
            f"Gemini markdown document not found in cache. "
            f"Run the external Gemini conversion first."
        )


class DoclingMarkdownProcessor(BasePreprocessor):
    """
    Converts raw documents (e.g. PDFs) to Markdown using Docling.

    The Docling ``DocumentConverter`` (which loads heavyweight TableFormer AI
    models) is initialised **lazily** on the first cache miss, so instantiating
    this preprocessor when all documents are already cached is free.
    """

    def __init__(self):
        self._converter = None  # lazy – initialised on first cache miss

    def _build_converter(self):
        """Construct and return a configured Docling DocumentConverter."""
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.table_structure_options.do_cell_matching = False
        return DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @property
    def name(self) -> str:
        return "markdown_docling"

    def process_document(self, source_path: str) -> str:
        if self._converter is None:
            self._converter = self._build_converter()
        result = self._converter.convert(source_path)
        return result.document.export_to_markdown()


class RawTextProcessor(BasePreprocessor):
    """Extracts plain text from PDF documents using pdfplumber."""

    @property
    def name(self) -> str:
        return "raw_text"

    def process_document(self, source_path: str) -> str:
        text = ""
        with pdfplumber.open(source_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + '\n'
        return text