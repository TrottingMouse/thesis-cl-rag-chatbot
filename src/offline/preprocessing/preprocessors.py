from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

import pdfplumber

from src.models import Document
from src.offline.preprocessing.base import BasePreprocessor
import re
from google import genai

class GeminiMarkdownProcessor(BasePreprocessor):
    """
    A dummy preprocessor. Just loads the documents converted by Gemini 3.1 Pro.
    """
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "markdown_gemini"

    def process_document(self, source_path: str) -> str:
        # This preprocessor only works from cache; actual conversion is done externally.
        raise FileNotFoundError(
            f"Gemini markdown document not found in cache. "
            f"Run the external Gemini conversion first."
        )

class DoclingMarkdownProcessor(BasePreprocessor):
    """
    A preprocessor that uses Docling to convert raw documents (e.g. PDFs)
    into Markdown.
    """

    def __init__(self):
        super().__init__()
        # Initialize the Docling converter
        pipeline_options = PdfPipelineOptions(do_table_structure=True)

        # 1. Force the high-accuracy model variant
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        # 2. Tell it to rely on its visual prediction rather than messy PDF text cells
        pipeline_options.table_structure_options.do_cell_matching = False 

        # Pass options into your converter
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @property
    def name(self) -> str:
        return "markdown_docling"

    def process_document(self, source_path: str) -> str:
        result = self.converter.convert(source_path)
        return result.document.export_to_markdown()


class RawTextProcessor(BasePreprocessor):
    """
    A preprocessor that converts PDF documents to raw text documents.
    """

    def __init__(self):
        super().__init__()

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