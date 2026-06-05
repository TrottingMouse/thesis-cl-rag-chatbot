from pathlib import Path

from docling.document_converter import DocumentConverter

from src.models import Document
from src.offline.preprocessing.base import BasePreprocessor

import os

class MarkdownProcessor(BasePreprocessor):
    """
    A preprocessor that uses Docling to convert raw documents (e.g. PDFs)
    into Markdown.
    """

    def __init__(self):
        super().__init__()
        # Initialize the Docling converter
        self.converter = DocumentConverter()

    @property
    def name(self) -> str:
        return "markdown"

    def preprocess(self, document: Document) -> Document:
        source_path = document.source_path
        cached_path = "cached_documents/" + document.doc_id
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = document.doc_id
            )

        # Convert the document from the file system
        result = self.converter.convert(source_path)
        markdown_text = result.document.export_to_markdown()

        # save the document as txt
        with open(cached_path) as outfile:
            outfile.write(markdown_text)

        # Build and return the processed document
        return Document(
            source_path=document.source_path,
            text=markdown_text,
            preprocessor_name=self.name,
            doc_id=document.doc_id
        )

doc = Document(Path("documents/25BA_Cli_FAEntwurfHP.pdf"))

markdown_processor = MarkdownProcessor()
processed_document = markdown_processor.preprocess(doc)
print(processed_document)