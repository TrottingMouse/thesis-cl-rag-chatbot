from pathlib import Path

from docling.document_converter import DocumentConverter

from src.models import Document
from src.offline.preprocessing.base import BasePreprocessor


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
        return "markdown_docling"

    def preprocess(self, document: Document) -> Document:
        source_path = document.source_path

        # Convert the document from the file system
        result = self.converter.convert(source_path)
        markdown_text = result.document.export_to_markdown()

        # Build and return the processed document
        return Document(
            source_path=document.source_path,
            text=markdown_text,
            preprocessor_name=self.name
        )

doc = Document(Path("documents/25BA_Cli_FAEntwurfHP.pdf"))

markdown_processor = MarkdownProcessor()
processed_document = markdown_processor.preprocess(doc)
print(processed_document)