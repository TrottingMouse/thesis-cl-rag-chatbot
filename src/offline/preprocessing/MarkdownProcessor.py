from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

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
        return "markdown"

    def preprocess(self, document: Document) -> Document:
        source_path = document.source_path
        cached_path = "storage/cached_documents/" + document.doc_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = document.doc_id
            )

        # convert the document
        result = self.converter.convert(source_path)
        markdown_text = result.document.export_to_markdown()

        # save the document as txt
        with open(cached_path, "w") as outfile:
            outfile.write(markdown_text)

        # Build and return the processed document
        return Document(
            source_path=document.source_path,
            text=markdown_text,
            preprocessor_name=self.name,
            doc_id=document.doc_id
        )

doc = Document(Path("documents/PO.pdf"), doc_id="PO_markdown")

markdown_processor = MarkdownProcessor()
processed_document = markdown_processor.preprocess(doc)
print(processed_document)