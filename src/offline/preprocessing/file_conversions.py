from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

import pdfplumber

from src.models import Document
from src.offline.preprocessing.base import BasePreprocessor
import re
import os
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

    def preprocess(self, document: Document) -> Document:
        new_id = document.doc_id + "_" + self.name
        cached_path = "storage/cached_documents/" + new_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = new_id
            )
        else:
            raise FileNotFoundError(f"Gemini markdown document not found at {cached_path}")

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

    def preprocess(self, document: Document) -> Document:
        source_path = document.source_path
        new_id = document.doc_id + "_" + self.name
        cached_path = "storage/cached_documents/" + new_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = document.doc_id + "_" + self.name
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


class RawTextProcessor(BasePreprocessor):
    """
    A preprocessor that converts PDF documents to raw text documents.
    """

    def __init__(self):
        super().__init__()
        

    @property
    def name(self) -> str:
        return "raw_text"



    def preprocess(self, document: Document) -> Document:
        source_path = document.source_path
        new_id = document.doc_id + "_" + self.name
        cached_path = "storage/cached_documents/" + new_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = new_id
            )

        # convert the document
        text = ""
        with pdfplumber.open(source_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + '\n'

        # save the document as txt
        with open(cached_path, "w") as outfile:
            outfile.write(text)

        # Build and return the processed document
        return Document(
            source_path=document.source_path,
            text=text,
            preprocessor_name=self.name,
            doc_id=new_id
        )

class LLMProcessor(BasePreprocessor):
    def __init__(self, model="gemini-3.1-flash-lite"):
        super().__init__()
        self.model = model
        

    @property
    def name(self) -> str:
        return "llm_" + self.model
    
    def preprocess(self, document: Document) -> Document:
        new_id = document.doc_id + "_" + self.name
        cached_path = "storage/cached_documents/" + new_id + ".txt"
        if os.path.exists(cached_path):
            with open(cached_path) as f:
                content = f.read()
            return Document(
                source_path=document.source_path,
                text=content,
                preprocessor_name=self.name,
                doc_id = new_id
            )

        # convert the document
        client = genai.Client()
        response = client.models.generate_content(
            model=self.model,
            contents=f"""
            
            """
        )
        text = response.text
        

        # save the document as txt
        with open(cached_path, "w") as outfile:
            outfile.write(text)

        # Build and return the processed document
        return Document(
            source_path=document.source_path,
            text=text,
            preprocessor_name=self.name,
            doc_id=new_id
        )

doc = Document(Path("documents/PO.pdf"), doc_id="PO")

# markdown_processor = MarkdownProcessor()
# processed_document = markdown_processor.preprocess(doc)
# print(processed_document)
# gemini_markdown_processor = GeminiMarkdownProcessor()
# # docling_markdown_processor = DoclingMarkdownProcessor()
# # raw_text_processor = RawTextProcessor()
# processed_document = gemini_markdown_processor.preprocess(doc)
# print(processed_document)
# processed_document = docling_markdown_processor.preprocess(doc)
# print(processed_document)
# processed_document = raw_text_processor.preprocess(doc)
# print(processed_document)
