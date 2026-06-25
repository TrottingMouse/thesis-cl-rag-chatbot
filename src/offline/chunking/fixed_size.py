from __future__ import annotations

from src.models import Document, Chunk
from .base import BaseChunker

import re
import spacy



class FixedCharacterChunker(BaseChunker):
    """
    Splits a document into fixed-size chunks based on a character count.
    Supports overlapping chunks to preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Parameters
        ----------
        chunk_size:
            The maximum number of characters per chunk.
        overlap:
            The number of characters to overlap between consecutive chunks.
            Must be strictly less than chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"fixed_size_char_{self.chunk_size}_{self.overlap}"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single processed document into fixed-size character chunks.

        Parameters
        ----------
        document:
            A preprocessed document whose `text` is ready for splitting.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.
        """
        text = document.text
        text_length = len(text)

        if text_length == 0:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_idx = 0

        # Determine step size
        step = self.chunk_size - self.overlap

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            chunk_id = f"{document.doc_id}_chunk_{chunk_idx}"

            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunker_name=self.name,
                metadata={}
            )
            chunks.append(chunk)

            if end == text_length:
                break

            start += step
            chunk_idx += 1

        return chunks

class FixedSentenceChunker(BaseChunker):
    """
    Splits a document into fixed-size chunks based on a sentence count.
    Supports overlapping chunks to preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 5, overlap: int = 2, spacy_model: str = "xx"):
        """
        Parameters
        ----------
        chunk_size:
            The maximum number of sentences per chunk.
        overlap:
            The number of sentences to overlap between consecutive chunks.
            Must be strictly less than chunk_size.
        spacy_model:
            The spaCy model to use. Defaults to 'xx' (multilingual blank model)
            which is lightweight and uses a rule-based sentencizer.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.spacy_model = spacy_model

        try:
            self.nlp = spacy.load(spacy_model, exclude=["ner", "tagger", "lemmatizer", "attribute_ruler", "tok2vec"])
        except OSError:
            self.nlp = spacy.blank(spacy_model)
        
        if not self.nlp.has_pipe("sentencizer") and not self.nlp.has_pipe("parser"):
            self.nlp.add_pipe("sentencizer")

    @property
    def name(self) -> str:
        return f"fixed_size_sentence_{self.chunk_size}_{self.overlap}"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single processed document into fixed-size sentence chunks.

        Parameters
        ----------
        document:
            A preprocessed document whose `text` is ready for splitting.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.
        """
        text = document.text
        if not text:
            return []

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return []

        chunks: list[Chunk] = []
        start_idx = 0
        chunk_idx = 0
        step = self.chunk_size - self.overlap

        while start_idx < len(sentences):
            end_idx = min(start_idx + self.chunk_size, len(sentences))
            chunk_sents = sentences[start_idx:end_idx]

            char_start = chunk_sents[0].start_char
            char_end = chunk_sents[-1].end_char
            chunk_text = text[char_start:char_end]

            chunk_id = f"{document.doc_id}_chunk_{chunk_idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunker_name=self.name,
                metadata={}
            )
            chunks.append(chunk)

            if end_idx == len(sentences):
                break

            start_idx += step
            chunk_idx += 1

        return chunks


class FixedParagraphChunker(BaseChunker):
    """
    Splits a document into fixed-size chunks based on a paragraph count.
    Supports overlapping chunks to preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 1, overlap: int = 0):
        """
        Parameters
        ----------
        chunk_size:
            The maximum number of paragraphs per chunk.
        overlap:
            The number of paragraphs to overlap between consecutive chunks.
            Must be strictly less than chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be strictly positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"fixed_size_paragraph_{self.chunk_size}_{self.overlap}"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a single processed document into fixed-size paragraph chunks.

        Parameters
        ----------
        document:
            A preprocessed document whose `text` is ready for splitting.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.
        """
        text = document.text
        if not text:
            return []

        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[Chunk] = []
        start_idx = 0
        chunk_idx = 0
        step = self.chunk_size - self.overlap

        while start_idx < len(paragraphs):
            end_idx = min(start_idx + self.chunk_size, len(paragraphs))
            chunk_paras = paragraphs[start_idx:end_idx]
            chunk_text = "\n\n".join(chunk_paras)

            chunk_id = f"{document.doc_id}_chunk_{chunk_idx}"
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunker_name=self.name,
                metadata={}
            )
            chunks.append(chunk)

            if end_idx == len(paragraphs):
                break

            start_idx += step
            chunk_idx += 1

        return chunks


def main():
    fix = FixedSentenceChunker()
    with open("storage/cached_documents/MHB_raw_text.txt", "r") as f:
        text = f.read()
    doc = Document(
        source_path="storage/cached_documents/MHB_raw_text.txt",
        doc_id="test",
        text=text
    )

    chunks = fix.chunk(doc)

    for chunk in chunks:
        print(chunk.text)
        print("\n" + "="*50 + "\n")
            
        
if __name__ == "__main__":
    main()