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

class DynamicTokenChunker(BaseChunker):
    """
    Splits a document into chunks by grouping sentences until a token limit is reached.
    If a single sentence exceeds the token limit, it recursively falls back to 
    token-level splitting for that sentence. Supports overlapping chunks to 
    preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 0):
        """
        Parameters
        ----------
        chunk_size:
            The maximum number of tokens per chunk.
        overlap:
            The maximum number of overlapping tokens between consecutive chunks.
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

        try:
            self.nlp = spacy.load("de_core_news_sm", exclude=["ner", "tagger", "lemmatizer", "attribute_ruler", "tok2vec"])
        except OSError:
            self.nlp = spacy.blank("de")
        
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")

    @property
    def name(self) -> str:
        return f"recursive_sentence_{self.chunk_size}_{self.overlap}"

    def _get_length(self, text: str, spacy_span=None) -> int:
        """Helper to get token length using spaCy's native tokenization."""
        if spacy_span is not None:
            return len(spacy_span)
        return len(self.nlp(text))

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks by accumulating sentences up to a token limit.

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
        current_chunk_sents = []
        current_length = 0
        chunk_idx = 0

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_len = self._get_length(sent.text, spacy_span=sent)

            # 1. Check if adding this sentence exceeds the token limit
            if current_length + sent_len > self.chunk_size and current_chunk_sents:
                # Yield the accumulated sentences as a chunk
                char_start = current_chunk_sents[0].start_char
                char_end = current_chunk_sents[-1].end_char
                chunk_text = text[char_start:char_end]

                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    chunker_name=self.name,
                    metadata={}
                ))
                chunk_idx += 1

                # Calculate overlap using tokens from the tail end of the current chunk
                overlap_length = 0
                overlap_sents = []
                for s in reversed(current_chunk_sents):
                    s_len = self._get_length(s.text, spacy_span=s)
                    if overlap_length + s_len <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_length += s_len
                    else:
                        break
                
                # Infinite loop protection (if overlap is exceptionally high)
                if len(overlap_sents) == len(current_chunk_sents):
                    dropped_sent = overlap_sents.pop(0)
                    overlap_length -= self._get_length(dropped_sent.text, spacy_span=dropped_sent)

                current_chunk_sents = overlap_sents
                current_length = overlap_length
                
                # Do not increment 'i'; re-evaluate the current sentence against the new buffer
                continue

            # 2. Recursive Fallback: The single sentence itself exceeds the max chunk size
            if not current_chunk_sents and sent_len > self.chunk_size:
                tokens = list(sent)
                t_idx = 0
                step = max(1, self.chunk_size - self.overlap)
                
                while t_idx < len(tokens):
                    t_end = min(t_idx + self.chunk_size, len(tokens))
                    
                    char_start = tokens[t_idx].idx
                    char_end = tokens[t_end - 1].idx + len(tokens[t_end - 1].text)
                    
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{chunk_idx}",
                        text=text[char_start:char_end],
                        chunker_name=self.name,
                        metadata={}
                    ))
                    chunk_idx += 1
                    t_idx += step
                
                i += 1
                continue

            # 3. Standard Accumulation: Sentence fits perfectly
            current_chunk_sents.append(sent)
            current_length += sent_len
            i += 1

        # 4. Handle any remaining sentences left in the buffer
        if current_chunk_sents:
            char_start = current_chunk_sents[0].start_char
            char_end = current_chunk_sents[-1].end_char
            
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}_chunk_{chunk_idx}",
                text=text[char_start:char_end],
                chunker_name=self.name,
                metadata={}
            ))

        return chunks

class FixedSentenceChunker(BaseChunker):
    """
    Splits a document into fixed-size chunks based on a sentence count.
    Supports overlapping chunks to preserve context across boundaries.
    """

    def __init__(self, chunk_size: int = 5, overlap: int = 2, spacy_model: str = "de_core_news_sm"):
        """
        Parameters
        ----------
        chunk_size:
            The maximum number of sentences per chunk.
        overlap:
            The number of sentences to overlap between consecutive chunks.
            Must be strictly less than chunk_size.
        spacy_model:
            The spaCy model to use. Defaults to 'de_core_news_sm' (German small model).
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
    fix = DynamicTokenChunker()
    with open("storage/cached_documents/MHB_paper_llm_processor.txt", "r") as f:
        text = f.read()
    doc = Document(
        source_path="storage/cached_documents/MHB_paper_llm_processor.txt",
        doc_id="test",
        text=text
    )

    chunks = fix.chunk(doc)

    for chunk in chunks:
        print(chunk.text)
        print("\n" + "="*50 + "\n")
            
        
if __name__ == "__main__":
    main()