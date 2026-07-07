from .base import BaseChunker
from src.models import Document, Chunk

import re

class WholeTableParagraphChunker(BaseChunker):
    """
    Chunks tables from markdown files as-is without splitting them.
    Text is split into paragraphs.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "whole_table"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Return a list containing chunks of tables plus context and non-table paragraphs 
        from a processed document.
        The table's descriptions/titles are prepended to the table content in the chunk.
        
        Parameters
        ----------
        document:
            A processed document whose `text` is ready for splitting.
        """
        chunks = []
        paragraphs = re.split(r'\n\s*\n', document.text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        previous_table = False
        
        for n_paragraph, paragraph in enumerate(paragraphs):
            if n_paragraph == 0:
                continue

            if paragraph.startswith('|'):
                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                    text=paragraphs[n_paragraph-1] + "\n" + paragraph, 
                    chunker_name=self.name
                ))
                previous_table = True
                continue
            if not previous_table:
                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                    text=paragraphs[n_paragraph-1], 
                    chunker_name=self.name
                ))
            if n_paragraph == len(paragraphs) - 1:
                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                    text=paragraph, 
                    chunker_name=self.name
                ))
            previous_table = False
            
            
        return chunks

# write main function to test
if __name__ == "__main__":
    with open("storage/cached_documents/MH_markdown_gemini.txt") as f:
        text = f.read()
    document = Document(doc_id="test", text=text, source_path="")
    chunker = WholeTableParagraphChunker()
    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(chunk.text)
        print("----------------------------------------------------------------------------------------------------")
