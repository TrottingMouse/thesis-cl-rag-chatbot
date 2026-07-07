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
        paragraphs = re.split(r'\n', document.text)
        for paragraph in paragraphs:
            print(paragraph)
            print("--------------------------------")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        previous_table = False
        previous_module = False
        current_chunk = ""
        
        for n_paragraph, paragraph in enumerate(paragraphs):
            if n_paragraph == 0:
                continue

            if paragraph.startswith('|'):
                if not previous_table:
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                        text=current_chunk, 
                        chunker_name=self.name
                    ))
                    previous_table = True
                    current_chunk = paragraphs[n_paragraph-2] + "\n" + paragraphs[n_paragraph-1] + "\n"+ paragraphs[n_paragraph]

                    chunks.pop(len(chunks) - 1)
                    chunks.pop(len(chunks) - 1)
                else:
                    current_chunk += "\n" + paragraphs[n_paragraph]
            elif paragraph.startswith("**"):
                current_chunk += "\n" + paragraphs[n_paragraph]
            else:
                if previous_table:
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                        text="""**Abkürzungen im Studienverlaufsplan:**
                            * Modulprüfung = MP
                            * Studienleistung = SL
                            * ECTS-Leistungspunkte = LP
                            * Semesterwochenstunden = SWS
                            * Profilbildungsbereich = PBB
                            * Prüfungsnummer = Pnr.
                            """
                            + current_chunk, 
                        chunker_name=self.name
                    ))
                    previous_table = False
                    current_chunk = paragraphs[n_paragraph]
                elif previous_module:
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                        text=current_chunk, 
                        chunker_name=self.name
                    ))
                    previous_module = False
                    current_chunk = paragraphs[n_paragraph]
                    

                else:
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                        text=current_chunk, 
                        chunker_name=self.name
                    ))
                    current_chunk = paragraphs[n_paragraph]
            if n_paragraph == len(paragraphs) - 1:
                if previous_table:
                    text = """**Abkürzungen im Studienverlaufsplan:**
* Modulprüfung = MP
* Studienleistung = SL
* ECTS-Leistungspunkte = LP
* Semesterwochenstunden = SWS
* Profilbildungsbereich = PBB
* Prüfungsnummer = Pnr.
""" + current_chunk
                else:
                    text = current_chunk

                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{len(chunks)}", 
                    text=text, 
                    chunker_name=self.name
                ))

            
            
        return chunks

# write main function to test
if __name__ == "__main__":
    with open("storage/cached_documents/PO_markdown_gemini.txt") as f:
        text = f.read()
    document = Document(doc_id="test", text=text, source_path="")
    chunker = WholeTableParagraphChunker()
    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(chunk.text)
        print("----------------------------------------------------------------------------------------------------")
