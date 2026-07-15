from .base import BaseChunker
from src.models import Document, Chunk

import re

class WholeTableParagraphChunker(BaseChunker):
    """
    Chunks tables and MHB modules from markdown files as-is without splitting them.
    Text is split into paragraphs.
    Tables from PO get a prefix of abbreviations prepended.
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


# ---------------------------------------------------------------------------
# Abbreviation expansion map (PO tables)
# ---------------------------------------------------------------------------

_ABBREV_LEGEND = (
    "**Abkürzungen im Studienverlaufsplan:**\n"
    "* Modulprüfung = MP\n"
    "* Studienleistung = SL\n"
    "* ECTS-Leistungspunkte = LP\n"
    "* Semesterwochenstunden = SWS\n"
    "* Profilbildungsbereich = PBB\n"
    "* Prüfungsnummer = Pnr.\n"
)

# Ordered so that longer/more-specific tokens come first.
_PO_ABBREVS: list[tuple[str, str]] = [
    (r"\bSWS\b", "Semesterwochenstunden"),
    (r"\bECTS-LP\b", "ECTS-Leistungspunkte"),
    (r"(\d+)\s*LP\b", r"\1 ECTS-Leistungspunkte"),
    (r"\bLP\b", "ECTS-Leistungspunkte"),
    (r"\bMP\b", "Modulprüfung"),
    (r"\bSL\b", "Studienleistung"),
    (r"\bPBB\b", "Profilbildungsbereich"),
    (r"\bPnr\.", "Prüfungsnummer"),
]


def _expand_po_abbrevs(text: str) -> str:
    """Replace PO table abbreviations with their full written-out forms."""
    for pattern, replacement in _PO_ABBREVS:
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Helpers – sentence splitting for prose paragraphs
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split a prose paragraph into individual sentences.

    Splits on '.', '!' or '?' followed by whitespace and an uppercase letter
    or digit.  German shorthand like "z.B." or "d.h." is avoided by the
    lookbehind – we only split when the full stop is preceded by more than
    one character that is not a lone lowercase letter.
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ\d])', text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Helpers – MHB course-letter map
# ---------------------------------------------------------------------------

_COURSE_LETTER_RE = re.compile(
    r'([a-z]):\s*(.+?)\s*/\s*\d+',   # e.g. "a: Course name / 1234"
)


def _parse_course_map(lv_line: str) -> dict[str, str]:
    """
    Parse the *Lehrveranstaltungen und Lehrformen* field value and return a
    mapping  ``{'a': 'Einführung in die computationelle Logik (Seminar & Übung)',
                'b': 'Mathematische Grundlagen (Seminar)', ...}``.
    """
    # Strip any leading bold-field prefix "**FieldName:** " and trailing <br>
    # MHB bold fields have the format **Name:** (colon is INSIDE the ** markers)
    value = re.sub(r'^\*\*[^*]+:\*\*\s*', '', lv_line).strip()
    value = re.sub(r'<br>\s*$', '', value).strip()
    course_map: dict[str, str] = {}
    for m in _COURSE_LETTER_RE.finditer(value):
        letter = m.group(1)
        name = m.group(2).strip()
        course_map[letter] = name
    return course_map


def _replace_course_letters(text: str, course_map: dict[str, str]) -> str:
    """
    Replace ``(a)``, ``(b)``, … in *text* with the actual course name from
    *course_map*, e.g. ``(a)`` →
    ``(Einführung in die computationelle Logik (Seminar & Übung))``.
    """
    if not course_map:
        return text

    def _repl(m: re.Match) -> str:
        letter = m.group(1)
        return f"({course_map.get(letter, letter)})"

    letters_pattern = "|".join(re.escape(k) for k in sorted(course_map))

    # Pass 1: replace parenthesised "(a)" / "(b)" forms
    text = re.sub(rf'\(({letters_pattern})\)', _repl, text)

    # Pass 2: replace bare "a: " / "b: " segment-prefix forms that appear
    # either right after the bold field prefix "**FieldName:** " or after
    # a pipe separator " | ".
    def _repl_bare(m: re.Match) -> str:
        sep = m.group(1)    # "** " or " | "
        letter = m.group(2)
        return f"{sep}{course_map.get(letter, letter)}: "

    text = re.sub(rf'(\*\* |\| )({letters_pattern}): ', _repl_bare, text)
    return text


# ---------------------------------------------------------------------------
# SplitTableParagraphChunker
# ---------------------------------------------------------------------------

class SplitTableParagraphChunker(BaseChunker):
    """
    Fine-grained chunker that understands two distinct document structures:

    **PO (Prüfungsordnung / Studienverlaufsplan)**
    - Each semester table is split before every bold module-header row
      (``| **Module name** | …``).
    - Every sub-table chunk gets the semester title (e.g.
      ``### 2. Semester Bachelor Computerlinguistik``) prepended, followed
      by the abbreviation legend.
    - Abbreviations (MP, SL, LP, SWS, PBB, Pnr.) are expanded inline.

    **MHB (Modulhandbuch)**
    - Each module (``### Modul: …``) is split into one chunk per field line
      (``**Field:** value<br>``).
    - The module title is prepended to every field chunk.
    - When a module has multiple sub-courses (a, b, c, …), letter references
      such as ``(a)`` / ``(b)`` in field values are replaced by the actual
      course name parsed from the *Lehrveranstaltungen und Lehrformen* field.

    **Prose text** (introductory paragraphs that are neither tables nor
    module fields) is split at sentence boundaries.

    Document type is detected automatically from the content (presence of
    ``### Modul:`` headings → MHB; otherwise → PO).
    """

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "split_table_paragraph"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if self._is_mhb(text):
            raw_chunks = self._chunk_mhb(text)
        else:
            raw_chunks = self._chunk_po(text)

        return [
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_{i}",
                text=c,
                chunker_name=self.name,
            )
            for i, c in enumerate(raw_chunks)
            if c.strip()
        ]

    # ------------------------------------------------------------------
    # Document type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_mhb(text: str) -> bool:
        return bool(re.search(r'^### Modul:', text, re.MULTILINE))

    # ------------------------------------------------------------------
    # PO chunking
    # ------------------------------------------------------------------

    def _chunk_po(self, text: str) -> list[str]:
        """
        Split PO tables at bold module-header rows; expand abbreviations.
        Prose paragraphs are split at sentence level.
        """
        lines = text.splitlines()
        chunks: list[str] = []

        semester_title: str = ""    # e.g. "### 2. Semester Bachelor …"
        semester_summary: str = ""  # e.g. "*Summe: 4 Module, …*"
        table_header: str = ""      # column-name row
        table_separator: str = ""   # | :--- | :--- | …
        current_sub: list[str] = [] # rows accumulated for current sub-table
        in_table: bool = False

        def flush_sub() -> None:
            """Emit the current sub-table as one chunk."""
            if not current_sub:
                return
            ctx = (semester_title + "\n\n") if semester_title else ""
            body = _expand_po_abbrevs("\n".join(current_sub))
            chunks.append(ctx + body)
            current_sub.clear()

        for line in lines:
            stripped = line.strip()

            # ---- non-table lines ----------------------------------------
            if not stripped.startswith('|'):
                if in_table:
                    # Leaving the table block – flush pending sub-table
                    flush_sub()
                    in_table = False
                    semester_title = ""
                    semester_summary = ""
                    table_header = ""
                    table_separator = ""

                if stripped.startswith('### '):
                    semester_title = stripped
                elif re.match(r'^\*[^*].*[^*]\*$', stripped):
                    # Italic summary line: *Summe: …* (single asterisk, not bold)
                    semester_summary = stripped
                elif stripped:
                    for sentence in _split_sentences(stripped):
                        chunks.append(sentence)
                continue

            # ---- table lines --------------------------------------------
            if not in_table:
                # First row of a table → always the column-header row
                table_header = stripped
                in_table = True
                continue

            if re.match(r'^\|\s*:?-+:?\s*\|', stripped):
                # Alignment/separator row
                table_separator = stripped
                continue

            # Bold module-header row → start a new sub-table
            if re.match(r'^\|\s*\*\*', stripped):
                flush_sub()
                current_sub.append(stripped)
            else:
                # Regular course / exam row
                current_sub.append(stripped)

        # Flush anything remaining after the last line
        if in_table:
            flush_sub()

        return chunks

    # ------------------------------------------------------------------
    # MHB chunking
    # ------------------------------------------------------------------

    def _chunk_mhb(self, text: str) -> list[str]:
        """
        Split MHB content: one chunk per module field line, module title
        prepended.  Prose (non-module) paragraphs are sentence-split.
        """
        lines = text.splitlines()
        chunks: list[str] = []

        module_title: str = ""
        course_map: dict[str, str] = {}
        in_module: bool = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # ---- module heading (### Modul: …) --------------------------
            if stripped.startswith('### Modul:'):
                in_module = True
                module_title = stripped
                course_map = {}
                continue

            # ---- top-level section headings (## … / # …) ----------------
            if stripped.startswith('## ') or stripped.startswith('# '):
                in_module = False
                module_title = ""
                course_map = {}
                continue

            # ---- module field lines -------------------------------------
            if in_module and stripped.startswith('**'):
                # Parse Lehrveranstaltungen first so later fields can use it
                if stripped.startswith('**Lehrveranstaltungen und Lehrformen:**'):
                    course_map = _parse_course_map(stripped)

                field_text = _replace_course_letters(stripped, course_map)
                chunks.append(f"{module_title}\n{field_text}")
                continue

            # ---- prose lines --------------------------------------------
            if stripped:
                for sentence in _split_sentences(stripped):
                    chunks.append(sentence)

        return chunks


# ---------------------------------------------------------------------------
# Manual test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # import sys

    # target = sys.argv[1] if len(sys.argv) > 1 else "PO"

    target = "MHB"

    if target == "MHB":
        path = "storage/cached_documents/MHB_markdown_gemini.txt"
    else:
        path = "storage/cached_documents/PO_markdown_gemini.txt"

    with open(path) as f:
        raw = f.read()

    document = Document(doc_id="test", text=raw, source_path=path)
    chunker = SplitTableParagraphChunker()
    chunks = chunker.chunk(document)
    print(f"Total chunks: {len(chunks)}\n")
    for chunk in chunks:
        print(chunk.text)
        print("-" * 100)
