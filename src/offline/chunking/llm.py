"""
LLM-based chunking strategies (Offline Pipeline – Step 2).

Implements the LumberChunker algorithm from the paper:
    "LumberChunker: Long-Form Narrative Document Segmentation"
    https://arxiv.org/abs/2406.17526

The chunker splits a document into semantically coherent segments by
iteratively asking a Gemini LLM to detect the first paragraph where the
content clearly shifts compared to the preceding context.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time

from google import genai
from google.genai import errors

from src.models import Document, Chunk
from .base import BaseChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "gemini-2.5-flash-lite"

# Approximate minimum token budget (words * 1.2) before asking the LLM.
# The original paper uses 550; keep as the default.
_DEFAULT_MIN_TOKENS = 550

_ANSWER_PATTERN = re.compile(r"Answer:\s*ID\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _approx_token_count(text: str) -> int:
    """Approximate token count: number of words * 1.2 (as in original paper)."""
    return round(1.2 * len(text.split()))


# ---------------------------------------------------------------------------
# LumberChunker
# ---------------------------------------------------------------------------


class LumberChunker(BaseChunker):
    """
    Semantically-aware chunker based on the LumberChunker paper.

    The algorithm:
    1. Split the document text into paragraphs (double-newline boundaries).
    2. Assign each paragraph a sequential integer ID.
    3. Slide a window over the paragraphs:
       a. Accumulate paragraphs until the approximate token count exceeds
          ``_DEFAULT_MIN_TOKENS``.
       b. Send the accumulated paragraphs (with their IDs) to the Gemini LLM
          together with the system prompt asking for the first paragraph where
          the content clearly shifts.
       c. Parse the returned ID and use it as the start of the next window.
    4. Assemble the final chunks by joining paragraphs between boundary IDs.
    """

    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # ------------------------------------------------------------------
    # BaseChunker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "lumber_chunker"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split *document* into semantically coherent chunks using LumberChunker.

        Parameters
        ----------
        document:
            A preprocessed document whose ``text`` is ready for splitting.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks.
        """
        text = document.text.strip()
        if not text:
            return []

        # 1. Split into non-empty paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []

        n = len(paragraphs)
        logger.info(
            "LumberChunker: document='%s' has %d paragraph(s).", document.doc_id, n
        )

        # Edge case: very short documents
        if n <= 5:
            return self._make_chunks(document, paragraphs, [n])

        # 2. Run the sliding-window LLM loop
        boundary_ids = self._find_boundaries(paragraphs)

        # 3. Assemble chunks
        return self._make_chunks(document, paragraphs, boundary_ids)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, window_text: str) -> str:
        return (
            "Du wirst als Eingabe ein deutsches Dokument mit Paragraphen erhalten, die durch"
            " 'ID XXXX: <text>' identifiziert werden.\n\n"
            "Aufgabe: Finde den ersten Paragraphen (nicht der erste), bei dem sich der Inhalt"
            " im Vergleich zu den vorherigen Paragraphen deutlich ändert.\n\n"
            "Ausgabe: Gib die ID des Paragraphen mit dem Inhaltswechsel im Format zurück:"
            " 'Answer: ID XXXX'.\n\n"
            "WICHTIGE REGELN:\n"
            "1. Erkläre NICHT deine Gedankengänge.\n"
            "2. Gib NUR die ID des Paragraphen mit dem Inhaltswechsel im Format zurück:"
            " 'Answer: ID XXXX'.\n"
            "3. Vermeide sehr lange Gruppen von Paragraphen. Strebe"
            " ein gutes Gleichgewicht zwischen der Identifizierung von Inhaltswechseln und"
            f" der Beibehaltung handhabbarer Gruppen an.\n\nDokument:\n {window_text}"
        )

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _find_boundaries(self, paragraphs: list[str]) -> list[int]:
        """
        Run the LumberChunker sliding-window loop and return the list of
        *exclusive* end indices that delimit each chunk.

        E.g. [3, 7, 10] means chunk 0 = paragraphs[0:3],
                                  chunk 1 = paragraphs[3:7],
                                  chunk 2 = paragraphs[7:10].
        """
        n = len(paragraphs)
        boundary_ids: list[int] = []
        chunk_start = 0  # current window start (paragraph index)

        while chunk_start < n - 5:
            # Grow window until we exceed _DEFAULT_MIN_TOKENS
            window_end = chunk_start  # exclusive
            while window_end < n - 1:
                window_end += 1
                window_text = "\n".join(
                    f"ID {chunk_start + k}: {paragraphs[chunk_start + k]}"
                    for k in range(window_end - chunk_start)
                )
                if _approx_token_count(window_text) >= _DEFAULT_MIN_TOKENS:
                    break

            # If the window grew to more than one paragraph, drop the last one
            # so the LLM sees a clean window without an incomplete boundary.
            effective_end = window_end  # exclusive upper bound for this batch
            if (effective_end - chunk_start) > 1:
                effective_end -= 1

            if (effective_end - chunk_start) == 1:
                logger.warning(
                    "LumberChunker: Paragraph ID %d exceeds token threshold alone. "
                    "Bypassing LLM and forcing a chunk boundary.", 
                    chunk_start
                )
                boundary_ids.append(effective_end)
                chunk_start = effective_end
                continue

            window_text = "\n".join(
                f"ID {chunk_start + k}: {paragraphs[chunk_start + k]}"
                for k in range(effective_end - chunk_start)
            )

            logger.debug(
                "LumberChunker: querying LLM for chunk_start=%d, "
                "effective_end=%d (~%d tokens).",
                chunk_start,
                effective_end,
                _approx_token_count(window_text),
            )

            prompt = self._build_prompt(window_text)
            llm_output = self._call_llm(prompt)

            # Advance the pointer (various error cases handled below)
            if llm_output == "_content_flagged_":
                # Safety block: skip one paragraph to avoid infinite loop
                chunk_start = effective_end
                logger.warning(
                    "LumberChunker: LLM flagged content near paragraph %d; "
                    "skipping to %d.",
                    chunk_start,
                    effective_end,
                )

            else:
                match = _ANSWER_PATTERN.search(llm_output)
                if match is None:
                    logger.warning(
                        "LumberChunker: could not parse LLM answer '%s'; "
                        "skipping window.",
                        llm_output,
                    )
                    chunk_start = effective_end
                else:
                    boundary_id = int(match.group(1))
                    logger.debug(
                        "LumberChunker: LLM returned boundary_id=%d.", boundary_id
                    )

                    # Guard: returned ID must be > chunk_start and <= effective_end
                    if boundary_id <= chunk_start or boundary_id > effective_end:
                        logger.warning(
                            "LumberChunker: boundary_id=%d out of expected range "
                            "[%d, %d]; advancing to effective_end.",
                            boundary_id,
                            chunk_start + 1,
                            effective_end,
                        )
                        boundary_id = effective_end

                    boundary_ids.append(boundary_id)
                    chunk_start = boundary_id

        # Append sentinel: the end of the document
        boundary_ids.append(n)
        return boundary_ids

    # ------------------------------------------------------------------
    # LLM call with retry logic
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Call the Gemini API with exponential-backoff retries.

        Returns
        -------
        str
            The model's text response, or ``'_content_flagged_'`` when the
            model refuses to answer due to a safety filter.
        """
        max_retries = 5
        base_delay = 2.0  # Starts with a 2-second delay

        for attempt in range(max_retries):
            try:
                interaction = self.client.interactions.create(
                    model=_MODEL,
                    input=prompt,
                    service_tier='flex'
                )
                return interaction.output_text

            except (errors.ServerError, errors.APIError) as e:
                # Safely extract the status code (503 for ServerError, 429 for APIError)
                status_code = getattr(e, 'code', None)

                # Check if the error is a temporary bottleneck we can wait out
                if status_code in [503, 429] or "503" in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s + 1-3 seconds of random jitter
                        sleep_time = (base_delay * (2 ** attempt)) + random.uniform(1.0, 3.0)
                        print(f"[API {status_code}] Backend stalled. Retrying in {sleep_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"LumberChunker: LLM call failed after {max_retries} attempts due to API limits. Last error: {e}")
                else:
                    # If it's a 400 Bad Request (e.g., token limit exceeded, bad JSON),
                    # retrying won't fix it. Raise the error immediately.
                    raise e

        # Safety net – should not be reached due to the raise above
        raise RuntimeError(f"LumberChunker: LLM call failed after {max_retries} attempts.")

    # ------------------------------------------------------------------
    # Chunk assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunks(
        document: Document,
        paragraphs: list[str],
        boundary_ids: list[int],
    ) -> list[Chunk]:
        """
        Assemble ``Chunk`` objects from *paragraphs* and *boundary_ids*.

        Parameters
        ----------
        document:
            Source document (for ``doc_id``).
        paragraphs:
            All paragraphs in order.
        boundary_ids:
            Exclusive end indices for each chunk (the last element must equal
            ``len(paragraphs)``).
        """
        chunks: list[Chunk] = []
        prev = 0
        for chunk_idx, end in enumerate(boundary_ids):
            chunk_paragraphs = paragraphs[prev:end]
            if not chunk_paragraphs:
                prev = end
                continue
            chunk_text = "\n\n".join(chunk_paragraphs)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    chunker_name="lumber_chunker",
                    metadata={
                        "para_start": prev,
                        "para_end": end - 1,
                    },
                )
            )
            prev = end
        return chunks
