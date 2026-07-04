"""
Embedding-based semantic chunker (Offline Pipeline – Step 2).

Implements the MaxMin chunking algorithm: sentences are grouped into
paragraphs by comparing each new sentence against all sentences already in
the current cluster.  The decision threshold is dynamically adjusted with a
sigmoid function based on the minimum pairwise similarity seen so far inside
the cluster, making the algorithm self-calibrating.

Reference: temporary.py (process_sentences)
"""

from __future__ import annotations

import logging

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Document, Chunk
from .base import BaseChunker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core algorithm (ported from temporary.py)
# ---------------------------------------------------------------------------


def _process_sentences(
    sentences: list[str],
    embeddings: np.ndarray,
    fixed_threshold: float = 0.6,
    c: float = 0.9,
    init_constant: float = 1.5,
) -> list[list[str]]:
    """
    Group *sentences* into paragraphs based on semantic similarity.

    Parameters
    ----------
    sentences:
        Ordered list of sentences to process.
    embeddings:
        Sentence embeddings of shape ``(n_sentences, embedding_dim)``.
    fixed_threshold:
        Minimum cosine-similarity a new sentence must have with the cluster
        to be appended to it (hard lower bound).
    c:
        Coefficient applied to the running minimum similarity when computing
        the dynamic threshold.  Values in (0, 1] dampen the threshold.
    init_constant:
        Multiplier used during the first comparison in a new cluster
        (cluster size == 1) to encourage the first extension.

    Returns
    -------
    list of list of str
        Each inner list is one paragraph (an ordered group of sentences).
    """

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    paragraphs: list[list[str]] = []
    current_paragraph: list[str] = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min: float = -float("inf")

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(
                embeddings[i].reshape(1, -1), cluster_embeddings
            )[0]

            # Dynamic threshold: scales up with cluster size and shrinks with
            # the minimum pairwise similarity in the cluster.
            adjusted_threshold = (
                pairwise_min * c * _sigmoid((cluster_end - cluster_start) - 1)
            )
            new_sentence_similarity = float(np.max(new_sentence_similarities))

            pairwise_min = min(float(np.min(new_sentence_similarities)), pairwise_min)
        else:
            # First extension: compare against single-sentence cluster.
            adjusted_threshold = 0.0
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1), cluster_embeddings
            )[0]
            pairwise_min = float(sim[0])
            new_sentence_similarity = init_constant * pairwise_min

        # Extend current cluster or start a new one.
        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float("inf")

    # Flush the last paragraph.
    paragraphs.append(current_paragraph)
    return paragraphs


# ---------------------------------------------------------------------------
# MaxMinChunker
# ---------------------------------------------------------------------------


class MaxMinChunker(BaseChunker):
    """
    Embedding-similarity chunker using the MaxMin algorithm.

    The chunker:
    1. Splits the document into sentences with spaCy.
    2. Embeds all sentences with a ``SentenceTransformer`` model.
    3. Groups sentences into paragraphs via :func:`_process_sentences`.
    4. Joins each sentence group into a single ``Chunk``.

    Parameters
    ----------
    embedding_model_name:
        Path or Hugging Face model identifier for the ``SentenceTransformer``
        embedding model (e.g. ``'storage/models/jina-embeddings'``).
    fixed_threshold:
        Hard lower bound on cosine similarity to include a sentence in the
        current cluster.
    c:
        Coefficient for the dynamic threshold adjustment (see algorithm).
    init_constant:
        Multiplier for the first comparison in a fresh cluster.
    spacy_model:
        spaCy model used for sentence segmentation.  Defaults to ``'de_core_news_sm'``
        (German small model).
    """

    def __init__(
        self,
        embedding_model_name: str,
        fixed_threshold: float = 0.6,
        c: float = 0.9,
        init_constant: float = 1.5,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.fixed_threshold = fixed_threshold
        self.c = c
        self.init_constant = init_constant

        # Load sentence encoder
        logger.info("MaxMinChunker: loading embedding model '%s'.", embedding_model_name)
        self._encoder = SentenceTransformer(embedding_model_name)

        # Load spaCy for sentence splitting
        try:
            self._nlp = spacy.load(
                "de_core_news_sm",
                exclude=["ner", "tagger", "lemmatizer", "attribute_ruler", "tok2vec"],
            )
        except OSError:
            self._nlp = spacy.blank("de_core_news_sm")

        if not self._nlp.has_pipe("sentencizer") and not self._nlp.has_pipe("parser"):
            self._nlp.add_pipe("sentencizer")

    # ------------------------------------------------------------------
    # BaseChunker interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        model_short = self.embedding_model_name.split("/")[-1]
        return (
            f"maxmin_{model_short}"
            f"_ft{self.fixed_threshold}"
            f"_c{self.c}"
            f"_ic{self.init_constant}"
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split *document* into semantically coherent chunks.

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

        # 1. Sentence segmentation
        doc = self._nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_0",
                    text=sentences[0],
                    chunker_name=self.name,
                    metadata={},
                )
            ]

        logger.info(
            "MaxMinChunker: embedding %d sentences for document '%s'.",
            len(sentences),
            document.doc_id,
        )

        # 2. Embed all sentences at once (batch for efficiency)
        embeddings: np.ndarray = self._encoder.encode(
            sentences, show_progress_bar=False, convert_to_numpy=True
        )

        # 3. Group sentences into paragraphs via MaxMin algorithm
        paragraphs = _process_sentences(
            sentences,
            embeddings,
            fixed_threshold=self.fixed_threshold,
            c=self.c,
            init_constant=self.init_constant,
        )

        # 4. Build Chunk objects
        chunks: list[Chunk] = []
        for chunk_idx, para_sentences in enumerate(paragraphs):
            chunk_text = " ".join(para_sentences)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    chunker_name=self.name
                )
            )

        logger.info(
            "MaxMinChunker: document '%s' → %d chunk(s).",
            document.doc_id,
            len(chunks),
        )
        return chunks
