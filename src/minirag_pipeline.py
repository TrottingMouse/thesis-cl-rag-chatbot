"""
MiniRAG Adapter Pipeline
========================

Wraps a :class:`minirag.MiniRAG` instance so that it speaks the same
interface as :class:`~src.online.pipeline.OnlinePipeline`.

MiniRAG manages its own storage, chunking, entity extraction, and vector
index internally.  This adapter handles:

* **Offline phase** (:meth:`run`) – inserts documents into MiniRAG's KG index.
* **Online phase**  (:meth:`query` / :meth:`multiple_queries`) – delegates to
  ``MiniRAG.query()`` and wraps the string result in an
  :class:`~src.online.pipeline.OnlinePipelineResult`.

The same embedding model (jina-embeddings) and generation model (Qwen) that
are used by the rest of the pipeline are wired into MiniRAG here, so no
additional model weights are loaded.

Usage (config_minirag.yaml)::

    pipeline_mode: minirag

    minirag_config:
      working_dir: "storage/minirag_index"
      query_mode: "mini"      # "mini" | "light" | "naive"
      embedding_dim: 768      # must match jina-embeddings hidden_size
      chunk_token_size: 1200

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (heavy – only pulled in when the adapter is actually used)
# ---------------------------------------------------------------------------

def _load_minirag(
    working_dir: str,
    embedding_model_name: str,
    generation_model_name: str,
    embedding_dim: int,
    chunk_token_size: int,
):
    """Instantiate a MiniRAG object with the project's models."""
    from minirag import MiniRAG, QueryParam  # noqa: F401  (re-exported below)
    from minirag.llm.hf import hf_model_complete, hf_embed
    from minirag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer

    # Load the jina embedding model once (same weights as FaissIndexBuilder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trust_remote = "jina" in embedding_model_name.lower()
    _st_model = SentenceTransformer(
        embedding_model_name,
        device=device,
        trust_remote_code=trust_remote,
    )

    async def _embed(texts: list[str]) -> np.ndarray:
        """Async embedding shim that calls SentenceTransformer.encode."""
        embeddings = _st_model.encode(
            texts,
            convert_to_numpy=True,
            task="retrieval",
            prompt_name="document",
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    # MiniRAG tries to open its log file before creating working_dir – pre-create it.
    os.makedirs(working_dir, exist_ok=True)

    rag = MiniRAG(
        working_dir=working_dir,
        llm_model_func=hf_model_complete,
        llm_model_name=generation_model_name,
        llm_model_max_token_size=512,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=_embed,
        ),
        chunk_token_size=chunk_token_size,
    )
    return rag


# ---------------------------------------------------------------------------
# Result type (mirrors OnlinePipelineResult to allow drop-in usage)
# ---------------------------------------------------------------------------

@dataclass
class MiniRAGPipelineResult:
    """
    Return type of :class:`MiniRAGPipeline`.

    Mirrors :class:`~src.online.pipeline.OnlinePipelineResult` so that
    ``main.py`` can access ``result.generation_result`` and
    ``result.reranked_results`` uniformly.

    Note: MiniRAG does not expose the retrieved chunks that fed into its
    answer, so ``reranked_results`` is always an empty list.
    """

    generation_result: str
    """The answer produced by MiniRAG."""

    reranked_results: list = None  # type: ignore[assignment]
    """Always [] – MiniRAG does not expose retrieved chunks externally."""

    augmented_query: object = None
    retrieval_candidates: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.reranked_results is None:
            self.reranked_results = []
        if self.retrieval_candidates is None:
            self.retrieval_candidates = []


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class MiniRAGPipeline:
    """
    Adapter that exposes :class:`minirag.MiniRAG` behind the same public
    interface as :class:`~src.online.pipeline.OnlinePipeline`.

    Parameters
    ----------
    embedding_model_name:
        Path/name of the embedding model (e.g. ``"storage/models/jina-embeddings"``).
    generation_model_name:
        HuggingFace model ID used for answer generation
        (e.g. ``"Qwen/Qwen3.5-0.8B"``).
    working_dir:
        Directory where MiniRAG persists its KG index and caches.
    query_mode:
        MiniRAG query strategy: ``"mini"`` (default), ``"light"``, or ``"naive"``.
    embedding_dim:
        Output dimensionality of the embedding model (must match the model).
    chunk_token_size:
        Approximate token budget per text chunk during indexing.
    """

    name = "MiniRAGPipeline"

    def __init__(
        self,
        embedding_model_name: str,
        generation_model_name: str,
        working_dir: str = "storage/minirag_index",
        query_mode: str = "mini",
        embedding_dim: int = 768,
        chunk_token_size: int = 1200,
        preprocessor_names: list[str] | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.generation_model_name = generation_model_name
        self.working_dir = working_dir
        self.query_mode = query_mode
        self.embedding_dim = embedding_dim
        self.chunk_token_size = chunk_token_size
        self.preprocessor_names = preprocessor_names or []

        self._rag = None  # lazy-initialised on first use

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rag(self):
        """Return the MiniRAG instance, initialising it on first call."""
        if self._rag is None:
            logger.info(
                "Initialising MiniRAG | working_dir=%s | embed=%s | gen=%s",
                self.working_dir,
                self.embedding_model_name,
                self.generation_model_name,
            )
            self._rag = _load_minirag(
                working_dir=self.working_dir,
                embedding_model_name=self.embedding_model_name,
                generation_model_name=self.generation_model_name,
                embedding_dim=self.embedding_dim,
                chunk_token_size=self.chunk_token_size,
            )
        return self._rag

    def _build_preprocessors(self):
        """Instantiate the preprocessor chain from names via the registry."""
        from registry import get_class
        return [get_class(name)() for name in self.preprocessor_names]

    def _preprocess_documents(self, document_paths: list[str]) -> list[str]:
        """
        Run the preprocessing chain on the document paths and return the
        processed text strings, using the same cache layer as OfflinePipeline.

        If no preprocessors are configured, falls back to reading the file
        as plain UTF-8 text.
        """
        preprocessors = self._build_preprocessors()

        if not preprocessors:
            logger.warning(
                "No preprocessors configured for MiniRAGPipeline – reading raw files. "
                "PDFs will likely cause UnicodeDecodeError."
            )
            texts = []
            for path in document_paths:
                p = Path(path)
                if not p.exists():
                    logger.warning("File not found, skipping: %s", path)
                    continue
                texts.append(p.read_text(encoding="utf-8"))
            return texts

        # Run the first preprocessor on raw paths, then chain the rest
        docs = preprocessors[0].preprocess_from_paths(document_paths)
        for preprocessor in preprocessors[1:]:
            docs = [preprocessor.preprocess(doc) for doc in docs]

        logger.info(
            "Preprocessed %d document(s) through chain: %s",
            len(docs),
            " -> ".join(p.name for p in preprocessors),
        )
        return [doc.text for doc in docs]

    # ------------------------------------------------------------------
    # Offline phase – mirrors OfflinePipeline.run()
    # ------------------------------------------------------------------

    def run(self, document_paths: list[str]) -> None:
        """
        Preprocess then index all documents into MiniRAG's knowledge graph.

        Preprocessing uses the same chain (and cache) as the normal
        OfflinePipeline, so already-processed documents are loaded from
        ``storage/cached_documents/`` without re-running any LLM calls.

        Parameters
        ----------
        document_paths:
            Paths to the raw source files (e.g. PDFs).
        """
        rag = self._get_rag()

        texts = self._preprocess_documents(document_paths)

        logger.info("MiniRAGPipeline: indexing %d preprocessed document(s)…", len(texts))
        for i, text in enumerate(texts):
            logger.info("Inserting document %d/%d (%d chars)…", i + 1, len(texts), len(text))
            rag.insert(text)
        logger.info("MiniRAGPipeline: indexing complete.")

    # ------------------------------------------------------------------
    # Online phase – mirrors OnlinePipeline.query()
    # ------------------------------------------------------------------

    def query(self, raw_query: str) -> MiniRAGPipelineResult:
        """
        Answer a single question using MiniRAG's graph-guided retrieval.

        Parameters
        ----------
        raw_query:
            The raw question string.

        Returns
        -------
        MiniRAGPipelineResult
            ``generation_result`` contains the answer.
            ``reranked_results`` is always ``[]`` (MiniRAG does not expose
            intermediate chunks).
        """
        from minirag import QueryParam

        rag = self._get_rag()
        logger.info("MiniRAGPipeline.query | mode=%s | query=%r", self.query_mode, raw_query[:80])
        answer: str = rag.query(raw_query, param=QueryParam(mode=self.query_mode))
        return MiniRAGPipelineResult(generation_result=answer)

    def multiple_queries(self, raw_queries: list[str]) -> list[MiniRAGPipelineResult]:
        """
        Answer multiple questions sequentially.

        Parameters
        ----------
        raw_queries:
            List of raw question strings.

        Returns
        -------
        list[MiniRAGPipelineResult]
            One result per input query, in the same order.
        """
        return [self.query(q) for q in raw_queries]

    def describe(self) -> dict[str, str]:
        """Return a summary dict of the pipeline configuration."""
        return {
            "pipeline": "MiniRAGPipeline",
            "preprocessors": self.preprocessor_names,
            "query_mode": self.query_mode,
            "embedding_model": self.embedding_model_name,
            "generation_model": self.generation_model_name,
            "working_dir": self.working_dir,
        }
