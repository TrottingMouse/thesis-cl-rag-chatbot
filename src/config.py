"""
Pipeline configuration.

Centralises all tuneable parameters in one place so that experiment scripts
can construct pipelines from a single config object rather than hard-coding
values throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OfflineConfig:
    """Tunable parameters for the offline pipeline."""

    chunking_params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded verbatim to the active chunker's constructor.
    Keeping this as an open dict makes OfflineConfig agnostic to the specific
    chunker being used (e.g. chunk_size/overlap for fixed chunkers, c/fixed_threshold
    for MaxMinChunker, etc.)."""

    embedding_model: str = "jinaai/jina-embeddings-v5-text-nano"
    """The embedding model to use for the pipeline."""


@dataclass
class OnlineConfig:
    """Tunable parameters for the online pipeline."""

    top_k: int = 20
    """Number of candidates fetched from the index before reranking."""

    top_n: int = 5
    """Number of results returned after reranking (must be ≤ top_k)."""

    reranking_score_threshold: float = 0.1
    """Minimum reranking score a result must achieve to be included in the
    output of :class:`~src.online.reranking.JinaReranker`.
    Results below this threshold are discarded even if they are within top_n."""

    embedding_model: str = "jinaai/jina-embeddings-v5-text-nano"
    """The embedding model to use for the pipeline."""

    generation_model: str = "Qwen/Qwen3.5-2B"
    """The generator model to use for the pipeline."""


@dataclass
class PipelineConfig:
    """
    Root configuration object.

    Pass an instance of this to factory functions or experiment scripts to
    ensure consistent settings across offline and online runs.
    """

    offline: OfflineConfig = field(default_factory=OfflineConfig)
    """Configuration for the offline pipeline."""

    online: OnlineConfig = field(default_factory=OnlineConfig)
    """Configuration for the online pipeline."""
