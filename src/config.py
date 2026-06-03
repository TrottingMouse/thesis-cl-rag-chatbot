"""
Pipeline configuration.

Centralises all tuneable parameters in one place so that experiment scripts
can construct pipelines from a single config object rather than hard-coding
values throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OfflineConfig:
    """Tunable parameters for the offline pipeline."""

    # Paths
    documents_dir: Path = Path("documents")
    """Directory containing the source PDF files."""

    storage_dir: Path = Path("storage")
    """Root directory for persisted indexes and processed documents."""

    # Chunking defaults (concrete chunkers may expose additional fields)
    chunk_size: int = 256
    """Target chunk size in tokens/characters (interpretation depends on chunker)."""

    chunk_overlap: int = 32
    """Overlap between consecutive chunks to preserve context across boundaries."""


@dataclass
class OnlineConfig:
    """Tunable parameters for the online pipeline."""

    top_k: int = 20
    """Number of candidates fetched from the index before reranking."""

    top_n: int = 5
    """Number of results returned after reranking (must be ≤ top_k)."""


@dataclass
class PipelineConfig:
    """
    Root configuration object.

    Pass an instance of this to factory functions or experiment scripts to
    ensure consistent settings across offline and online runs.
    """

    offline: OfflineConfig = field(default_factory=OfflineConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)

    experiment_name: str = "default"
    """Human-readable label for the current experiment run."""
