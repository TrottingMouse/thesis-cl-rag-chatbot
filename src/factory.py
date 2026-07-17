"""
Factory helpers for building RAG pipeline components and running experiments.

Public API
----------
load_yaml_config(file_path)
    Load a YAML file and return it as a dict.

build_preprocessors(names)
    Instantiate a list of preprocessors from registry names.

build_chunker(name, **kwargs)
    Instantiate a chunker from a registry name.

build_index_builder(name, storage_path, embedding_model)
    Instantiate an index builder from a registry name.

build_online_pipeline(cfg, index_builder, top_k, top_n, generation_model)
    Assemble a fully-wired OnlinePipeline from a pipeline config dict.

build_offline_pipeline(preprocessor_names, chunker_name, index_builder_name,
                       storage_path, embedding_model, **chunker_kwargs)
    Assemble a fully-wired OfflinePipeline.

run_queries(online_pipeline, queries, qa_pairs_template)
    Execute multiple_queries and attach results to a deep copy of the QA template.

write_summary_csv(summary_path, rows)
    Write a list of result dicts to a CSV, collecting all keys dynamically.

build_pipelines_from_config(yaml_path)
    Legacy convenience wrapper – builds both pipelines from a YAML config file.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import yaml
import logging
import torch

from src.registry import get_class
from src.config import OfflineConfig, OnlineConfig
from src.offline.pipeline import OfflinePipeline
from src.online.pipeline import OnlinePipeline



# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(file_path: str) -> dict:
    """Load a YAML file and return its contents as a plain dict."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_preprocessors(names: list[str]):
    """Instantiate a list of preprocessors from their registry names."""
    return [get_class(name)() for name in names]


def build_chunker(name: str, **kwargs):
    """
    Instantiate a chunker from its registry name.

    Extra keyword arguments are forwarded verbatim to the chunker constructor,
    so callers can pass ``chunk_size``, ``overlap``, ``c``, etc.
    """
    return get_class(name)(**kwargs)


def build_index_builder(name: str, storage_path: Path, embedding_model: str):
    """
    Instantiate an index builder from its registry name.

    ``PassthroughIndexBuilder`` does not accept a model name; all other
    builders receive ``storage_path`` and ``model_name``.
    """
    IndexBuilderClass = get_class(name)
    if name == "PassthroughIndexBuilder":
        return IndexBuilderClass(storage_path=storage_path)
    return IndexBuilderClass(storage_path=storage_path, model_name=embedding_model)


def build_offline_pipeline(
    preprocessor_names: list[str],
    chunker_name: str,
    index_builder_name: str,
    storage_path: Path,
    embedding_model: str,
    **chunker_kwargs,
) -> OfflinePipeline:
    """
    Assemble a fully-wired OfflinePipeline.

    Parameters
    ----------
    preprocessor_names:
        Ordered list of registry names for the preprocessor chain.
    chunker_name:
        Registry name of the chunker to use.
    index_builder_name:
        Registry name of the index builder to use.
    storage_path:
        Directory where the index will be persisted.
    embedding_model:
        Name/path of the embedding model passed to the index builder.
    **chunker_kwargs:
        Additional keyword arguments forwarded to the chunker constructor
        (e.g. ``chunk_size``, ``overlap``, ``c``, ``fixed_threshold``).
    """
    preprocessors = build_preprocessors(preprocessor_names)
    chunker = build_chunker(chunker_name, **chunker_kwargs)
    index_builder = build_index_builder(index_builder_name, storage_path, embedding_model)
    return OfflinePipeline(preprocessors, chunker, index_builder)


def build_online_pipeline(
    cfg: dict,
    index_builder,
    top_k: int,
    top_n: int,
    generation_model: str,
) -> OnlinePipeline:
    """
    Assemble a fully-wired OnlinePipeline from a pipeline config section.

    Parameters
    ----------
    cfg:
        The ``online_pipeline`` sub-dict from the YAML config, containing keys
        ``query_processor``, ``retriever``, ``reranker``, and ``generator``.
    index_builder:
        An already-populated index builder instance (produced by the offline pipeline).
    top_k:
        Number of candidates to retrieve before reranking.
    top_n:
        Number of results to keep after reranking.
    generation_model:
        Name/path of the generation model (used for ``HuggingfaceGenerator``
        and ``HyDEQueryProcessor``).
    """
    QueryProcessorClass = get_class(cfg["query_processor"])
    query_processor = (
        QueryProcessorClass(model_name=generation_model)
        if cfg["query_processor"] in {"HyDEQueryProcessor", "CoTQueryProcessor"}
        else QueryProcessorClass()
    )

    RetrieverClass = get_class(cfg["retriever"])
    retriever = (
        RetrieverClass(index_builder)
        if cfg["retriever"] == "PassthroughRetriever"
        else RetrieverClass(index_builder, top_k=top_k)
    )

    reranker = get_class(cfg["reranker"])(top_n=top_n)

    GeneratorClass = get_class(cfg["generator"])
    generator = (
        GeneratorClass(model_name=generation_model)
        if cfg["generator"] == "HuggingfaceGenerator"
        else GeneratorClass()
    )

    return OnlinePipeline(query_processor, retriever, reranker, generator)


# ---------------------------------------------------------------------------
# Experiment utilities
# ---------------------------------------------------------------------------

def run_queries(
    online_pipeline: OnlinePipeline,
    queries: list[str],
    qa_pairs_template: list[dict],
    batching: bool = True
) -> list[dict]:
    """
    Execute *queries* through *online_pipeline* and attach the generated
    answers and retrieved contexts to a **deep copy** of *qa_pairs_template*.
    Batching can be set to false for high memory usage (long queries).

    Returns
    -------
    list[dict]
        A copy of the template, with ``response`` and ``retrieved_contexts``
        fields populated for each item.
    """
    if batching:
        try:
            results = online_pipeline.batch_query(queries)
        except torch.OutOfMemoryError:
            logging.warning("Batching failed: Out of memory. Using single-query mode.")
            results = online_pipeline.multiple_queries(queries)
    else:
        results = online_pipeline.multiple_queries(queries)
    qa_pairs = copy.deepcopy(qa_pairs_template)
    for i, pipeline_result in enumerate(results):
        qa_pairs[i]["response"] = pipeline_result.generation_result
        qa_pairs[i]["retrieved_contexts"] = [
            result.chunk.text for result in pipeline_result.reranked_results
        ]
    return qa_pairs


def write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
    """
    Write *rows* to a CSV file at *summary_path*.

    Field names are collected from all rows in insertion order, so that runs
    reporting different metric sets are handled gracefully.
    """
    if not rows:
        return

    all_fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_fields.append(k)
                seen.add(k)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Legacy convenience wrapper (kept for backward-compatibility)
# ---------------------------------------------------------------------------

def build_pipelines_from_config(yaml_path: str):
    """
    Build both the offline and online pipelines from a single YAML config file.

    Returns
    -------
    tuple[OfflinePipeline, OnlinePipeline, dict, str]
        ``(offline_pipeline, online_pipeline, config_dict, pipeline_name)``
    """
    config = load_yaml_config(yaml_path)

    offline_kwargs = config.get("offline_config", {})
    online_kwargs = config.get("online_config", {})

    offline_config = OfflineConfig(**offline_kwargs)
    online_config = OnlineConfig(**online_kwargs)

    offline_components = config["offline_pipeline"]
    online_components = config["online_pipeline"]

    offline_pipeline = build_offline_pipeline(
        preprocessor_names=offline_components["preprocessors"],
        chunker_name=offline_components["chunker"],
        index_builder_name=offline_components["index_builder"],
        storage_path=Path("storage/index"),
        embedding_model=offline_config.embedding_model,
        **offline_config.chunking_params,
    )

    # The index_builder lives inside offline_pipeline; reuse it for online.
    online_pipeline = build_online_pipeline(
        cfg=online_components,
        index_builder=offline_pipeline.index_builder,
        top_k=online_config.top_k,
        top_n=online_config.top_n,
        generation_model=online_config.generation_model,
    )

    component_names = (
        offline_components["preprocessors"]
        + [offline_components["chunker"]]
        + [offline_components["index_builder"]]
        + [online_components["query_processor"]]
        + [online_components["retriever"]]
        + [online_components["reranker"]]
        + [online_components["generator"]]
    )
    pipeline_name = "_".join(name[:6] for name in component_names)

    return offline_pipeline, online_pipeline, config, pipeline_name