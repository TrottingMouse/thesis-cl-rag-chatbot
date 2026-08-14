# Thesis CL-RAG Chatbot

A modular **Retrieval-Augmented Generation (RAG)** pipeline built for a bachelor's thesis on continual-learning chatbots. The system is designed to answer questions grounded in PDF documents (e.g. examination regulations, module handbooks) using a fully configurable, two-stage pipeline architecture.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Architecture](#pipeline-architecture)
  - [Offline Pipeline](#offline-pipeline)
  - [Online Pipeline](#online-pipeline)
- [Adding Documents](#adding-documents)
- [Adding New Components](#adding-new-components)
  - [Adding a Preprocessor](#adding-a-preprocessor)
  - [Adding a Chunker](#adding-a-chunker)
  - [Adding an Index Builder](#adding-an-index-builder)
  - [Adding a Query Processor](#adding-a-query-processor)
  - [Adding a Retriever](#adding-a-retriever)
  - [Adding a Reranker](#adding-a-reranker)
  - [Adding a Generator](#adding-a-generator)
- [What `main.py` Does](#what-mainpy-does)
- [Project Structure](#project-structure)

---

## Project Overview

The system is split into two pipelines that mirror a typical RAG deployment:

| Pipeline | Runs when… | Purpose |
|---|---|---|
| **Offline** | Documents change | Preprocess → chunk → embed → index |
| **Online** | A query arrives | Process query → retrieve → rerank → generate |

Both pipelines are fully **plug-and-play**: every stage (preprocessor, chunker, index builder, query processor, retriever, reranker, generator) is identified by a string name in a central registry and can be swapped via the YAML config file without touching any Python code.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** A GPU is strongly recommended for the Jina reranker and HuggingFace generator. If you have a CUDA 12 GPU, replace `faiss-cpu` with `faiss-gpu-cu12` in `requirements.txt`.

**API keys** (required by some components) are loaded from a `.env` file in the project root:

```ini
# .env
GOOGLE_API_KEY=your_gemini_key_here
```

---

## Quick Start

### 1. Add your documents

Place PDF files in the `documents/` folder and reference them in `config/config.yaml` (see [Adding Documents](#adding-documents)).

### 2. Configure the pipeline

Edit `config/config.yaml` to select the components you want (see [Configuration](#configuration)).

### 3. Run

```bash
python main.py
```

`main.py` builds both pipelines from the config, runs the offline indexing pass, queries the online pipeline with a pre-built evaluation set, and saves results and metrics to `storage/results/`.

### Minimal usage script

If you want to use the pipelines programmatically without the evaluation harness:

```python
from dotenv import load_dotenv
from src.factory import build_pipelines_from_config

load_dotenv()

# Build both pipelines from the YAML config
offline_pipeline, online_pipeline, config, pipeline_name = \
    build_pipelines_from_config("config/config.yaml")

# --- Offline pass (index your documents) ---
document_paths = config["documents"]   # list of PDF paths from the config
offline_pipeline.run(document_paths)   # preprocesses, chunks, and indexes

# --- Online pass (answer a question) ---
result = online_pipeline.query("What are the admission requirements?")
print(result.generation_result)

# For multiple questions in one batch (more efficient):
results = online_pipeline.batch_query([
    "What modules are compulsory?",
    "How many credits are needed to graduate?",
])
for r in results:
    print(r.generation_result)
```

---

## Configuration

All pipeline settings live in a single YAML file (`config/config.yaml`). The factory reads this file and wires up all components automatically.

```yaml
# Paths to the source PDF documents
documents:
  - "documents/PO.pdf"
  - "documents/MHB.pdf"
  - "documents/POges.pdf"

# Numeric / model hyperparameters for the offline pipeline
offline_config:
  embedding_model: "jinaai/jina-embeddings-v5-text-nano"
  chunking_params:
    chunk_size: 500
    overlap: 50

# Numeric / model hyperparameters for the online pipeline
online_config:
  top_k: 36                         # candidates fetched before reranking
  top_n: 12                         # candidates kept after reranking
  reranking_score_threshold: 0.00   # discard results below this score
  generation_model: "Qwen/Qwen3.5-2B"

# Component names (must match keys in src/registry.py)
offline_pipeline:
  preprocessors:
    - "GeminiMarkdownProcessor"
    - "DirectLLMProcessor"
  chunker: "FixedCharacterChunker"
  index_builder: "FaissIndexBuilder"

online_pipeline:
  query_processor: "CoTQueryProcessor"
  retriever: "FaissRetriever"
  reranker: "JinaReranker"
  generator: "HuggingfaceGenerator"
```

### How the config works with the factory

`build_pipelines_from_config(yaml_path)` in [`src/factory.py`](src/factory.py):

1. Loads the YAML and splits it into four sections: `offline_config`, `online_config`, `offline_pipeline`, `online_pipeline`.
2. Creates `OfflineConfig` and `OnlineConfig` dataclasses ([`src/config.py`](src/config.py)) from the `*_config` sections. These carry the numeric hyperparameters.
3. Looks up each component **name** in the central `COMPONENT_REGISTRY` ([`src/registry.py`](src/registry.py)) and instantiates the corresponding class.
4. Returns `(offline_pipeline, online_pipeline, config_dict, pipeline_name)`.

The `pipeline_name` is a short identifier built from the first six characters of each component name, used to name output files when running experiments.

---

## Pipeline Architecture

### Offline Pipeline

```
PDF files
    │
    ▼
[Preprocessors]  (chained, output of one feeds the next)
    │
    ▼
[Chunker]
    │
    ▼
[Index Builder]  ──► storage/index/  (persisted to disk)
```

**Stage 1 – Preprocessing** converts raw files into clean text. Multiple preprocessors can be chained: each preprocessor's output becomes the next preprocessor's input. Results are cached in `storage/cached_documents/` so that expensive LLM-based conversions are not repeated on subsequent runs.

**Stage 2 – Chunking** splits each processed document into smaller `Chunk` objects suitable for embedding.

**Stage 3 – Indexing** embeds the chunks with the configured embedding model and persists the index to `storage/index/`.

---

### Online Pipeline

```
User query
    │
    ▼
[Query Processor]  (e.g. CoT expansion, HyDE)
    │
    ▼
[Retriever]        top_k candidates from the FAISS index
    │
    ▼
[Reranker]         prunes to top_n most relevant chunks
    │
    ▼
[Generator]        produces the final answer string
```

**Stage 1 – Query Processing** normalises or augments the raw query (e.g. generates a chain-of-thought reasoning trace, or a hypothetical answer document).

**Stage 2 – Retrieval** queries the FAISS index and returns the `top_k` nearest-neighbour chunks.

**Stage 3 – Reranking** re-scores the candidates with a cross-encoder model and returns only the `top_n` chunks that exceed `reranking_score_threshold`.

**Stage 4 – Generation** formats a prompt from the query and context chunks, runs it through the configured language model, and returns the answer string.

---

## Adding Documents

1. **Place the PDF** in the `documents/` directory.
2. **Reference it** in `config/config.yaml` under the `documents` key:

```yaml
documents:
  - "documents/PO.pdf"
  - "documents/MHB.pdf"
  - "documents/my_new_document.pdf"   # ← add your file here
```

3. **Re-run the offline pipeline** (e.g. `python main.py`) so the new document is preprocessed, chunked, and indexed. Preprocessing results are cached, so only the new file will be processed from scratch.

> If you are using `GeminiMarkdownProcessor`, the converted Markdown for the new document must already exist in `storage/cached_documents/` before running — this preprocessor does not perform conversion itself.

---

## Adding New Components

Every component type has an **abstract base class** that specifies the interface to implement, and a **central registry** ([`src/registry.py`](src/registry.py)) where all concrete classes are registered by name.

The general workflow for adding any component is:

1. Implement the base class.
2. Register the new class in `src/registry.py`.
3. Reference the registration name in `config/config.yaml`.

### Adding a Preprocessor

Base class: [`src/offline/preprocessing/base.py → BasePreprocessor`](src/offline/preprocessing/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier (e.g. `"my_preprocessor"`) |
| `process_document(source_path: str) -> str` | Convert the raw file at `source_path` to clean text |

**Example:**

```python
# src/offline/preprocessing/my_preprocessor.py
from src.offline.preprocessing.base import BasePreprocessor

class MyPreprocessor(BasePreprocessor):
    @property
    def name(self) -> str:
        return "my_preprocessor"

    def process_document(self, source_path: str) -> str:
        with open(source_path) as f:
            return f.read()
```

Then register it in [`src/registry.py`](src/registry.py):

```python
from src.offline.preprocessing.my_preprocessor import MyPreprocessor

COMPONENT_REGISTRY = {
    ...
    "MyPreprocessor": MyPreprocessor,
}
```

And reference it in the config:

```yaml
offline_pipeline:
  preprocessors:
    - "MyPreprocessor"
```

---

### Adding a Chunker

Base class: [`src/offline/chunking/base.py → BaseChunker`](src/offline/chunking/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `chunk(document: Document) -> list[Chunk]` | Split one document into chunks |

Optionally override `chunk_batch(documents)` for a batched implementation.

**Chunker constructor kwargs** (e.g. `chunk_size`, `overlap`) are forwarded verbatim from `offline_config.chunking_params` in the YAML config.

---

### Adding an Index Builder

Base class: [`src/offline/indexing/base.py → BaseIndexBuilder`](src/offline/indexing/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `build(chunks: list[Chunk]) -> None` | Embed chunks and persist the index |

The constructor receives `storage_path` (and optionally `model_name`) from the factory.

---

### Adding a Query Processor

Base class: [`src/online/query/base.py → BaseQueryProcessor`](src/online/query/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `process(query: str) -> AugmentedQuery` | Transform one raw query |

Optionally override `process_batch(queries)` for efficiency. If the processor requires a language model, name it in the factory's special-case set so that `model_name` is injected automatically (see `build_online_pipeline` in [`src/factory.py`](src/factory.py)).

---

### Adding a Retriever

Base class: [`src/online/retrieval/base.py → BaseRetriever`](src/online/retrieval/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `retrieve(augmented_query: AugmentedQuery) -> list[RetrievalResult]` | Return up to `top_k` ranked results |

The constructor receives the `index_builder` instance and `top_k` from the factory.

---

### Adding a Reranker

Base class: [`src/online/reranking/base.py → BaseReranker`](src/online/reranking/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `rerank(augmented_query, candidates) -> list[RetrievalResult]` | Return up to `top_n` reranked results |

The constructor receives `top_n` from the factory. For `JinaReranker` the factory also passes `threshold`; add a similar special-case to `build_online_pipeline` if your reranker needs extra constructor arguments.

---

### Adding a Generator

Base class: [`src/online/generation/base.py → BaseGenerator`](src/online/generation/base.py)

**Required methods:**

| Method | Description |
|---|---|
| `name` (property) | Short string identifier |
| `generate(augmented_query, context) -> str` | Produce the final answer string |

Optionally override `generate_batch` for batched inference. Use the inherited `construct_prompt(query, context)` helper to format the standard German-language RAG prompt, or override it for a custom prompt.

---

## What `main.py` Does

`main.py` is the full evaluation entry point used for thesis experiments. It:

1. **Builds both pipelines** from `config/config.yaml`.
2. **Runs the offline pipeline** on the configured document paths, producing a FAISS index in `storage/index/`.
3. **Loads evaluation sets** from:
   - `storage/evaluation/qa_pairs.json` — positive question/answer pairs (questions that *should* be answerable from the documents).
   - `storage/evaluation/negative_qa_pairs.json` — negative question/answer pairs (questions that should be *rejected* as out-of-scope).
4. **Runs the online pipeline** (via `batch_query`) on both question sets and attaches the generated responses and retrieved contexts to each pair.
5. **Saves the enriched results** to:
   - `storage/results/positive/<pipeline_name>.json`
   - `storage/results/negative/<pipeline_name>.json`
6. **Evaluates** results using `src.evaluation.Evaluator`:
   - `evaluate_minimal()` on positive pairs (RAG faithfulness / correctness metrics).
   - `evaluate_rejection()` on negative pairs (abstention / hallucination metrics).
7. **Saves evaluation CSVs** alongside the result JSON files.

The `pipeline_name` suffix in all output filenames is derived from the first six characters of each component name, making it easy to compare results across different configurations.

---

## Project Structure

```
.
├── config/
│   └── config.yaml              # Pipeline configuration
├── documents/                   # Source PDF documents
├── experiments/                 # Standalone experiment scripts
│   ├── main_exp.py
│   ├── context_exp.py
│   ├── query_exp.py
│   ├── preprocessing_exp.py
│   └── grid_search.py
├── src/
│   ├── config.py                # OfflineConfig / OnlineConfig dataclasses
│   ├── factory.py               # Pipeline builders and experiment utilities
│   ├── registry.py              # Central component registry
│   ├── models/                  # Shared dataclass models (Document, Chunk, …)
│   ├── evaluation/              # RAGAS-based evaluation harness
│   ├── offline/
│   │   ├── pipeline.py          # OfflinePipeline orchestrator
│   │   ├── preprocessing/       # BasePreprocessor + concrete preprocessors
│   │   ├── chunking/            # BaseChunker + concrete chunkers
│   │   └── indexing/            # BaseIndexBuilder + FAISS / passthrough builders
│   └── online/
│       ├── pipeline.py          # OnlinePipeline orchestrator
│       ├── query/               # BaseQueryProcessor + processors
│       ├── retrieval/           # BaseRetriever + FAISS / passthrough retrievers
│       ├── reranking/           # BaseReranker + Jina / passthrough rerankers
│       └── generation/          # BaseGenerator + HuggingFace / passthrough generators
├── storage/
│   ├── cached_documents/        # Preprocessed text cache (auto-generated)
│   ├── index/                   # FAISS index artefacts (auto-generated)
│   ├── evaluation/              # QA pair evaluation sets
│   └── results/                 # Pipeline outputs and evaluation CSVs
├── main.py                      # Full evaluation entry point
└── requirements.txt
```
