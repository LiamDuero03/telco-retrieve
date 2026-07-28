# TelcoLens

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GSMA Open Telco AI](https://img.shields.io/badge/GSMA-Open%20Telco%20AI-red.svg)](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-GSMA-orange.svg)](https://huggingface.co/datasets/GSMA/telcolens-chunks)

An open, vendor-neutral RAG pipeline for telecom specifications, covering 7 standards bodies, 4.6M+ chunks and 1B+ tokens across O-RAN, 3GPP, ETSI, ITU-T, TM Forum, CAMARA, and GSMA.

Part of the [GSMA Open Telco AI](https://www.open-telco.ai/).

---

## Overview

TelcoLens enables engineers and researchers to ask natural language questions directly against authoritative telecom standards, with source citations, no hallucination, and full corpus coverage across all major SDOs.

```
open-telco-rag ask "What does O-RAN WG3 define for the E2 interface?" --source-type oran
open-telco-rag ask "What does 3GPP TS 28.312 define for intent management?" --source-type 3gpp
open-telco-rag ask "What are the ETSI ZSM closed-loop automation requirements?" --source-type etsi
```

---

## Supported Corpora

| Corpus | Source | Format | Status |
|--------|--------|--------|--------|
| O-RAN Alliance | O-RAN R003–R005, all WGs | PDF, DOCX | ✅ Available |
| 3GPP | TSpec-LLM dataset, Rel-8 through Rel-19 | Markdown | ✅ Available |
| ETSI | ENI, ZSM, MEC, NFV, SAI, F5G, RIS | PDF, YAML | ✅ Available |
| ITU-T | All 27 active series (G, Y, Q, X, M…) | PDF | ✅ Available |
| CAMARA | Fall25 — 75+ API repos | Markdown, YAML | ✅ Available |
| GSMA | OPG, IG, AID, SGP, RCS and more | PDF, DOCX | ✅ Available |
| TM Forum | TMF Open APIs, ODA Component specs ([tmforum-rand/tmf-otel](https://github.com/tmforum-rand/tmf-otel)) | YAML, JSON | ✅ Available |

---

## Pre-built Dataset

The full indexed corpus — 4.6M+ chunks across all 7 SDOs — is available on HuggingFace:

🤗 **[GSMA/telcolens-chunks](https://huggingface.co/datasets/GSMA/telcolens-chunks)**

Download individual corpora or the full dataset without running ingestion yourself:

```python
from datasets import load_dataset

# Load one corpus
ds = load_dataset("GSMA/telcolens-chunks", data_files="etsi.jsonl")

# Load all corpora
ds = load_dataset("GSMA/telcolens-chunks")
```

---

## Installation

```bash
pip install open-telco-rag[local-embed,pdf,docx]
```

For GPU-accelerated embedding (recommended for large corpus ingestion):

```bash
pip install open-telco-rag[local-embed,pdf,docx]
# Then install PyTorch with ROCm or CUDA — see https://pytorch.org/get-started/locally/
```

---

## Quick Start

### 1. Configure API keys

Create a `.env` file in your project root:

```bash
OPENROUTER_API_KEY=your_key_here      # for LLM generation via OpenRouter (Llama, Claude, GPT-4o, etc.)
ANTHROPIC_API_KEY=your_key_here       # optional — for direct Claude access
DATALAB_API_KEY=your_key_here         # optional — for Marker PDF preprocessing
```

### 2. Download a corpus

Download O-RAN specifications from [o-ran.org](https://www.o-ran.org/specifications) into `data/oran/`.

### 3. Ingest

```bash
open-telco-rag ingest oran --model bge --data data/oran/
```

Available embedding models:

| Key | Model | Dimensions | Notes |
|-----|-------|------------|-------|
| `minilm` | all-MiniLM-L6-v2 | 384 | Fast — good for development |
| `mpnet` | all-mpnet-base-v2 | 768 | Balanced quality and speed |
| `e5` | intfloat/e5-large-v2 | 1024 | High quality — recommended for production |
| `bge` | BAAI/bge-large-en-v1.5 | 1024 | Strong on technical domains |
| `otel-109m` | OTel-Embedding-109M | 768 | Telecom-domain fine-tuned — 109M params |
| `otel-0.6b` | OTel-Embedding-0.6B | 1024 | Telecom-domain fine-tuned — 0.6B params |

### 4. Search

```bash
open-telco-rag search "near-RT RIC control loop" --source-type oran --top-k 5
```

### 5. Ask a question

```bash
open-telco-rag ask "What interfaces does the Near-RT RIC expose?" --source-type oran
```

---

## GUI

TelcoLens ships with a Streamlit web UI (`app.py`) with two separate pipelines selectable from the sidebar:

```bash
streamlit run app.py
```

### 🔍 RAG pipeline
| Tab | Purpose |
|-----|---------|
| Home | Corpus overview and quick-start guide |
| Ask | Retrieval + LLM generation with confidence indicator |
| Search | Pure vector search, no LLM |
| Ingest | Ingest documents into ChromaDB from the UI |
| Evaluate | Run the full evaluation suite with live progress |
| Leaderboard | Compare models, embedders, and corpora |
| DB info | Inspect ChromaDB collections |

### 🧪 QnA Generation pipeline
| Tab | Purpose |
|-----|---------|
| Generate QnA | Build a generation pipeline (sampling strategy, question types, concurrency, summarization, validation) and generate draft evaluation questions |
| Review QnA | Review and edit draft or already-validated QnA sets (`configs/eval_questions/`), view source chunk text with document links, approve / correct / reject questions, save updates in place, and promote completed drafts to official evaluation sets |

The QnA Generation pipeline requires [telco-qna-generation](https://github.com/Znbne/telco-qna-generation) to be installed:

```bash
pip install -e ../telco-qna-generation
```

---

## Cookbook

The Open Telco RAG Cookbook is available as a [PDF](https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf) and as interactive Jupyter notebooks in the [`notebooks/`](notebooks/) directory — one per corpus:

| Notebook | Corpus |
|----------|--------|
| [`01_oran_quickstart.ipynb`](notebooks/01_oran_quickstart.ipynb) | O-RAN Alliance R005 |
| [`02_3gpp_quickstart.ipynb`](notebooks/02_3gpp_quickstart.ipynb) | 3GPP Release 8 through 19 |
| [`03_camara_quickstart.ipynb`](notebooks/03_camara_quickstart.ipynb) | CAMARA Fall25 |
| [`04_tmforum_quickstart.ipynb`](notebooks/04_tmforum_quickstart.ipynb) | TM Forum Open APIs & ODA Components |
| [`05_gsma_quickstart.ipynb`](notebooks/05_gsma_quickstart.ipynb) | GSMA Open Gateway |
| [`06_itu_quickstart.ipynb`](notebooks/06_itu_quickstart.ipynb) | ITU-T |
| [`07_etsi_quickstart.ipynb`](notebooks/07_etsi_quickstart.ipynb) | ETSI ISG groups |

Each notebook covers ingestion, search, question answering, and evaluation for its corpus.

---

## CLI Reference

```
open-telco-rag ingest <source>   Ingest documents into the vector database
open-telco-rag search <query>    Search the vector database
open-telco-rag ask <question>    Ask a question (search + generate)
open-telco-rag evaluate <corpus> Run evaluation suite
open-telco-rag info              Show database info
open-telco-rag delete <collection> Delete a collection from the vector database
```

### Ingest options

```bash
open-telco-rag ingest oran \
  --data data/oran/ \
  --model bge \
  --strategy sliding_window_tokens \
  --chunk-size 512 \
  --overlap 50 \
  --enrichment llm_metadata \
  --enrichment-model google/qwen-2.5-coder-32b \
  --distance-metric cosine \
  --db-path chroma_db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/<source>` | Path to corpus directory |
| `--model` | `minilm` | Embedding model key |
| `--db-path` | `chroma_db` | ChromaDB storage path |
| `--strategy` | `text_baseline` | Segmentation strategy registered in chunking_strategies.json |
| `--chunk-size` | `1024` | Chunk size in characters |
| `--overlap` | `200` | Overlap between chunks in characters |
| `--enrichment` | `none` | Pre-embedding strategy registry token mapped from enrichment_strategies.json |
| `--enrichment-model` | `google/qwen-2.5-coder-32b` | LLM model target utilized when running LLM-aided metadata parsing routines |
| `--distance-metric` | `l2` | Vector space index geometry calculation mapping: l2, cosine or ip |
| `--no-resume` | off | Disable resume — re-ingest all files |
| `--use-marker` | off | Use Datalab Marker API for PDF preprocessing |
| `--collection` | `source_model` | Override collection name |
| `--embed-backend` | `local` | Backend for computing vector embeddings: `local` (CPU/GPU) or `openrouter` (Cloud APIs) |
| `--batch-size` | 64 / 500 | Parallel payload size. Defaults to 64 locally, or 500 for high-throughput OpenRouter bulk ingestion |

Ingest also builds a per-collection SQLite FTS5 BM25 index alongside ChromaDB, enabling `--search-method bm25` and `--search-method hybrid` retrieval.



### Marker preprocessing

For table-heavy corpora (ETSI, TM Forum), the Marker API produces higher quality chunks by preserving table structure and headings:

```bash
open-telco-rag ingest etsi --model bge --use-marker --data data/etsi/
```

Requires `DATALAB_API_KEY` in `.env`. See [datalab.to](https://www.datalab.to) for pricing.

### Search and Ask options

```bash
open-telco-rag ask "What does TS 28.312 define for intent?" \
  --source-type 3gpp \
  --model bge \
  --search-method dense \
  --top-k 8 \
  --rerank \
  --reranker-model ms-marco-minilm \
  --rerank-top-k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source-type` | `oran` | Source type |
| `--model` | `minilm` | Embedding model key |
| `--db-path` | `chroma_db` | ChromaDB storage path |
| `--search-method` | `dense` | Multi-method search routing pass algorithm selection: `dense`, `bm25`, or `hybrid` |
| `--top-k` | 5 / 8 | Chunks to retrieve |
| `--rerank` | off | Enable reranking after retrieval |
| `--reranker-model` | `ms-marco-minilm` | Reranker key from `configs/reranker_models.json` |
| `--rerank-top-k` | 5 | Chunks to keep after reranking |
| `--collection` | `source_model` | Override collection name |
| `--embed-backend` | `local` | Backend for computing vector embeddings (`local` or `openrouter`) |
| `--resolve-parents` | off | Expand retrieved child chunks into full parent contexts |

### Reranker models

| Key | Model | Notes |
|-----|-------|-------|
| `ms-marco-minilm` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast, general-purpose |
| `ms-marco-large` | cross-encoder/ms-marco-MiniLM-L-12-v2 | Larger, more accurate |
| `bge-reranker` | BAAI/bge-reranker-large | Strong on technical domains |
| `otel-reranker-0.6b` | OTel-Reranker-0.6B | Telecom-domain fine-tuned (recommended) |
| `otel-reranker-4b` | OTel-Reranker-4B | Higher accuracy, requires GPU |

### Evaluate options

```bash
open-telco-rag evaluate 3gpp \
  --model bge \
  --db-path chroma_db \
  --collection 3gpp_r19_marker_bge \
  --questions-file configs/eval_questions/3gpp_r19_qna.json \
  --rag-generator openrouter \
  --rag-generator-model meta-llama/llama-3.3-70b-instruct \
  --generator openrouter \
  --generator-model anthropic/claude-sonnet-4.5 \
  --top-k 5 \
  --include-baseline \
  --output results/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `minilm` | Embedding model key. |
| `--db-path` | `chroma_db` | ChromaDB storage path. |
| `--collection` | `corpus_model` | Override collection name (e.g., `3gpp_r19_marker_bge`). |
| `--questions-file` | `configs/eval_questions/{corpus}.json` | Path to a custom QnA JSON file. |
| `--top-k` | `5` | Chunks to retrieve per question. |
| `--threshold` | `0.5` | Similarity threshold for `coverage_ratio` metric. |
| `--output` | `results` | Directory to save result JSON files. |
| `--delay` | `1.0` | Seconds between questions for rate-limiting API calls. |
| `--no-save` | `off` | Print summary only, do not save JSON results. |
| `--generator` | `openrouter` | LLM provider for evaluation metrics (`claude` or `openrouter`). |
| `--generator-model` | `None` | Model for evaluation metrics (Claude default: `claude-sonnet-4-20250514`, OpenRouter default: `meta-llama/llama-3.3-70b-instruct`). |
| `--rag-generator` | `None` | LLM provider for RAG answer generation (defaults to `--generator`). |
| `--rag-generator-model` | `None` | Model for RAG answer generation (defaults to `--generator-model`). |
| `--debug` | `off` | Log raw LLM responses for every metric call. |
| `--leaderboard` | `results/leaderboard.json` | Path to leaderboard JSON file. |
| `--no-leaderboard` | `off` | Do not update the leaderboard file. |
| `--include-baseline` | `off` | Run second pass with no retrieval and store baseline metrics alongside RAG metrics. |
| `--rerank` | `off` | Enable reranking during evaluation retrieval. |
| `--rerank-top-k` | `5` | Chunks to keep after reranking. |
| `--reranker-model` | `ms-marco-minilm` | Reranker model key from `configs/reranker_models.json` (telecom: `otel-reranker-0.6b`). |
| `--embed-backend` | `local` | Backend for computing vector embeddings (`local` or `openrouter`). |
| `--inject-metadata` | `off` | Simulate domain expert queries by prepending canonical source metadata to the question. |
| `--run-name` | `None` | Custom label to identify this evaluation run (e.g., `exp_prompt_v2`). |
| `--include-keywords` | `off` | Append question keywords to the search query. |
| `--resolve-parents` | `off` | Expand retrieved child chunks into full parent contexts. |

---

### Delete collection options

```bash
open-telco-rag delete my_collection --db-path chroma_db
```


| Flag | Default | Description |
|------|---------|-------------|
| `collection` | required | Name of the ChromaDB collection to permanently delete |
| `--db-path` | `chroma_db` | ChromaDB storage path |

### Info options

```bash
open-telco-rag info --db-path chroma_db --collection oran_bge
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db-path` | `chroma_db` | ChromaDB storage path |
| `--collection` | `None` | Collection name to inspect |


---

## Python API

```python
from open_telco_rag.sources.oran import ORANSource
from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.generators.openrouter import OpenRouterGenerator
from open_telco_rag.core.pipeline import ingest, search, generate

# Ingest (Using OpenRouter for High-Throughput Bulk Processing)
source   = ORANSource()
embedder = OpenRouterEmbedder(model_key="bge", batch_size=200) # Throttled safely for OpenAI limits
store    = ChromaStore(path="chroma_db", collection_name="oran_bge", metadata={"hnsw:space": "cosine"})
ingest(
    source=source, 
    store=store, 
    embedder=embedder, 
    data_dir="data/oran/",
    strategy="sliding_window_tokens",
    chunk_tokens=512,
    overlap_tokens=50,
    enrichment="llm_metadata",
    enrichment_model="google/qwen-2.5-coder-32b",
    batch_size=5000 # Feed 5,000 chunks concurrently to the embedder!
)

# Search
results = search("E2 interface service models", store=store, embedder=embedder, top_k=5, search_method="dense")

# Generate
generator = OpenRouterGenerator(model="meta-llama/llama-3.3-70b-instruct")
answer    = generate("What does WG3 define for the E2 interface?", results, generator=generator)
print(answer["answer"])
```

---

## Architecture

```
Documents (PDF / DOCX / YAML / Markdown)
        │
        ▼
   Source class          load() → clean() → metadata()
   (per SDO)             optional: Marker API for PDF
        │
        ▼
   Dynamic Chunker       Routes text to selected strategies (e.g. baseline, tokens, 
                         parent-child, lumberchunker) via configs
        │
        ▼
   Enrichment            Pre-embedding passes: Structural Tagging, Acronym Expansion 
                         or dynamic LLM Domain Expansion (metadata tracking)
        │
        ▼
   Embedder              SentenceTransformerEmbedder (local GPU/CPU)
                         OR OpenRouterEmbedder (Cloud API ThreadPool)
        |                 general-purpose or OTel telecom-domain models
        |
        ▼
   ChromaDB              Persistent vector store via configurable space metrics 
                         (L2 Euclidean geometric, Cosine angular, Inner Product)
        │
        ▼
   Search                Multi-method retrieval pipeline: Dense vector, Sparse keyword (BM25),
                         or Reciprocal Rank Fusion (Hybrid)
                         optional: cross-encoder or OTel reranking
        |
        ▼
   Generator             any LLM via OpenRouter or direct API
        │
        ▼
   Answer + sources
```

---

## Project Structure

```
open-telco-rag/
├── app.py                       # Streamlit GUI (RAG + QnA Generation pipelines)
├── open_telco_rag/
│   ├── types.py
│   ├── core/
│   │   ├── pipeline.py          # Master orchestrator tracking stats and logging
│   │   ├── chunking.py          # Implements multi-strategy splitters (Token, etc.)
│   │   ├── enrichment.py        # Implements metadata tagging, glossary, and LLM-aided expansion
│   │   ├── retrieval.py         # Implements Multi-Method retrieval passes (Dense, BM25, and RRF)
│   │   ├── cleaning.py
│   │   └── marker_loader.py     # Datalab Marker API mixin
│   ├── sources/                 # one class per SDO
│   ├── embedders/
│   │   ├── sentense_transformers.py
│   │   └── reranker.py          # CrossEncoderReranker, OTelReranker
│   ├── generators/
│   │   ├── claude.py
│   │   └── openrouter.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── questions.py
│   │   ├── results.py
│   │   └── metrics/
│   │       ├── generation.py
│   │       ├── retrieval.py
│   │       └── ground_truth.py
│   └── stores/
│       └── chroma.py
├── cli/
│   └── main.py
├── notebooks/                   # quickstart cookbook
├── configs/
│   ├── embedding_models.json    # embedder registry
│   ├── reranker_models.json     # reranker registry
|   ├── chunking_strategies.json     # Central registry for character/token splitters
|   ├── enrichment_strategies.json   # Central registry for tagging, glossary, & LLM expansion
│   └── eval_questions/          # evaluation question sets per corpus
└── pyproject.toml
```

---

## Extending the Pipeline

### Adding a new source

```python
from open_telco_rag.types import Source
from pathlib import Path

class MySource:
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, path: Path) -> str: ...
    def clean(self, text: str) -> str: ...

    def metadata(self, path: Path) -> dict:
        return {"source": path.name, "source_type": "my_source"}
```

### Adding a new embedding model

Add an entry to `configs/embedding_models.json`:

```json
"my-model": {
    "model_name": "org/model-name",
    "dimensions": 768,
    "description": "My custom embedding model"
}
```

Then use `--model my-model` in any CLI command.

### Adding a new reranker

Add an entry to `configs/reranker_models.json`:

```json
"my-reranker": {
    "model_name": "org/reranker-name",
    "description": "My custom reranker",
    "backend": "cross-encoder"
}
```

---

## Related

- [telco-qna-generation](https://github.com/Znbne/telco-qna-generation) — synthetic QnA generation toolkit for building evaluation datasets
- [GSMA/telcolens-chunks](https://huggingface.co/datasets/GSMA/telcolens-chunks) — 4.6M chunks, 1.09B tokens across 7 SDOs
- [telcolens-cookbook](https://github.com/Znbne/telcolens-cookbook) — worked examples per corpus

---

## Contributing

Contributions welcome. Please open an issue before submitting a pull request for significant changes.

---

## Acknowledgements

This work is part of the GSMA Open Telco AI Initiative. For more information visit [open-telco-ai](https://www.open-telco.ai/).

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
