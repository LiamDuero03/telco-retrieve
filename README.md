# TelcoLens

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GSMA Open Telco AI](https://img.shields.io/badge/GSMA-Open%20Telco%20AI-red.svg)](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-GSMA-orange.svg)](https://huggingface.co/datasets/GSMA/telcolens-chunks)

An open, vendor-neutral RAG pipeline for telecom specifications — covering 7 standards bodies, 4.6M+ chunks and 1B+ tokens across O-RAN, 3GPP, ETSI, ITU-T, TM Forum, CAMARA, and GSMA.

Part of the [GSMA Open Telco AI](https://www.open-telco.ai/).

---

## Overview

Open Telco RAG enables engineers and researchers to ask natural language questions directly against authoritative telecom standards — with source citations, no hallucination, and full corpus coverage across all major SDOs.

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

## Cookbook

The Open Telco RAG Cookbook is available as a [Cookbook](https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf) and as interactive Jupyter notebooks in the [`notebooks/`](notebooks/) directory — one per corpus:

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
```

### Ingest options

```bash
open-telco-rag ingest oran \
  --data data/oran/ \
  --model bge \
  --chunk-size 1024 \
  --overlap 200 \
  --db-path chroma_db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/<source>` | Path to corpus directory |
| `--model` | `minilm` | Embedding model key |
| `--db-path` | `chroma_db` | ChromaDB storage path |
| `--chunk-size` | `1024` | Chunk size in characters |
| `--overlap` | `200` | Overlap between chunks in characters |
| `--no-resume` | off | Disable resume — re-ingest all files |
| `--use-marker` | off | Use Datalab Marker API for PDF preprocessing |
| `--collection` | `source_model` | Override collection name |

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
  --top-k 8 \
  --rerank \
  --reranker-model ms-marco-minilm \
  --rerank-top-k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | 5 / 8 | Chunks to retrieve |
| `--rerank` | off | Enable reranking after retrieval |
| `--reranker-model` | `ms-marco-minilm` | Reranker key from `configs/reranker_models.json` |
| `--rerank-top-k` | 5 | Chunks to keep after reranking |
| `--collection` | `source_model` | Override collection name |

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
| `--model` | `minilm` | Embedding model key |
| `--collection` | `corpus_model` | Override collection name |
| `--questions-file` | corpus default | Path to QnA JSON file |
| `--top-k` | 5 | Chunks to retrieve per question |
| `--rag-generator` | openrouter | LLM provider for answer generation |
| `--rag-generator-model` | — | Model for answer generation |
| `--generator` | openrouter | LLM provider for metric scoring |
| `--generator-model` | — | Model for metric scoring |
| `--rerank` | off | Enable reranking during evaluation |
| `--reranker-model` | `ms-marco-minilm` | Reranker model key |
| `--include-baseline` | off | Run no-retrieval baseline pass |
| `--leaderboard` | `results/leaderboard.json` | Leaderboard output path |

---

## Python API

```python
from open_telco_rag.sources.oran import ORANSource
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.generators.openrouter import OpenRouterGenerator
from open_telco_rag.core.pipeline import ingest, search, generate

# Ingest
source   = ORANSource()
embedder = SentenceTransformerEmbedder(model_key="bge")
store    = ChromaStore(path="chroma_db", collection_name="oran_bge")

ingest(source=source, store=store, embedder=embedder, data_dir="data/oran/")

# Search
results = search("E2 interface service models", store=store, embedder=embedder, top_k=5)

# Generate
generator = OpenRouterGenerator(model="meta-llama/llama-3.3-70b-instruct")
answer    = generate("What does WG3 define for the E2 interface?", results,
                     generator=generator)
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
   Chunker               chunk_markdown() or chunk_text()
        │
        ▼
   Embedder              SentenceTransformerEmbedder (local)
        │                general-purpose or OTel telecom-domain models
        ▼
   ChromaDB              persistent vector store
        │
        ▼
   Search                cosine similarity retrieval
        │                optional: cross-encoder or OTel reranking
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
├── open_telco_rag/
│   ├── types.py
│   ├── core/
│   │   ├── pipeline.py          # ingest, search, generate
│   │   ├── chunking.py          # chunk_text, chunk_markdown
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
│   └── eval_questions/          # evaluation question sets
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

## Contributing

Contributions welcome. Please open an issue before submitting a pull request for significant changes.

---

## Acknowledgements

This work is part of the GSMA Open Telco AI Initiative. For more information visit [open-telco-ai](https://www.open-telco.ai/).

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
