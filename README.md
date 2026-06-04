# Open Telco RAG

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GSMA Open Telco AI](https://img.shields.io/badge/GSMA-Open%20Telco%20AI-red.svg)](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-GSMA-orange.svg)](https://huggingface.co/GSMA)

An open, vendor-neutral RAG pipeline for telecom specifications — covering 7 standards bodies across O-RAN, 3GPP, ETSI, ITU-T, TM Forum, CAMARA, and GSMA.

Part of the [GSMA Open Telco AI Initiative](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/).

---

## Overview

Open Telco RAG enables engineers and researchers to ask natural language questions directly against authoritative telecom standards — with source citations, no hallucination, and full corpus coverage across all major SDOs.

```
open-telco-rag ask "What does O-RAN WG3 define for the E2 interface?" --source-type oran
open-telco-rag ask "What are the TM Forum L4 autonomy requirements?" --source-type tmforum
open-telco-rag ask "What does 3GPP TS 28.312 define for intent management?" --source-type 3gpp
```

---

## Supported Corpora

| Corpus | Source | Documents | Format |
|--------|--------|-----------|--------|
| O-RAN Alliance | O-RAN R003–R005, all WGs | PDF, DOCX | `oran` |
| 3GPP | TSpec-LLM dataset, Rel-18 | Markdown | `3gpp` |
| ETSI | ENI, ZSM, MEC, NFV, SAI, F5G, RIS | PDF, YAML | `etsi` |
| ITU-T | 14 active series (G, Y, Q, X, M...) | PDF | `itu` |
| TM Forum | TMF Open APIs, ODA | DOCX, PDF, YAML, JSON | `tmforum` |
| CAMARA | Fall25 — 75+ API repos | Markdown, YAML | `camara` |
| GSMA | OPG, IG, AID, SGP, RCS and more | PDF, DOCX | `gsma` |

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
ANTHROPIC_API_KEY=your_key_here       # for answer generation via Claude
OPENROUTER_API_KEY=your_key_here      # for open-source LLM generation (Llama, Mistral, etc.)
DATALAB_API_KEY=your_key_here         # optional — for Marker PDF preprocessing
```

### 2. Download a corpus

Download O-RAN specifications from [o-ran.org](https://www.o-ran.org/specifications) into `data/oran/`.

### 3. Ingest

```bash
open-telco-rag ingest oran --model minilm --data data/oran/
```

Available embedding models:

| Key | Model | Dimensions | Notes |
|-----|-------|------------|-------|
| `minilm` | all-MiniLM-L6-v2 | 384 | Fast — recommended for development |
| `mpnet` | all-mpnet-base-v2 | 768 | Balanced quality and speed |
| `e5` | intfloat/e5-large-v2 | 1024 | High quality — recommended for production |
| `bge` | BAAI/bge-large-en-v1.5 | 1024 | Strong on technical domains |

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

The Open Telco RAG cookbook is available as a [paper](docs/paper/cookbook.pdf) and as interactive Jupyter notebooks in the [`notebooks/`](notebooks/) directory — one per corpus:

| Notebook | Corpus |
|----------|--------|
| [`01_oran_quickstart.ipynb`](notebooks/01_oran_quickstart.ipynb) | O-RAN Alliance R005 |
| [`02_3gpp_quickstart.ipynb`](notebooks/02_3gpp_quickstart.ipynb) | 3GPP Release 18 |
| [`03_camara_quickstart.ipynb`](notebooks/03_camara_quickstart.ipynb) | CAMARA Fall25 |
| [`04_tmforum_quickstart.ipynb`](notebooks/04_tmforum_quickstart.ipynb) | TM Forum Open APIs |
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

```
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

### Ask options

```bash
open-telco-rag ask "What does TS 28.312 define for intent?" \
  --source-type 3gpp \
  --model bge \
  --top-k 8 \
  --rerank \
  --rerank-top-k 5
```

### Evaluate options

```bash
open-telco-rag evaluate 3gpp \
  --model bge \
  --db-path chroma_db \
  --collection 3gpp_r19_bge \
  --questions-file configs/eval_questions/3gpp.json \
  --rag-generator openrouter \
  --rag-generator-model meta-llama/llama-3.3-70b-instruct \
  --generator openrouter \
  --generator-model anthropic/claude-sonnet-4.5 \
  --output results/
```

---

## Python API

```python
from open_telco_rag import ingest, search, generate
from open_telco_rag.sources.oran import ORANSource
from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.generators.openrouter import OpenRouterGenerator

# Ingest
source   = ORANSource()
embedder = SentenceTransformerEmbedder(model_key="bge")
store    = ChromaStore(path="chroma_db", collection_name="oran_bge")

ingest(source=source, store=store, embedder=embedder, data_dir="data/oran/")

# Search
results = search("E2 interface service models", store=store, embedder=embedder, top_k=5)

# Generate
generator = OpenRouterGenerator(model="meta-llama/llama-3.3-70b-instruct")
answer    = generator.generate("What does WG3 define for the E2 interface?", results)
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
        │
        ▼
   ChromaDB              persistent vector store
        │
        ▼
   Search                cosine similarity retrieval + optional reranking
        │
        ▼
   Generator             Claude or OpenRouter LLM
        │
        ▼
   Answer + sources
```

---

## Project Structure

```
open-telco-rag/
├── open_telco_rag/
│   ├── types.py              # shared protocols and dataclasses
│   ├── core/
│   │   ├── pipeline.py       # ingest, search, generate
│   │   ├── chunking.py       # chunk_text, chunk_markdown
│   │   ├── cleaning.py       # base_clean and helpers
│   │   └── marker_loader.py  # Datalab Marker API mixin
│   ├── sources/              # one class per SDO
│   │   ├── oran.py
│   │   ├── tgpp.py           # 3GPP
│   │   ├── etsi.py
│   │   ├── itu.py
│   │   ├── tmforum.py
│   │   ├── camara.py
│   │   └── gsma.py
│   ├── embedders/
│   │   ├── sentense_transformers.py
│   │   └── reranker.py
│   ├── generators/
│   │   ├── claude.py
│   │   └── openrouter.py
│   ├── evaluation/           # evaluation framework
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
│   └── main.py               # CLI entry point
├── notebooks/                # quickstart cookbook
│   ├── 01_oran_quickstart.ipynb
│   ├── 02_3gpp_quickstart.ipynb
│   ├── 03_camara_quickstart.ipynb
│   ├── 04_tmforum_quickstart.ipynb
│   ├── 05_gsma_quickstart.ipynb
│   ├── 06_itu_quickstart.ipynb
│   └── 07_etsi_quickstart.ipynb
├── configs/
│   ├── embedding_models.json
│   ├── eval_questions/       # evaluation question sets
│   └── reranker_models.json
└── pyproject.toml
```

---

## Extending the Pipeline

### Adding a new source

Implement the `Source` protocol from `open_telco_rag.types`:

```python
from open_telco_rag.types import Source
from pathlib import Path

class MySource:
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, path: Path) -> str:
        ...

    def clean(self, text: str) -> str:
        ...

    def metadata(self, path: Path) -> dict:
        return {"source": path.name, "source_type": "my_source"}
```

### Using a different LLM

```python
from open_telco_rag.generators.openrouter import OpenRouterGenerator

generator = OpenRouterGenerator(model="meta-llama/llama-3.3-70b-instruct")
answer = generator.generate(question, results)
```

---

## Contributing

Contributions welcome. Please open an issue before submitting a pull request for significant changes.

---

## Acknowledgements

This work is part of the GSMA Open Telco AI Initiative. For more information visit [gsma.com/open-telco-ai](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/).

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
