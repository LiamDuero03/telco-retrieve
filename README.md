# TelcoRetrieve

## One retrieval layer for telecom knowledge every AI model can reason over

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GSMA Open Telco AI](https://img.shields.io/badge/GSMA-Open%20Telco%20AI-red.svg)](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-GSMA-orange.svg)](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks)

An open, vendor-neutral RAG pipeline for telecom specifications. It ingests documents, searches them, reranks the hits and answers with sources.

Part of the [GSMA Open Telco AI](https://www.open-telco.ai/).

> **TelcoRetrieve Challenge: open 13 Nov 2026 – 13 Jan 2027.** Optimise this pipeline on 3GPP Rel-19 and compete for €25,000 in prizes. [Challenge guide](docs/challenge/README.md) · [Join on Zindi](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge) · [Challenge slides](https://github.com/Znbne/telco-retrieve/blob/main/TelcoRetrieve%20Challenge%20Slides.pdf)

<p align="center">
  <img src="docs/graphics/telcoretrieve-pipeline.png" alt="TelcoRetrieve pipeline: documents are chunked and embedded into an index, then retrieved, reranked and passed to the generator to produce an answer with sources" width="700">
</p>

---

## Overview

TelcoRetrieve answers questions about telecom standards from the standards themselves. RAG (retrieval-augmented generation) means the tool first finds the relevant passages, then a language model writes the answer from them and cites them.

```bash
open-telco-rag ask "What does O-RAN WG3 define for the E2 interface?" --source-type oran
open-telco-rag ask "What does 3GPP TS 28.312 define for intent management?" --source-type 3gpp
open-telco-rag ask "What are the ETSI ZSM closed-loop automation requirements?" --source-type etsi
```

The pipeline has five steps:

1. **Ingest:** read documents and split them into chunks (short passages).
2. **Embed:** turn each chunk into a vector with an embedding model, and store it in an index (a searchable store).
3. **Retrieve:** find the chunks closest to the question, by vectors, keywords or both (hybrid).
4. **Rerank:** reorder the hits with a reranker (a model that scores question and chunk together). Optional.
5. **Generate:** a language model answers from the best chunks and lists its sources.

Details: [Architecture](docs/architecture.md).

---

## Supported Corpora

A corpus is one body of documents. Each has its own source loader.

| Corpus | Source | Format |
|--------|--------|--------|
| O-RAN Alliance | O-RAN R003–R005, all WGs | PDF, DOCX |
| 3GPP | TSpec-LLM dataset, Rel-8 through Rel-19 | Markdown |
| ETSI | ENI, ZSM, MEC, NFV, SAI, F5G, RIS | PDF, YAML |
| ITU-T | All 27 active series (G, Y, Q, X, M…) | PDF |
| CAMARA | Fall25, 75+ API repos | Markdown, YAML |
| GSMA | OPG, IG, AID, SGP, RCS and more | PDF, DOCX |
| TM Forum | TMF Open APIs, ODA Component specs ([tmforum-rand/tmf-otel](https://github.com/tmforum-rand/tmf-otel)) | YAML, JSON |

---

## Pre-built Dataset

The indexed corpus (4.6M+ chunks, 1B+ tokens, 7 standards bodies) is on Hugging Face as [GSMA/telco-retrieve-chunks](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks). Use it to skip ingestion.

```python
from datasets import load_dataset

ds = load_dataset("GSMA/telco-retrieve-chunks", data_files="etsi.jsonl")   # one corpus
ds = load_dataset("GSMA/telco-retrieve-chunks")                            # all corpora
```

---

## Installation

This installs the package and the local embedding models. The package is installed from a clone of this repository.

```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed,pdf,docx]"
```

API keys go in a `.env` file (see Quick Start, step 1). The extras are `local-embed` (local models), `pdf` and `docx` (document readers).

### NVIDIA, AMD ROCm or CPU: install PyTorch first

The `local-embed` extra depends on `torch`. If you install the package first, pip pulls the default PyTorch build from PyPI, which on Linux is a CUDA build of several GB. Install the build for your hardware first, then the package, then check:

```bash
pip install --upgrade pip

# 1. PyTorch first, from the index that matches your hardware (see https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu128      # NVIDIA, CUDA 12.8 (cu126, cu130, ... to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2    # AMD, ROCm 7.2
pip install torch --index-url https://download.pytorch.org/whl/cpu        # no GPU

# 2. then the package
pip install -e ".[local-embed,pdf,docx]"

# 3. check which build you have
python -c "import torch; print('gpu:', torch.cuda.is_available(), '| cuda:', torch.version.cuda, '| hip:', torch.version.hip)"
```

Run only the line of step 1 that matches your machine. NVIDIA prints `gpu: True` with a `cuda` version. AMD ROCm prints `gpu: True` with a `hip` version. A CPU build prints `gpu: False`.

Tip: for ROCm, use the index that matches the ROCm version of the host (`rocm-smi` shows it).

Details: [Installation](docs/reference/installation.md).

---

## Quick Start

Five steps: set keys, get documents, ingest, search, ask.

### 1. Configure API keys

The `ask` command needs a key for the language model. Create a `.env` file in your project root:

```bash
OPENROUTER_API_KEY=your_key_here      # for LLM generation via OpenRouter (Llama, Claude, GPT-4o, etc.)
ANTHROPIC_API_KEY=your_key_here       # optional — for direct Claude access
DATALAB_API_KEY=your_key_here         # optional — for Marker PDF preprocessing
```

Tip: `.env.example` lists every variable, including the ones for a local model server.

### 2. Download a corpus

You supply the documents. Download O-RAN specifications from [o-ran.org](https://www.o-ran.org/specifications) into `data/oran/`.

**3GPP:** the Markdown conversions are in the TSpec-LLM dataset on Hugging Face (`rasoul-nikbakht/TSpec-LLM`). Read its terms first. The dataset is gated, so log in with `huggingface-cli login` and accept them. It holds a `.md` and a `.docx` for every spec; download only the Markdown:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('rasoul-nikbakht/TSpec-LLM', repo_type='dataset', allow_patterns=['3GPP-clean/*.md'], local_dir='data/3gpp')"
open-telco-rag ingest 3gpp --data data/3gpp/3GPP-clean --strategy hierarchical_markdown
```

Details: [Corpora](docs/reference/corpora.md).

### Preparing the GSMA/3GPP Rel-19 data

This step stages the Rel-19 specifications from the [`GSMA/3GPP`](https://huggingface.co/datasets/GSMA/3GPP) dataset under the names the 3GPP loader reads: `<out>/Rel-19/<NN>_series/<spec>-<version>.md`. It is opt-in and changes no default. The files are downloaded onto your machine; read the dataset's licence and the 3GPP terms first.

Install the package first (`pip install -e ".[local-embed,pdf,docx]"` from the repository root). The script imports `open_telco_rag` and stops with `ModuleNotFoundError` without it.

```bash
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged --dry-run   # plan only
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged --verify    # offline re-check
open-telco-rag ingest 3gpp --data data/3gpp/staged --model bge-m3 \
  --strategy hierarchical_markdown --chunk-size 2048 --enrichment none --distance-metric cosine \
  --bound-tokens 512 \
  --bound-verify minilm=256,bge=512,bge-m3=512,qwen3-embedding-0.6b=512,embeddinggemma-300m=512 \
  --bound-outliers split
```

**Why the bound.** Without `--bound-tokens`, chunks of the staged set reach about 89,000 tokens and 46.6% exceed minilm's 256-token limit. The options above cap every chunk at 512 tokens and cut further any chunk that is over another embedder's limit. For each embedder, change only `--model`. The `--bound-*` options, with the same `--bound-verify` list, must be identical for every embedder, so that all collections hold identical chunks.

- `--revision` is a 40-character commit hash, so everyone stages the same files.
- Files under `--stub-bytes` (default 5,120 bytes) are flagged `is_stub` in the manifest and still staged: 8 specs are under 5,000 bytes and 11 under 5,120.
- `--dry-run` takes about 6 to 7 minutes because it lists the whole tree. The real run downloads about 0.73 GB, about 1.5 GB with staging.
- The dataset is public. `HF_TOKEN` is used if set and never printed. Rate-limit (429) answers are retried and counted in `staging_run.json`.
- The `manifest.json` is byte-identical for the same revision, whether the files were downloaded or staged with `--source DIR` from a tree you already have.

Tip: if you do not use minilm, remove it from the `--bound-verify` list for every embedder.

Details: [Prepare the Rel-19 data](docs/reference/3gpp-data.md).

### 3. Ingest

Ingest reads the files, splits them into chunks, embeds them and stores them in an index (a ChromaDB folder, `chroma_db` by default).

```bash
open-telco-rag ingest oran --model bge --data data/oran/
```

Pick the embedding model with `--model`:

| Key | Model | Dimensions | Notes |
|-----|-------|------------|-------|
| `minilm` | all-MiniLM-L6-v2 | 384 | Fast, good for development |
| `mpnet` | all-mpnet-base-v2 | 768 | Balanced quality and speed |
| `e5` | intfloat/e5-large-v2 | 1024 | High quality |
| `bge` | BAAI/bge-large-en-v1.5 | 1024 | Strong on technical domains |
| `otel-109m` | OTel-Embedding-109M | 768 | Telecom fine-tuned, 109M parameters |
| `otel-0.6b` | OTel-Embedding-0.6B | 1024 | Telecom fine-tuned, 0.6B parameters |

Tip: use the same `--model` when you search and ask. Details: [CLI reference](docs/reference/cli.md), [embedding models](docs/reference/embedding.md).

### 4. Search

Search prints the chunks closest to your query.

```bash
open-telco-rag search "near-RT RIC control loop" --source-type oran --top-k 5
```

### 5. Ask a question

Ask searches, then the language model answers from the chunks and lists its sources.

```bash
open-telco-rag ask "What interfaces does the Near-RT RIC expose?" --source-type oran
```

---

## Configuration

Every option is a command-line flag with a default. These are the ones you will use most:

| Flag | Default | Used by | Meaning |
|------|---------|---------|---------|
| `--model` | `minilm` | ingest, search, ask | Embedding model key |
| `--db-path` | `chroma_db` | all | Where the index is stored |
| `--strategy` | `text_baseline` | ingest | How documents are split into chunks |
| `--chunk-size` | `1024` | ingest | Chunk size in characters |
| `--distance-metric` | `l2` | ingest | `l2`, `cosine` or `ip` |
| `--search-method` | `dense` | search, ask | `dense`, `bm25` or `hybrid` |
| `--top-k` | 5 / 8 | search, ask | Chunks to retrieve |
| `--rerank` | off | search, ask | Rerank the hits |
| `--generator` | `openrouter` | ask | `claude`, `openrouter` or `local` |

Models, chunking and enrichment choices come from the JSON registries in `configs/`. Details: [CLI reference](docs/reference/cli.md), [chunking](docs/reference/chunking.md).

---

## Commands

Run `open-telco-rag <command> --help` for every flag.

| Command | What it does |
|---------|--------------|
| `ingest <source>` | Read, chunk, embed and store documents |
| `search <query>` | Print the closest chunks |
| `ask <question>` | Search, then answer with sources |
| `evaluate <corpus>` | Run the evaluation suite |
| `info` | Show what a database holds |
| `delete <collection>` | Delete a collection |
| `push-hf`, `pull-hf` | Push or pull a database folder on the Hugging Face Hub |

Details: [CLI reference](docs/reference/cli.md), [Evaluation](docs/reference/evaluation.md).

---

## TelcoRetrieve Cookbook

The notebooks in [`notebooks/`](notebooks/) are the TelcoRetrieve Cookbook, one per corpus. Each covers ingestion, search, question answering and evaluation. The PDF [Telco-Retrieve-Guidebook.pdf](Telco-Retrieve-Guidebook.pdf) is the Guidebook.

| Notebook | Corpus |
|----------|--------|
| [`01_oran_quickstart.ipynb`](notebooks/01_oran_quickstart.ipynb) | O-RAN Alliance R005 |
| [`02_3gpp_quickstart.ipynb`](notebooks/02_3gpp_quickstart.ipynb) | 3GPP Release 8 through 19 |
| [`03_camara_quickstart.ipynb`](notebooks/03_camara_quickstart.ipynb) | CAMARA Fall25 |
| [`04_tmforum_quickstart.ipynb`](notebooks/04_tmforum_quickstart.ipynb) | TM Forum Open APIs & ODA Components |
| [`05_gsma_quickstart.ipynb`](notebooks/05_gsma_quickstart.ipynb) | GSMA Open Gateway |
| [`06_itu_quickstart.ipynb`](notebooks/06_itu_quickstart.ipynb) | ITU-T |
| [`07_etsi_quickstart.ipynb`](notebooks/07_etsi_quickstart.ipynb) | ETSI ISG groups |

---

## Learn more

- [Documentation index and glossary](docs/README.md)
- [CLI reference](docs/reference/cli.md), [chunking](docs/reference/chunking.md), [embedding models](docs/reference/embedding.md), [evaluation](docs/reference/evaluation.md), [Python API](docs/reference/python-api.md)
- [Architecture](docs/architecture.md) and [project structure](docs/architecture.md#project-structure)
- [Contributing](docs/contributing.md)
- [TelcoRetrieve Challenge](docs/challenge/README.md). The challenge question set is on [Zindi](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge).
- Q&A generation: under development

---

## License

Apache 2.0. See [LICENSE](LICENSE).
