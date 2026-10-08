# Telco-Retrieve

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GSMA Open Telco AI](https://img.shields.io/badge/GSMA-Open%20Telco%20AI-red.svg)](https://www.gsma.com/solutions-and-impact/technologies/networks/gsma_resources/open-telco-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-GSMA-orange.svg)](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks)

An open, vendor-neutral RAG pipeline for telecom specifications, covering 7 standards bodies, 4.6M+ chunks and 1B+ tokens across O-RAN, 3GPP, ETSI, ITU-T, TM Forum, CAMARA, and GSMA.

Part of the [GSMA Open Telco AI](https://www.open-telco.ai/).

---

## Overview

Telco-Retrieve enables engineers and researchers to ask natural language questions directly against authoritative telecom standards, with source citations, no hallucination, and full corpus coverage across all major SDOs.

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

🤗 **[GSMA/telco-retrieve-chunks](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks)**

Download individual corpora or the full dataset without running ingestion yourself:

```python
from datasets import load_dataset

# Load one corpus
ds = load_dataset("GSMA/telco-retrieve-chunks", data_files="etsi.jsonl")

# Load all corpora
ds = load_dataset("GSMA/telco-retrieve-chunks")
```

---

## Installation

From a clone of this repository (the package is not published on PyPI):

```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed,pdf,docx]"
```

If a release is published, `pip install open-telco-rag[local-embed,pdf,docx]` installs the same extras.

### NVIDIA, AMD ROCm or CPU: install PyTorch first

The `local-embed` extra depends on `torch`. If you install the package first, pip pulls the default PyTorch build from
PyPI, which on Linux is a CUDA build of several GB. That is the wrong build on an AMD GPU and a waste on a machine
without a GPU, and replacing it afterwards means a forced reinstall. So install PyTorch **first**, from the index that
matches your hardware, then the package, then check:

```bash
# 1. PyTorch first. The index URL names the build; take the exact name for your CUDA or ROCm version from
#    https://pytorch.org/get-started/locally/ (the version numbers below are examples)
pip install torch --index-url https://download.pytorch.org/whl/cu128      # NVIDIA, CUDA 12.8 (cu126, cu130, ... to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4    # AMD, ROCm 6.4 (to match your ROCm)
pip install torch --index-url https://download.pytorch.org/whl/cpu        # no GPU

# 2. then the package: it finds torch already installed and leaves it alone
pip install -e ".[local-embed,pdf,docx]"

# 3. check which build you have
python -c "import torch; print('gpu:', torch.cuda.is_available(), '| cuda:', torch.version.cuda, '| hip:', torch.version.hip)"
```

Run only the line of step 1 that matches your machine. NVIDIA prints `gpu: True | cuda: 12.x | hip: None`. An AMD ROCm
build also answers `torch.cuda.is_available()`, so it prints `gpu: True | cuda: None | hip: 6.x`. A CPU build prints
`gpu: False | cuda: None | hip: None`. If you have a GPU and see `gpu: False`, or the wrong one of `cuda` and `hip`, the wrong
build is installed: run the matching line of step 1 again with `--force-reinstall`.

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

**3GPP:** the CLI does not download specs; you supply the files. The Markdown conversions of the 3GPP
specifications are in the TSpec-LLM dataset on Hugging Face (`rasoul-nikbakht/TSpec-LLM`; the id was checked
on 2026-09-30, but verify it and read its terms before use: the dataset is gated, so log in with
`huggingface-cli login` and accept them first). It holds a `.md` and a `.docx` for every spec, and the 3GPP
source ingests **both** extensions, so download only the Markdown:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('rasoul-nikbakht/TSpec-LLM', repo_type='dataset', allow_patterns=['3GPP-clean/*.md'], local_dir='data/3gpp')"
open-telco-rag ingest 3gpp --data data/3gpp/3GPP-clean --strategy hierarchical_markdown
```

The layout is `3GPP-clean/Rel-<N>/<NN>_series/<spec>.md`; the release, series, spec number and version come from
those folder and file names (`28312-i11.md` is TS 28.312, Rel-18). Files whose names do not follow that pattern
still ingest but may lack `spec_number`. All releases are millions of chunks: see "Resume" below and ingest one
release at a time if you want the smallest possible restart unit.

### Preparing the GSMA/3GPP Rel-19 data

`scripts/stage_3gpp.py` downloads the Markdown conversions of the Rel-19 specifications from the
[`GSMA/3GPP`](https://huggingface.co/datasets/GSMA/3GPP) dataset, checks them, and stages them under names that the 3GPP
loader reads: `<out>/Rel-19/<NN>_series/<spec>-<version>.md`. It downloads only `marked/Rel-19/**/raw.md` and
`conversion_log.jsonl`, never the ~32,760 figure images. It is opt-in: the loader and the default behaviour are unchanged.
The files are downloaded from the original source onto your machine; this repository does not contain or redistribute
them. Read the dataset's licence and the 3GPP terms before you use the data.

Install the package first (`pip install -e ".[local-embed,pdf,docx]"` from the repository root, see "Installation"): the
script imports `open_telco_rag`, and without the install it stops with `ModuleNotFoundError`.

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

**Why the bound.** Without `--bound-tokens`, chunks of the staged set reach about 89,000 tokens and 46.6% of them exceed
minilm's 256-token limit (measured on the Rel-19 set in a GPU test run; I have not re-measured them since). The options
above cut every chunk to at most 512 tokens and, with `--bound-verify ... --bound-outliers split`, cut further any chunk
that is still over another embedder's limit. Repeat the command for each embedder and change only `--model`: the
`--bound-*` options, with the same `--bound-verify` list, must be identical for every embedder, so that all collections
hold identical chunks and their results can be compared.

- `--revision` must be a 40-character commit hash (a branch name is refused), so that everyone stages the same files.
  The one above is the revision the counts below were measured on.
- Expected for that revision: 1,566 conversion-log rows (1,557 OK, 9 ERROR), 1,557 `raw.md` files and 1,557 staged files;
  67 multi-part specs; 20 specs with a `_cover` file name; 49 `.doc` log rows; 7 specs whose first conversion failed and
  a later one worked; one spec with only failed conversions and no file (TS 25.331).
- Stubs: files smaller than `--stub-bytes` (default 5,120 bytes) are flagged `is_stub` in the manifest and still staged.
  8 specs are under 5,000 bytes (38.460, 38.462, 38.461, 25.461, 37.481, 22.242, 25.431, 38.421) and 11 under 5,120 bytes
  (the same eight plus 38.471, 41.101 and 25.421). They hold little more than a cover and a contents list.
- Time and disk: `--dry-run` takes about 6 to 7 minutes, because it lists the whole tree (about 34,000 files, in pages of
  50) to read the size and hash of each spec file; it downloads no spec file. The real run then downloads about 0.73 GB in
  parallel (`--workers`) into `<out>_raw`, next to `--out`, not inside it, so that `ingest --data <out>` never reads it, and
  stages the same again: about 1.5 GB in total. A run that stops can be started again.
- The dataset is public. If you set `HF_TOKEN` in the environment it is used and never printed. A token does not
  prevent HTTP 429 (rate limit) answers: the script, and the Hugging Face library under it, retry with backoff, and the
  retries are counted in `staging_run.json`.
- For each spec the last conversion-log row with status OK decides the version (`38300-j20.docx` is V19.2.0). The
  version is compared with a `V<major>.<minor>.<patch>` string in the first 40 lines of the file; a mismatch is
  counted in the manifest, never an error.
- `manifest.json` records for every file its source path, staged name, size, sha256, version and part, and lists the excluded
  specs. It has no timestamps, no local paths and no run-dependent numbers: for the same revision it is byte-identical
  whether the files were downloaded or staged with `--source` from an already downloaded tree (with the same `raw.md` files
  and log). What differs from run to run (finish time, retry counts, the number of figure files in the listing) is in
  `staging_run.json`.
- `--source DIR` stages from an already downloaded tree with the dataset's layout, offline.

#### Known limitations

- **minilm's 256-token limit sets the chunk size for everyone.** With the verify list above, every collection gets the same
  chunks, cut so that none is over any listed embedder's limit; minilm's 256 is the strictest, so the other models also get
  chunks that are shorter than the 512 tokens they could take. If you do not need minilm, take it out of the list, for
  every embedder.
- **Multi-part specs have no version in the chunk metadata.** The loader allows one suffix after the five digits, so a spec
  with a part (for example `32111-1`) is staged as `32111-1.md`: its chunk metadata has `doc_part` but no `version`; the
  version is only in the manifest. For study reports (`23700-75`) the "part" is a study-item number, marked
  `doc_part_kind: study_item_number` in the manifest (a rule of thumb: spec numbers x.700 to x.899).
- **The cleaning leaves noise.** The cover page and contents list are kept in about 170 specs (the cover pattern is
  deliberately limited to the start of a document), and table separator rows (long runs of `-`) and dot leaders in contents
  tables are not collapsed. They cost tokens, and are why a few chunks are mostly formatting.
- **float16 on a GPU, and EmbeddingGemma.** Embedders load in float16 on a GPU. `embeddinggemma-300m` returned NaN vectors in a
  GPU test. Until the per-key `dtype` option and the `embeddinggemma-300m-fp32` key are merged, do not rely on it on a GPU. On
  the current main a NaN batch is rejected by Chroma while the BM25 index keeps its rows, and ingest still exits with code 0.
- **Paths.** The loader reads the release and the series from any `Rel-NN` or `NN_series` folder in the whole path, so do not
  put `--out` below such a folder (the script warns).
- Only Rel-19 has been tested.

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

The Open Telco RAG Cookbook is available as a [PDF](https://github.com/Znbne/telco-retrieve/blob/main/Telco-Retrieve-Guidebook.pdf) and as interactive Jupyter notebooks in the [`notebooks/`](notebooks/) directory — one per corpus:

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
open-telco-rag push-hf <db_path> <repo_id>            Push a local Chroma+FTS database folder to HF Hub
open-telco-rag pull-hf <repo_id> <path_in_repo>       Pull a database folder from HF Hub for evaluation
```

`pull_collection_from_hf()` (`open_telco_rag/stores/hf.py`, used by both `pull-hf` and the evaluation grid scripts below) correctly handles a multi-segment `path_in_repo` (e.g. `final-ingestion-grid/{run_name}`), not just a bare `run_name` — a real bug previously made a multi-segment path land the downloaded files one directory level deeper than the function claimed to return, so `ChromaStore` silently opened an empty directory and reported `0` chunks even though the download itself had genuinely succeeded.

### Ingest options

```bash
open-telco-rag ingest oran \
  --data data/oran/ \
  --model bge \
  --strategy sliding_window_tokens \
  --chunk-size 512 \
  --overlap 50 \
  --enrichment acronym_expansion \
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
| `--target-tokens` | off | Optional: size chunks in tokens instead of characters (see [Token-based chunk sizing](#optional-token-based-chunk-sizing)) |
| `--max-tokens` | off | Optional: hard cap on tokens per chunk (defaults to `--target-tokens`) |
| `--overlap-tokens` | off | Optional: token overlap between chunks (`text_baseline` and `sliding_window_tokens` only) |
| `--enrichment` | `none` | Pre-embedding strategy registry token mapped from enrichment_strategies.json |
| `--enrichment-model` | `google/qwen-2.5-coder-32b` | Model name to request for the `llm_metadata` enrichment strategy — sent to whichever server `--llm-backend`/`--llm-base-url` point at |
| `--distance-metric` | `l2` | Vector space index geometry calculation mapping: l2, cosine or ip |
| `--no-resume` | off | Disable resume — re-ingest all files |
| `--use-marker` | off | Use Datalab Marker API for PDF preprocessing |
| `--collection` | `source_model` | Override collection name |
| `--embed-backend` | `local` | Backend for computing vector embeddings: `local` (CPU/GPU) or `openrouter` (Cloud APIs) |
| `--batch-size` | 256 / 500 | Embedding batch size: 256 locally, 500 with `--embed-backend openrouter` (the `ingest()` Python function itself defaults to 64) |
| `--embed-max-workers` | `10` | OpenRouter embeddings only: in-flight requests per `embed()` call |

`--overlap` has no effect on `hierarchical_markdown`, which splits on headings and `--chunk-size` only.

**Resume**: by default an interrupted ingest continues where it stopped. Progress is saved during the run, as
soon as every chunk of a file is confirmed stored in both ChromaDB and the FTS5 index, so a run that is killed
resumes with the files that were not fully stored (a file with any unwritten chunk is not recorded). The
progress file in `logs/` belongs to one configuration (database path, collection,
embedding model, chunking and enrichment settings; secrets are never recorded),
so ingesting into a different collection or with different settings starts from
scratch instead of skipping files. An unreadable or foreign progress file is
renamed to `*.bak` with a warning, and progress files from older versions
(which did not record their configuration) are set aside the same way, so an
ingest interrupted before upgrading restarts from scratch. Re-ingesting into a
collection that already holds chunks from other settings warns that they can end
up mixed; use `--no-resume` to rebuild it or a new `--collection`.

**Failed files:** a file that cannot be parsed (a corrupt `.docx`, an empty file) is counted under `Errors:` or
`Skipped:`, logged in `logs/ingest_*.csv`, and retried by the next run; the rest of the run continues. `ingest`
exits with status 0 even when some files failed, and it does not check that a parsed file's text is sensible
(a binary file named `.md` is ingested as text), so check the `Errors:` line or the CSV, and the chunk count with
`open-telco-rag info`.

**Chunk ids** are `<path relative to --data>::<model key>::<chunk index>`, with the path written using the
operating system's separator: a collection built on Windows has backslashes in its ids. The evaluation metrics
compare ids independently of the separator; the stored ids are not rewritten.

Ingest also builds a per-collection SQLite FTS5 BM25 index alongside ChromaDB, enabling `--search-method bm25` and `--search-method hybrid` retrieval.

**Write reliability**: `ChromaStore` queries its Chroma installation's real max batch size live at construction (`ChromaStore.max_batch_size`, from `chromadb`'s own `get_max_batch_size()` — not a hardcoded constant, since this depends on the local SQLite build). Every internal write is capped at that value regardless of `--batch-size`, so a large `--batch-size` on a bulk/high-throughput run can no longer produce a Chroma `"Batch size of N is greater than max batch size of M"` crash partway through — previously possible on large runs once accumulated writes exceeded Chroma's real limit, sometimes only surfacing after the progress bar had already reported 100%. Reported `chunks_stored` counts also only ever reflect confirmed-successful writes, not chunks merely attempted, so a partial failure produces an accurate partial count instead of continuing to count chunks that were never actually persisted.

A failure in purely cosmetic bookkeeping *after* every chunk is already safely stored (writing the resume-progress JSON file, a CSV log row, or a run's `_ingest_complete.json` completion marker) is now isolated from the write-critical path — it's logged as a `[WARN]`, but it can no longer retroactively misreport a fully successful ingest as `[ERROR] Bulk indexing failed` with an inflated `error_files` count. This applies to both `ingest()` and `ingest_multi_enrichment()` — the latter isolates it per enrichment variant, so one variant's marker-write hiccup doesn't take down the reported status of every other variant in the same run.

**Config resolution**: `--strategy`/`--enrichment`'s available choices, and whether a given strategy/enrichment needs an LLM client, are read from `configs/chunking_strategies.json`/`configs/enrichment_strategies.json` relative to the installed package location as a fallback if no local `configs/` directory is found relative to the current working directory — so these are no longer silently degraded (to `text_baseline`-only / `none`-only, with LLM-requiring strategies going undetected) when the CLI isn't run from the exact repository root.

### Optional: token-based chunk sizing

By default every strategy sizes chunks in **characters** (`--chunk-size`,
`--overlap`; `parent_size`/`child_size` for `parent_child`;
`chunk_tokens`/`overlap_tokens` for `sliding_window_tokens`), exactly as
before. Token sizing is opt-in and works for all five strategies:

```bash
open-telco-rag ingest 3gpp --model bge --strategy hierarchical_markdown \
  --target-tokens 256 --max-tokens 320
```

- `--target-tokens` is the size to aim for; `--max-tokens` is a hard cap
  (defaults to `--target-tokens`). Either one switches the run to token
  sizing. Tokens are counted with `cl100k_base`.
- Every chunk is an exact slice of the cleaned source text, and the ingest
  metadata of token-sized chunks additionally stores `char_start`, `char_end`
  and `n_tokens` (used by the span metrics below). Default ingests store
  exactly the metadata they always did.
- Combining token sizing with a character size (`--chunk-size`, `--overlap`,
  `parent_size`, `child_size`, or `chunk_tokens` for `sliding_window_tokens`)
  is an error rather than a silent choice. `max_tokens` must be at least
  `target_tokens`, and `overlap_tokens` must be smaller than the chunk size.
- `parent_child`: `target_tokens` sizes the child chunks (parents default to
  4x), so `max_tokens` bounds the retrieved chunk.
- The tokenizer is only initialized when token sizing is used.

```python
from open_telco_rag.core.chunking import chunk_document
chunks = chunk_document(text, "text_baseline", target_tokens=200, max_tokens=256)
```

### Optional: hard token bound

Embedding models cut their input at a fixed length (512 tokens for many of them), and the text past it
is silently ignored. `--bound-tokens N` guarantees that no stored chunk is longer than N tokens, for
every strategy and with character or token sizing; without it nothing changes.

```bash
open-telco-rag ingest 3gpp --model bge-m3 --strategy hierarchical_markdown \
  --target-tokens 400 --bound-tokens 512 --bound-tokenizer bge-m3
```

- A chunk over the bound is split after chunking at the coarsest boundary that works: paragraphs, then
  table and text blocks, then sentences, then lines, and last a cut by token count. A markdown table is
  split by rows and its header rows are repeated at the top of every part (those parts are not exact
  slices of the source). The heading and the metadata stay on every piece; chunks within the bound are
  untouched.
- **The bound is counted in one tokenizer, `tiktoken:cl100k_base` by default, and only guarantees that
  chunks fit in those tokens.** Other embedders count differently (on technical text 4-20% more than
  cl100k), so set the ceiling with headroom for the strictest embedder you use, and verify it with
  `scripts/check_chunk_bound.py`, which reports chunks over each model's own limit and can calibrate the
  ceiling from your data.
- `--bound-tokenizer` changes the counting tokenizer: `tiktoken:<encoding>`, an embedding model key from
  `configs/embedding_models.json`, a Hugging Face model name or a `tokenizer.json` path. Only tokenizer
  files are downloaded, never model weights.
- `--bound-margin K` reserves K tokens for special tokens (default 2: a 512 limit cuts at 510).
- The bound, the tokenizer name and the margin (with their defaults filled in) are part of the resume
  record and of the settings sidecar: changing any of them starts a new ingest instead of skipping
  finished files.
- `--bound-verify NAME=LIMIT,...` (for example `bge-m3=512,minilm=256`) names other embedders to check
  the chunks against after the cl100k pass. A chunk still over one of those limits is only logged
  (`--bound-outliers warn`, the default) or cut further (`--bound-outliers split`). Nothing happens
  without `--bound-verify`.
- Enrichment (`--enrichment`) adds text after chunking, and `--resolve-parents` returns the parent
  text; neither is bounded. Leave room with `--bound-margin` when enrichment is on.
- Check a collection or a staged directory without changing anything:
  `python scripts/check_chunk_bound.py --data-dir DIR --source-type 3gpp --tokenizers minilm bge bge-m3 --calibrate`
  reports, per tokenizer, chunks over its limit, how its token counts compare with cl100k (p50, p99,
  p99.9, max), the largest chunks and what makes them long, the text retained by the bound pass and by
  the `min_length` filter, and with `--calibrate` the largest cl100k ceiling that leaves no chunk (or at
  most 0.1%) over each limit, so the ceiling is chosen from your data.

```python
from open_telco_rag.core.chunking import chunk_document
chunks = chunk_document(text, "text_baseline", chunk_size=2000, bound_tokens=512, bound_tokenizer="bge-m3")
```

### Embedding models and prefixes

`--model` takes any key from `configs/embedding_models.json`. Besides the
original keys (`minilm`, `mpnet`, `e5`, `bge`, `otel-*`, `or-*`) the registry
includes:

| Key | Model | Notes |
|-----|-------|-------|
| `bge-m3` | BAAI/bge-m3 | local; no prefix needed |
| `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | local; instruction-style query prefix |
| `embeddinggemma-300m` | google/embeddinggemma-300m | local; uses the model's named prompts (`Retrieval-query` / `Retrieval-document`) |
| `qwen3-embedding-8b` | qwen/qwen3-embedding-8b | served through OpenRouter (`--embed-backend openrouter`, needs `OPENROUTER_API_KEY`), 4096 dims, instruction-style query prefix |
| `e5-prefixed` | intfloat/e5-large-v2 | opt-in variant of `e5` with `query: ` / `passage: ` prefixes |
| `bge-prefixed` | BAAI/bge-large-en-v1.5 | opt-in variant of `bge` with BGE's query instruction |

```bash
export OPENROUTER_API_KEY=...            # your key; never commit it
open-telco-rag ingest 3gpp --model qwen3-embedding-8b --embed-backend openrouter
open-telco-rag evaluate 3gpp --model qwen3-embedding-8b --embed-backend openrouter
```

**Known issue: `otel-300m`.** Loading `farbodtavakkoli/OTel-Embedding-300M` fails
with current sentence-transformers releases (checked on 5.5.0 and 5.7.0: `Dense.__init__()
missing 'in_features' and 'out_features'`) because of the model's saved layer
configuration. It fails the same way with plain `SentenceTransformer(...)`, so it
is not specific to this project; the other local models load normally.

Prefixes are configured per registry entry (`query_prefix`,
`document_prefix`, or `query_prompt_name`/`document_prompt_name`); entries
without them embed text unchanged. The plain `e5` and `bge` keys keep their
original behavior (no prefixes), so existing indexes still match.

**An index built with a key must be queried with the same key.** Documents and
queries are embedded with that key's prefixes, so `e5` and `e5-prefixed`
produce different, incompatible vectors even though the underlying model is the
same (they also get different collection names). Metrics that compare two
plain texts (answer relevancy, answer correctness) never use prefixes, whatever
the key.

`SentenceTransformerEmbedder(track_truncation=True)` (off by default) counts
texts longer than the model's `max_seq_length` and reports them in the
evaluation results. OpenRouter embedders report token usage and cost when the
API returns it.

### LLM-based chunking and enrichment

Two pipeline stages can optionally call an LLM during ingestion instead of running purely deterministic logic — a strategy/enrichment opts into this via `"requires_llm": true` in `chunking_strategies.json` / `enrichment_strategies.json`:

- **`--strategy lumber_chunker`** — uses an LLM to detect content-shift boundaries instead of fixed-size windows.
- **`--enrichment llm_metadata`** — uses an LLM to extract and explain domain terms/acronyms per chunk, instead of the deterministic `metadata_tagging`/`acronym_expansion` strategies.

Both share the same connection, configured the same way:

```bash
open-telco-rag ingest 3gpp \
  --strategy lumber_chunker \
  --enrichment llm_metadata \
  --llm-backend local \
  --llm-base-url http://localhost:8000/v1 \
  --chunking-model Qwen/Qwen2.5-32B-Instruct \
  --enrichment-model Qwen/Qwen2.5-32B-Instruct
```

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-backend` | `local` | `local` talks to any OpenAI-compatible server — **vLLM**, Ollama, llama.cpp, etc. — on `--llm-base-url`. `openrouter` reuses `OPENROUTER_API_KEY`, useful for testing without a GPU. |
| `--llm-base-url` | `http://localhost:8000/v1` (or `LOCAL_LLM_BASE_URL`) | Where to reach the local server, e.g. after `vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000`. Ignored for `--llm-backend openrouter`. |
| `--chunking-model` | `Qwen/Qwen2.5-32B-Instruct` (local) / `openai/gpt-4o-mini` (openrouter) | Model name sent for `lumber_chunker`'s chunking calls. |
| `--enrichment-model` | `google/qwen-2.5-coder-32b` | Model name sent for `llm_metadata`'s enrichment calls. Can differ from `--chunking-model` — both are just the `"model"` field sent to the same `--llm-base-url` server. |

Connectivity is checked once, up front, before any files are processed — a bad endpoint/model fails fast with one clear message instead of every parsing worker independently discovering (and retrying against) a dead server.

**Minimizing LLM calls across a grid**: neither chunking nor enrichment depends on the embedding model, so for a sweep over many embedding models, chunk and enrich each (chunking config, enrichment strategy) pair once, cache the result, and repeat only the cheap LLM-free embedding step per model — rather than rerunning the LLM passes for every model.

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
| `--collection` | `source_model` | Override collection name. Repeat it to search several collections of one database with the same embedder: the query is embedded once, each collection is searched with `--search-method`, and the ranked lists are fused by rank (reciprocal rank fusion, k=60). Collections must agree on model, prefixes and distance space; a name cannot contain a comma |
| `--embed-backend` | `local` | Backend for computing vector embeddings (`local` or `openrouter`) |
| `--resolve-parents` | off | Expand retrieved child chunks into full parent contexts |

`search` prints one hit per source file and heading. With the default `text_baseline` strategy the stored
`heading` is empty, so all hits from one file collapse into a single result and fewer than `--top-k` results
are shown; ingest with `--strategy hierarchical_markdown` to store section headings.

### Reranker models

| Key | Model | Notes |
|-----|-------|-------|
| `ms-marco-minilm` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast, general-purpose |
| `ms-marco-large` | cross-encoder/ms-marco-MiniLM-L-12-v2 | Larger, more accurate |
| `bge-reranker` | BAAI/bge-reranker-large | Strong on technical domains |
| `otel-reranker-0.6b` | OTel-Reranker-0.6B | Telecom-domain fine-tuned (recommended) |
| `otel-reranker-4b` | OTel-Reranker-4B | Higher accuracy, requires GPU |

The `otel-*` rerankers truncate the query and each document to 512 / 1024
characters before scoring (unchanged default). A registry entry can override
this with optional `query_max_chars` / `doc_max_chars` keys; truncation counts
are written to the evaluation results (`reranker_stats`) when the reranker
reports them.

### Evaluate options

```bash
# Cloud (OpenRouter/Claude)
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

# Local — self-hosted GPU node serving both RAG generation and the
# eval-metrics judge on the same OpenAI-compatible endpoint
open-telco-rag evaluate 3gpp \
  --model bge --db-path chroma_db \
  --generator local --generator-model Qwen/Qwen3-32B \
  --llm-base-url http://localhost:8001/v1 \
  --rag-generator local --rag-generator-model openai/gpt-oss-120b \
  --rag-llm-base-url http://localhost:8000/v1 \
  --top-k 5 --output results/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `minilm` | Embedding model key. |
| `--db-path` | `chroma_db` | ChromaDB storage path. |
| `--collection` | `corpus_model` | Override collection name (e.g., `3gpp_r19_marker_bge`). |
| `--questions-file` | `configs/eval_questions/{corpus}.json` | Path to a custom QnA JSON file — supports mixed `mc`/`tf`/`open`/`fill_blank` question sets in one run, see "Per-format results" below. |
| `--top-k` | `5` | Chunks to retrieve per question. |
| `--threshold` | `0.5` | Similarity threshold for the `coverage_ratio` metric (fraction of retrieved chunks meeting this similarity floor). |
| `--output` | `results` | Directory to save result JSON files. |
| `--delay` | `1.0` | Seconds between questions for rate-limiting API calls. |
| `--no-save` | `off` | Print summary only, do not save JSON results. |
| `--generator` | `openrouter` | LLM provider for evaluation metrics (`claude`, `openrouter`, or `local`). |
| `--generator-model` | `None` | Model for evaluation metrics (Claude default: `claude-sonnet-4-20250514`, OpenRouter default: `meta-llama/llama-3.3-70b-instruct`, local: required — whatever your server is serving). |
| `--llm-base-url` | `None` | Base URL for `--generator local` (default: env var `LOCAL_LLM_BASE_URL`, or `http://localhost:8000/v1`). Ignored for claude/openrouter. |
| `--llm-api-key` | `None` | API key for `--generator local` (default: env var `LOCAL_LLM_API_KEY`). Omitted entirely if unset — unauthenticated local servers still work. Ignored for claude/openrouter. |
| `--rag-generator` | `None` | LLM provider for RAG answer generation (defaults to `--generator`). |
| `--rag-generator-model` | `None` | Model for RAG answer generation (defaults to `--generator-model`). |
| `--rag-llm-base-url` | `None` | Base URL for `--rag-generator local`. Defaults to `--llm-base-url` — set this explicitly to route RAG generation and metrics judging to two different endpoints/models. |
| `--rag-llm-api-key` | `None` | API key for `--rag-generator local`. Defaults to `--llm-api-key`. |
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
| `--search-method` | `dense` | Retrieval method: `dense`, `bm25` or `hybrid` (same choices as `search`/`ask`). |
| `--context-token-budget` | off | Optional: fill the generator's context up to this many tokens instead of a fixed chunk count (see below). |
| `--candidate-pool` | `30` with a budget | With `--context-token-budget`: candidates to retrieve and rerank before filling the budget. |
| `--budget-overflow` | `stop` with a budget | With `--context-token-budget`: `stop`, `skip` or `truncate` the first chunk that does not fit. |
| `--retrieval-only` | `off` | Optional: run retrieval and retrieval metrics only, with no generation, no LLM calls and no API keys needed. Does not update the leaderboard. |
| `--log-level` | unset | Global option (before the command): configure Python logging at `DEBUG`/`INFO`/`WARNING`/`ERROR` with timestamped lines on stderr. Unset, logging behaves as before. |
| `--record-retrieval-scores` | `off` | Optional: store the store distance of every retrieved chunk per question (`candidate_distances` / `used_distances`, aligned with the chunk ids). |
| `--record-judge-responses` | `off` | Optional: store every judge prompt and raw response per question in the results file. |

#### How scoring works, per question type

Every question goes through the same retrieve → (optional rerank) →
generate pipeline; scoring then branches by `answer_type`:

- **`mc` / `tf`** — deterministic exact-match against `reference_answer`,
  stored in `QuestionResult.accuracy` (1.0/0.0). A correctly-labeled
  *false* `tf` statement scores 1.0, not 0.0 — this checks whether the
  label matches, not whether the statement reads true at face value.
- **`open` / `fill_blank`** — four LLM-judged/embedding-based metrics in
  `QuestionResult.faithfulness` / `answer_relevancy` / `context_recall`
  (needs `reference_answer`) / `answer_correctness` (needs
  `reference_answer`): claim-decomposition-and-verification against the
  retrieved chunks, reverse-question embedding similarity, reference-
  statement attribution, and fact-F1 + semantic similarity, respectively.

`accuracy` and the four OE metrics are mutually exclusive per question —
each is `-1.0` ("not applicable") for the format it doesn't apply to, so a
run mixing `mc`+`tf`+`open` in one questions file doesn't blend two
different kinds of score into one uninterpretable mean.

#### Per-format results

`CorpusResult.by_format` (and the "Per-format breakdown" section printed
after every run) reports metrics separately per `answer_type`, in addition
to the overall aggregate — needed for exactly the mixed `mc`+`tf`+`open`
questions-file case: run one `--questions-file` containing all three
formats, and get an accuracy figure for `mc`, a separate accuracy figure
for `tf`, and separate faithfulness/relevancy/context-recall/correctness
figures for `open`, rather than one flat number across everything. Also
saved in the leaderboard row (`by_format` column) and the JSON output file.

#### Using your own LLM endpoint

Any OpenAI-compatible server (vLLM, Ollama, a hosted gateway, ...) can serve
the LLM stages. Point the CLI at it with the existing options; nothing is
hardcoded to a particular host:

```bash
export LOCAL_LLM_BASE_URL=http://my-host:8000/v1     # or pass --llm-base-url
export LOCAL_LLM_API_KEY=...                         # optional; omit for unauthenticated servers

# answer generation for `ask`
open-telco-rag ask "What does the AMF do?" --source-type 3gpp \
  --generator local --generator-model my-model

# evaluation: one endpoint for the judge, another (or the same) for answers
open-telco-rag evaluate 3gpp --model bge \
  --generator local --generator-model my-judge-model --llm-base-url http://judge-host:8001/v1 \
  --rag-generator local --rag-generator-model my-model --rag-llm-base-url http://rag-host:8000/v1

# LLM-based chunking / enrichment during ingest
open-telco-rag ingest 3gpp --strategy lumber_chunker \
  --llm-backend local --llm-base-url http://my-host:8000/v1 --chunking-model my-model
```

`--llm-base-url` / `--rag-llm-base-url` fall back to `LOCAL_LLM_BASE_URL`, then
to `http://localhost:8000/v1`. Use `--generator claude` or `openrouter` for the
hosted providers instead (their keys come from `ANTHROPIC_API_KEY` /
`OPENROUTER_API_KEY`).

#### Optional: budgeted context

By default the generator receives the top `--top-k` chunks (or
`--rerank-top-k` after reranking). With `--context-token-budget N` the
context is instead filled in rank order up to `N` tokens (`cl100k_base`), so
runs with different chunk sizes can be compared at the same context size:
`--candidate-pool` candidates are retrieved (and reranked) first, then
`--budget-overflow` decides what happens at the first chunk that does not fit:
`stop` (strict rank order), `skip` (keep looking for smaller chunks) or
`truncate` (cut that chunk to fill the budget exactly). `--top-k` and
`--rerank-top-k` are ignored in this mode. `--candidate-pool` and
`--budget-overflow` require a budget; giving either without one is an error.
Budgeted runs add `context_tokens`, `n_chunks_used`, `budget`, `fill_ratio` and
`budget_stats` (what the fill did: candidates, kept, skipped, truncated, stopped by
policy, pool exhausted, empty) per question, and `mean_context_tokens` / `mean_fill_ratio` to the totals.

#### Optional: retrieval-only runs

`--retrieval-only` skips answer generation and judging, so it is cheap and
needs no LLM key. Retrieval (reranker and budget respected) and every
retrieval metric are computed; accuracy and the four LLM-judged metrics stay
at their "not applicable" value. With `--include-baseline` the baseline pass
still runs normally (and then needs LLM access).

#### Reproducibility record

Every saved result contains a `run_record` (new in this version; older files
simply lack it) describing what was run:

- `code`: package version, git commit and whether the working tree was modified (both
  `"unknown"` when there is no `.git` directory, e.g. an installed wheel);
  `environment`: Python, platform and the versions of the key dependencies
- `settings`: the resolved evaluation settings, including every optional feature
  (search method, reranking, budget, retrieval-only, keyword/metadata injection,
  generation temperature, top-p, prompt mode, concurrency, ...)
- `embedder` (model key and name, dimensions, prefixes, sequence limit),
  `reranker` (model and truncation limits), `tokenizer` (`cl100k_base`), and under
  `extra.judge` the judge provider and model
- `index`: collection name, number of chunks, distance space and the settings the
  collection was ingested with (chunking and enrichment, read from a small
  `.ingest_settings_<collection>.json` that `ingest` writes next to the database;
  `configurations_seen` is above 1 if the collection was ingested under more
  than one configuration); `delete` removes it together with the collection
- `dataset`: question file name, number of questions and a SHA-256 of the
  questions as evaluated
- `started_at` / `finished_at` (UTC), and `randomness`: the pipeline has no seeded
  randomness (only retry jitter), so there is no seed to record; generation is
  deterministic only at temperature 0 and only as far as the provider is

The record never contains API keys, base URLs or local file paths. Results are
otherwise deterministic for the same inputs: questions keep their order under
`--max-concurrency`, and retrieval and metrics use no random choices.

#### Metrics reference

Retrieval metrics (computed for every non-baseline question):

| Metric | Meaning |
|--------|---------|
| `mean_similarity`, `min_similarity`, `max_similarity` | Similarity of the retrieved chunks to the query (`exp(-distance)`). |
| `coverage_ratio` | Fraction of retrieved chunks with similarity at or above `--threshold`. |
| `distinct_sources_retrieved` | Number of distinct source documents among the retrieved chunks. |
| `canonical_hit` | 1 if the question's expected source (and release, if given) appears in a retrieved chunk's metadata or id, 0 if not, -1 when the question has no (or only a too-generic) canonical source. |
| `exact_chunk_hit` | 1 if the exact chunk the question was generated from was retrieved (matched by chunk id, or by text containment across configurations), 0 if not, -1 without chunk provenance. |
| `candidate_chunk_rank`, `used_chunk_rank` | 1-based rank of that chunk before / after reranking; 0 = not found, -1 = not applicable. The corpus-level `mean_reciprocal_rank` is the mean of `1/used_chunk_rank` (0 when not found) over questions with provenance. |
| `span_hit_at_k`, `span_hit_in_context` | Chunker-agnostic companions to `exact_chunk_hit`: 1 if any retrieved chunk (respectively any chunk placed in the budgeted context) overlaps the question's evidence span, 0 if none does. See the three states below. |

**Span metrics have three states.** They are computed only for questions that
carry an evidence span (`evidence_doc_id`, `evidence_char_start`,
`evidence_char_end` in the question file):

- absent (`None`, left out of the result files and totals): the question has no
  evidence span, so the metric does not apply. Runs without any spans report
  no span metrics at all.
- `-1`: the question has a span, but no retrieved chunk carries character
  offsets (the collection was ingested without `--target-tokens` /
  `--max-tokens`), so it cannot be checked.
- `0` or `1`: a real result.

The `span_hit_*_rate` totals are the mean over questions with a real value.

With `--resolve-parents` on a token-sized `parent_child` collection, a retrieved
child is replaced by its parent text and the span metrics use the parent's
span (collections ingested before this was recorded fall back to the child's).

**Evidence-span fields in question files.** A question file is
`{"questions": [...]}`. To enable the span metrics, each question needs three
optional fields, next to the usual `question`, `corpus`, `answer_type`, ...:

| Field | Meaning |
|-------|---------|
| `evidence_doc_id` | The source document, as the relative file path used at ingest (the part of a chunk id before the first `::`). |
| `evidence_char_start` | Start offset (inclusive) of the evidence in the cleaned text of that document. |
| `evidence_char_end` | End offset (exclusive). |

"Cleaned text" is what ingestion sees after the source's `load()` and `clean()`.
`scripts/add_evidence_spans.py` computes the three fields for a question file
whose questions carry `chunk_id` and `source_text` (see the script's `--help`);
questions it cannot match are kept unchanged, without spans. Files without these
fields work as before and simply produce no span metrics.

Generation metrics: `accuracy` (mc/tf exact match; abstentions count as 0 but
are tracked separately), `faithfulness`, `answer_relevancy`, `context_recall`
and `answer_correctness` (open questions, LLM-judged; see "How scoring works"
above).

**Denominators.** Every mean or rate skips the `-1` "not applicable" value, so each
one is taken over the questions it applies to (and that did not fail):

| Metric | Computed over |
|--------|---------------|
| `mean_accuracy` | mc/tf questions that produced an outcome; `accuracy` per question is 1 for correct and 0 for incorrect or abstained |
| `mean_accuracy_attempted` | correct / (correct + incorrect), abstentions excluded |
| `abstention_rate`, `wrong_rate` | abstained (resp. incorrect) / all mc/tf outcomes |
| `weighted_score_pct` | (mc/tf correct + 5 x open correct) / (mc/tf total + 5 x open total) x 100; open questions without a reference answer are left out |
| `mean_faithfulness`, `mean_answer_relevancy` | open questions that were scored (non-baseline for faithfulness) |
| `mean_context_recall`, `mean_answer_correctness` | open questions that have a reference answer |
| `canonical_hit_rate`, `exact_chunk_hit_rate`, span rates | questions where the check applies (value 0 or 1, not -1) |
| `mean_similarity`, `mean_coverage_ratio`, `mean_distinct_sources_retrieved` | questions that ran retrieval without failing; N/A (-1) if none did |
| latency, token and cost totals | every question, including failed ones |

**Failed questions.** A question that fails (network error, missing key, ...) is
recorded with `error` and `error_type` and the run continues. When at least one
question failed, the results add `n_errors`, `error_rate` and `errors_by_type`
(also per answer type), and the summary prints a "FAILED questions" line; runs
with no failures do not have these keys. Failed questions are excluded from the
retrieval means (`mean_similarity`, coverage, distinct sources) and from every
rate, so a failure never counts as a score of zero; `n_questions` still includes
them. If **every** question of a corpus fails, `evaluate` still saves the results
but prints an error, skips the leaderboard row and exits with status **3**
(status 1 stays "the command itself failed"); partial failures exit 0.

Optional per-question fields (`difficulty`, the span and budget fields,
`judge_calls`) appear in result files only when they were set. Run-level
`reranker_stats` and `embedder_stats` (truncation counts, OpenRouter token
usage and cost) appear only when the components report them.

---

### Searching one or several collections

A database folder (`--db-path`) can hold many collections. A collection holds chunks made with **one embedder and
one distance space**; it can hold the chunks of one source or of several.

| Setup | Search one asset | Search all assets |
|---|---|---|
| One collection, several sources (e.g. `3gpp` and `camara` ingested into `--collection docs_bge`) | not possible | `search ... --collection docs_bge` |
| One collection per source, same embedder and distance space | `--collection camara_bge` | `--collection 3gpp_bge --collection camara_bge` |
| One collection per source, **different embedders** | `--collection X --model <its model>` | not possible: one query vector cannot be compared with vectors of another model |
| Same embedder, **different distance space** (`l2` vs `cosine`, set at ingest with `--distance-metric`) | works | refused, naming the collection; re-ingest one with the same `--distance-metric` |
| Same model name, different prefixes | works | refused |
| A collection that does not exist | | an error naming it; nothing is created |

How several collections are combined: each collection is searched with `--search-method` (`dense`, `bm25` or
`hybrid`; hybrid still fuses inside the collection). The query is embedded once. The ranked lists are then merged
by reciprocal rank fusion (k=60), keyed by collection and chunk id, so equal ids in two collections never merge.
Only ranks are used. Distances and BM25 scores are not mixed because they are not comparable: the same chunk
scored 1.31 in an `l2` collection and 0.66 in a `cosine` one, and a BM25 score depends on what else is in the
collection. `--top-k`, `--rerank`, `--rerank-top-k` and `--resolve-parents` apply after the merge. Each result
shows its collection, and the distance label shows that collection's real space (`L2`, `COSINE`, `IP`). A
collection without an `.ingest_settings_*.json` file is checked by its stored vector size only, with a warning;
an empty collection contributes nothing and prints a note.

```bash
open-telco-rag search "near-RT RIC control loop" --model bge --db-path chroma_db \
  --collection oran_bge --collection 3gpp_bge --top-k 8
```

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
from open_telco_rag.generators.local_llm import LocalLLMClient
from open_telco_rag.core.pipeline import ingest, search, generate

# Ingest (Using OpenRouter for High-Throughput Bulk Processing)
source   = ORANSource()
embedder = OpenRouterEmbedder(model_key="bge", batch_size=200) # Throttled safely for OpenAI limits
store    = ChromaStore(path="chroma_db", collection_name="oran_bge", metadata={"hnsw:space": "cosine"})

# llm_metadata needs an LLM client explicitly passed as enrichment_llm_client
# — LocalLLMClient talks to any OpenAI-compatible server (vLLM, Ollama, etc.);
# OpenRouterLLMClient (same module) is the cloud alternative.
llm_client = LocalLLMClient(base_url="http://localhost:8000/v1")

ingest(
    source=source, 
    store=store, 
    embedder=embedder, 
    data_dir="data/oran/",
    strategy="sliding_window_tokens",
    chunk_tokens=512,
    overlap_tokens=50,
    enrichment="llm_metadata",
    enrichment_model="Qwen/Qwen2.5-32B-Instruct",
    enrichment_llm_client=llm_client,
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
│   │   ├── openrouter_embedder.py
│   │   ├── calls.py             # embed_query / embed_plain helpers
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
├── scripts/                     # helper tools (embedder smoke test, cost estimate, throughput timing)
├── docs/graphics/               # pipeline diagrams and tables
├── tests/                       # pytest suite (incl. golden default-behavior tests)
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

## Helper scripts

Small tools in `scripts/` (run from the repository root, after `pip install -e .`):

| Script | Purpose |
|--------|---------|
| `add_evidence_spans.py` | Optional: add evidence spans to a question file so the span metrics can be computed. |
| `check_chunk_bound.py` | Read-only: per tokenizer, chunks over its limit, token-count ratio against cl100k, the largest chunks, text retained, and a calibration of the cl100k ceiling, for a collection or a staged directory. |
| `embedder_smoke_test.py` | Embed a few sample texts with every locally hosted embedder and report dimension, throughput and truncation. |
| `estimate_embedding_cost.py` | Estimate what re-embedding a collection would cost through OpenRouter (prices are a dated snapshot; `--refresh-pricing` fetches live ones). |
| `stage_3gpp.py` | Download the GSMA/3GPP Rel-19 Markdown files (only `raw.md` and the conversion log) from a pinned revision and stage them under names the 3GPP loader reads, with a manifest. See "Preparing the GSMA/3GPP Rel-19 data". |
| `time_embedding_throughput.py` | Measure local embedding throughput on real chunk texts from an existing collection. |
| `time_openrouter_embedding.py` | Same for OpenRouter. Makes real, paid API calls, so keep `--sample-size` small. |

Each supports `--help`.

## Testing

```bash
pip install -e ".[dev,local-embed,pdf,docx,cloud-embed]"
pytest
```

`tests/test_default_behavior_golden.py` and `tests/test_ingest_metadata_golden.py`
pin the output of the default settings (chunking, retrieval, evaluation
results, stored chunk metadata) against files in `tests/golden/`. They should
only ever change when a default is changed on purpose; regenerate with
`python tests/test_default_behavior_golden.py --write` (and the ingest
equivalent) in that case.

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

Optional keys: `query_prefix` / `document_prefix` (plain strings prepended to
queries / documents), `query_prompt_name` / `document_prompt_name`
(sentence-transformers named prompts) and `"backend": "openrouter"` (marks an
entry served through OpenRouter). Then use `--model my-model` in any CLI command.

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
- [GSMA/telco-retrieve-chunks](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks) — 4.6M chunks, 1.09B tokens across 7 SDOs
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
