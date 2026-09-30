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
| `--batch-size` | 64 / 500 | Parallel payload size. Defaults to 64 locally, or 500 for high-throughput OpenRouter bulk ingestion |
| `--embed-max-workers` | `10` | OpenRouter embeddings only: in-flight requests per `embed()` call |

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
  `target_tokens`.
- `parent_child`: `target_tokens` sizes the child chunks (parents default to
  4x), so `max_tokens` bounds the retrieved chunk.
- The tokenizer is only initialized when token sizing is used.

```python
from open_telco_rag.core.chunking import chunk_document
chunks = chunk_document(text, "text_baseline", target_tokens=200, max_tokens=256)
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
Budgeted runs add `context_tokens`, `n_chunks_used`, `budget` and `fill_ratio`
per question, and `mean_context_tokens` / `mean_fill_ratio` to the totals.

#### Optional: retrieval-only runs

`--retrieval-only` skips answer generation and judging, so it is cheap and
needs no LLM key. Retrieval (reranker and budget respected) and every
retrieval metric are computed; accuracy and the four LLM-judged metrics stay
at their "not applicable" value. With `--include-baseline` the baseline pass
still runs normally (and then needs LLM access).

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

Generation metrics: `accuracy` (mc/tf exact match; abstentions count as 0 but
are tracked separately), `faithfulness`, `answer_relevancy`, `context_recall`
and `answer_correctness` (open questions, LLM-judged; see "How scoring works"
above). A question that fails is recorded with `error` and `error_type`.

Optional per-question fields (`difficulty`, the span and budget fields,
`judge_calls`) appear in result files only when they were set. Run-level
`reranker_stats` and `embedder_stats` (truncation counts, OpenRouter token
usage and cost) appear only when the components report them.

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
| `embedder_smoke_test.py` | Embed a few sample texts with every locally hosted embedder and report dimension, throughput and truncation. |
| `estimate_embedding_cost.py` | Estimate what re-embedding a collection would cost through OpenRouter (prices are a dated snapshot; `--refresh-pricing` fetches live ones). |
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
