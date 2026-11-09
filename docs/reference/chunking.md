# Chunking and enrichment

Chunking splits documents into passages before embedding. These options belong to `open-telco-rag ingest`; see [CLI reference](cli.md) for the other ingest flags.

## Chunking strategies

`--strategy` takes one of the keys of `configs/chunking_strategies.json`. Only `lumber_chunker` needs an LLM (its entry has
`requires_llm`; give it `--llm-backend`, `--llm-base-url` and `--chunking-model`). The other four need none.

| Strategy | Unit of `--chunk-size` | Headings stored | Notes |
|----------|------------------------|-----------------|-------|
| `text_baseline` (default) | characters | no | plain paragraph packing with `--overlap` |
| `sliding_window_tokens` | **tokens** (so is `--overlap`) | no | fixed token windows |
| `hierarchical_markdown` | characters | yes (`Title > Part A`) | splits on headings; `--overlap` is ignored; `--chunk-size` is a soft target: a block bigger than it is not split, so without the bound a chunk can be far larger (about 89,000 tokens were seen on the Rel-19 set) |
| `parent_child` | ignored | yes | see below |
| `lumber_chunker` | ignored | not checked | an LLM chooses the boundaries (needs an LLM server) |

- **`parent_child`** stores the whole parent text in the metadata of every child chunk (`full_parent_text`), so the database is
  about 6 times larger (measured in the GPU test run). The parent text is not bounded. `--resolve-parents` (a `search`, `ask` and
  `evaluate` flag, not an `ingest` flag) returns that parent text instead of the child, so what the generator sees is not
  bounded either. `--chunk-size` and `--overlap` are silently ignored for it.
- **The bound** (`--bound-tokens`, see "Optional: hard token bound") applies to all strategies as a final pass over the
  chunks. It does not cover a parent's text or text added by `--enrichment`.
- **Sizes and the config file.** The CLI default is `--chunk-size 1024 --overlap 200` for every strategy (the per-strategy
  `default_size` values in the JSON are not used). The choices of `--strategy` and the `requires_llm` flag are read from
  `configs/chunking_strategies.json` in the current directory first, then from the installed package. The step that decides
  whether `--chunk-size` is counted in tokens looks for a file called `chunking_strategies.json` in the **current
  directory** (not under `configs/`), which the repository does not have; it is skipped, and the strategy itself reads the
  number in its own unit. So run `ingest` from the repository root, and pass explicit sizes when you do not.

## Optional: token-based chunk sizing

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

## Optional: hard token bound

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

## LLM-based chunking and enrichment

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

## Marker preprocessing

For table-heavy corpora (ETSI, TM Forum), the Marker API produces higher quality chunks by preserving table structure and headings:

```bash
open-telco-rag ingest etsi --model bge --use-marker --data data/etsi/
```

Requires `DATALAB_API_KEY` in `.env`. See [datalab.to](https://www.datalab.to) for pricing.
