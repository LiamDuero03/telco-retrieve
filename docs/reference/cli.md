# CLI reference

Every command and flag is listed by `open-telco-rag <command> --help`; that output is the source of truth. This page explains the common options and what they do.

| Command | Purpose |
|---|---|
| `ingest <source>` | Chunk, embed and store documents |
| `search <query>` | Search the vector database |
| `ask <question>` | Search, then generate an answer with sources |
| `evaluate <corpus>` | Run the evaluation suite, see [evaluation](evaluation.md) |
| `info` | Show what a database holds |
| `delete <collection>` | Delete a collection |
| `push-hf`, `pull-hf` | Push or pull a database folder on the Hugging Face Hub |

## Ingest options

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
| `--target-tokens` | off | Optional: size chunks in tokens instead of characters (see [Token-based chunk sizing](chunking.md#optional-token-based-chunk-sizing)) |
| `--max-tokens` | off | Optional: hard cap on tokens per chunk (defaults to `--target-tokens`) |
| `--overlap-tokens` | off | Optional: token overlap between chunks (`text_baseline` and `sliding_window_tokens` only) |
| `--enrichment` | `none` | Pre-embedding strategy registry token mapped from enrichment_strategies.json |
| `--enrichment-model` | `google/qwen-2.5-coder-32b` | Model name to request for the `llm_metadata` enrichment strategy — sent to whichever server `--llm-backend`/`--llm-base-url` point at |
| `--distance-metric` | `l2` | Vector space index geometry calculation mapping: l2, cosine or ip |
| `--no-resume` | off | Disable resume — re-ingest all files. The BM25 rows of the chunks written again are replaced (and counted in the output); rows of other chunks in the collection are kept |
| `--fail-on-error` | off | Exit with code 4 when any file failed or the run stopped on an error (see "Failed files"); without it the exit code is 0 as always |
| `--use-marker` | off | Use Datalab Marker API for PDF preprocessing |
| `--collection` | `<source-type>_<model>` | Override collection name |
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
exits with status 0 by default even when some files failed, and it does not check that a parsed file's text is sensible
(a binary file named `.md` is ingested as text), so check the `Errors:` line or the CSV, and the chunk count with
`open-telco-rag info`.

**`--fail-on-error` and exit code 4.** With `--fail-on-error`, `ingest` exits with code 4 when the `Errors:` count is above
zero (a file that could not be parsed, or a file that did not get all its chunks stored) or when the run itself stopped on an
error: a failed Chroma or BM25 write, or an embedder that returned a vector with NaN or infinite values, a wrong length, or
the wrong number of vectors (the run stops before that batch is written and names the chunk ids). Files that are only
`Skipped:` do not count. A clean run exits 0 with or without the flag. After a failed write nothing more is dispatched and the
BM25 index is never ahead of Chroma, so a later run resumes the rest.

**Chunk ids** are `<path relative to --data>::<model key>::<chunk index>`, with the path written using the
operating system's separator: a collection built on Windows has backslashes in its ids. The evaluation metrics
compare ids independently of the separator; the stored ids are not rewritten.

Ingest also builds a per-collection SQLite FTS5 BM25 index alongside ChromaDB, enabling `--search-method bm25` and `--search-method hybrid` retrieval.

**Write reliability**: `ChromaStore` queries its Chroma installation's real max batch size live at construction (`ChromaStore.max_batch_size`, from `chromadb`'s own `get_max_batch_size()` — not a hardcoded constant, since this depends on the local SQLite build). Every internal write is capped at that value regardless of `--batch-size`, so a large `--batch-size` on a bulk/high-throughput run can no longer produce a Chroma `"Batch size of N is greater than max batch size of M"` crash partway through — previously possible on large runs once accumulated writes exceeded Chroma's real limit, sometimes only surfacing after the progress bar had already reported 100%. Reported `chunks_stored` counts also only ever reflect confirmed-successful writes, not chunks merely attempted, so a partial failure produces an accurate partial count instead of continuing to count chunks that were never actually persisted.

A failure in purely cosmetic bookkeeping *after* every chunk is already safely stored (writing the resume-progress JSON file, a CSV log row, or a run's `_ingest_complete.json` completion marker) is now isolated from the write-critical path — it's logged as a `[WARN]`, but it can no longer retroactively misreport a fully successful ingest as `[ERROR] Bulk indexing failed` with an inflated `error_files` count. This applies to both `ingest()` and `ingest_multi_enrichment()` — the latter isolates it per enrichment variant, so one variant's marker-write hiccup doesn't take down the reported status of every other variant in the same run.

**Config resolution**: `--strategy`/`--enrichment`'s available choices, and whether a given strategy/enrichment needs an LLM client, are read from `configs/chunking_strategies.json`/`configs/enrichment_strategies.json` relative to the installed package location as a fallback if no local `configs/` directory is found relative to the current working directory — so these are no longer silently degraded (to `text_baseline`-only / `none`-only, with LLM-requiring strategies going undetected) when the CLI isn't run from the exact repository root.

## Search and Ask options

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

`ask --label-sources` (off by default) puts a one-line label in front of each passage (collection, source type, file, release or version, heading), builds the "technical expert" label of the prompt from the sources retrieved unless `--source-type` is given, and lists collection and file per passage under `Sources:` (the file's path when two retrieved files share a name).

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
| `--collection` | `<source-type>_<model>` | Override collection name. Repeat it to search several collections of one database with the same embedder: the query is embedded once, each collection is searched with `--search-method`, and the ranked lists are fused by rank (reciprocal rank fusion, k=60). Collections must agree on model, prefixes and distance space; a name cannot contain a comma |
| `--embed-backend` | `local` | Backend for computing vector embeddings (`local` or `openrouter`) |
| `--resolve-parents` | off | Expand retrieved child chunks into full parent contexts |
| `--where-source-type` | off | Only chunks of this source type (`source_type` in the chunk metadata, e.g. `3gpp`, `camara`); repeat for several. For a collection that holds several sources. An unknown name lists the available ones. Dense search is filtered inside Chroma; BM25 hits are post-filtered from the best 10x candidates, so a rare source can come up short (a warning says so) |

`search` prints one hit per source file and heading. With the default `text_baseline` strategy the stored
`heading` is empty, so all hits from one file collapse into a single result and fewer than `--top-k` results
are shown; ingest with `--strategy hierarchical_markdown` to store section headings.

## Searching one or several collections

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

## Which asset did a passage come from? (`ask --label-sources`)

By default `ask` sends the chunk texts only. With `--label-sources` every passage starts with a label such as
`[collection: camara_bge | source_type: camara | file: QualityOnDemand/README.md | heading: Usage]`, the prompt's
"technical expert" label is built from the source types retrieved (unless you give `--source-type`), and
`Sources:` lists `collection: file` per passage in context order, with the file's path when two retrieved files
share a name.

```bash
open-telco-rag ask "Which API creates a boost session?" --model bge --db-path chroma_db \
  --collection camara_bge --label-sources
```

## Adding another asset such as CAMARA

CAMARA is already a source (`camara`): markdown (`.md`) and OpenAPI (`.yaml`, `.yml`) files, one folder per API
repository. Ingest it like any other source.

```bash
# its own collection (recommended): same embedder as the assets you want to search it with
open-telco-rag ingest camara --data data/camara --model bge --db-path chroma_db

# or into a collection that already holds another source
open-telco-rag ingest 3gpp   --data data/3gpp   --model bge --db-path chroma_db --collection docs_bge
open-telco-rag ingest camara --data data/camara --model bge --db-path chroma_db --collection docs_bge
```

The first form creates `camara_bge`. Rules that keep it working:

1. **Same embedder.** Assets searched together need the same `--model` and the same `--distance-metric`.
2. **Unique relative paths.** A chunk id is `<path relative to --data>::<model>::<n>`. Two sources that share a
   relative path (for example a `README.md` at the top of both data folders) get the same ids in a shared
   collection, so one file is skipped or replaces the other. Keep them in distinct folders, or use one
   collection per source.
3. **Use a relative `--data` path** (`data/camara`, not `C:\data\camara`). The CAMARA source takes the repo name
   from the first capitalised part of the path it is given; with an absolute path on Windows that part is the
   drive (`C:\`), so every chunk gets `repo_name: C:\`.
4. **Avoid `--no-resume` when ingesting a second source into a shared collection.** It re-ingests every file
   of that run and replaces the BM25 rows of their chunks; a separate collection per source needs none of this.
5. Release notes and changelogs (`CHANGELOG*.md`, `release-plan.yaml`) are skipped on purpose by the CAMARA
   source.

To remove one asset without touching the others (own collection per asset): `open-telco-rag delete camara_bge`.

## Info options

```bash
open-telco-rag info --db-path chroma_db --collection oran_bge
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db-path` | `chroma_db` | ChromaDB storage path |
| `--collection` | `None` | Collection name to inspect |

## Delete collection options

```bash
open-telco-rag delete my_collection --db-path chroma_db
```


| Flag | Default | Description |
|------|---------|-------------|
| `collection` | required | Name of the ChromaDB collection to permanently delete |
| `--db-path` | `chroma_db` | ChromaDB storage path |

## Hugging Face Hub: push-hf and pull-hf

```bash
open-telco-rag pull-hf <repo_id> <path_in_repo> --local-dir ./runs
```

The folder lands in `<local-dir>/<path_in_repo>`. 
