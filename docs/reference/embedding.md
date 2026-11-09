# Embedding and reranker models

`--model` takes a key from `configs/embedding_models.json`; `--reranker-model` takes a key from `configs/reranker_models.json`.

## Core embedding models

Available embedding models:

| Key | Model | Dimensions | Notes |
|-----|-------|------------|-------|
| `minilm` | all-MiniLM-L6-v2 | 384 | Fast — good for development |
| `mpnet` | all-mpnet-base-v2 | 768 | Balanced quality and speed |
| `e5` | intfloat/e5-large-v2 | 1024 | High quality — recommended for production |
| `bge` | BAAI/bge-large-en-v1.5 | 1024 | Strong on technical domains |
| `otel-109m` | OTel-Embedding-109M | 768 | Telecom-domain fine-tuned — 109M params |
| `otel-0.6b` | OTel-Embedding-0.6B | 1024 | Telecom-domain fine-tuned — 0.6B params |

## More embedding models and prefixes

`--model` takes any key from `configs/embedding_models.json`. Besides the
original keys (`minilm`, `mpnet`, `e5`, `bge`, `otel-*`, `or-*`) the registry
includes:

| Key | Model | Notes |
|-----|-------|-------|
| `bge-m3` | BAAI/bge-m3 | local; no prefix needed |
| `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | local; instruction-style query prefix |
| `embeddinggemma-300m` | google/embeddinggemma-300m | local; uses the model's named prompts (`Retrieval-query` / `Retrieval-document`); loads in float16 on a GPU, which gave NaN vectors in a GPU test |
| `embeddinggemma-300m-fp32` | google/embeddinggemma-300m | the same model and prompts with `"dtype": "float32"`: loaded in float32 and run without autocast on a GPU, to avoid that overflow. A separate key, so its chunk ids and collection differ from `embeddinggemma-300m`. 768 dims. Finite vectors for 19,100 chunks in a GPU run; retrieval quality not measured yet |
| `qwen3-embedding-8b` | qwen/qwen3-embedding-8b | served through OpenRouter (`--embed-backend openrouter`, needs `OPENROUTER_API_KEY`), 4096 dims, instruction-style query prefix |
| `e5-prefixed` | intfloat/e5-large-v2 | opt-in variant of `e5` with `query: ` / `passage: ` prefixes |
| `bge-prefixed` | BAAI/bge-large-en-v1.5 | opt-in variant of `bge` with BGE's query instruction |

```bash
export OPENROUTER_API_KEY=...            # your key; never commit it
open-telco-rag ingest 3gpp --model qwen3-embedding-8b --embed-backend openrouter
open-telco-rag evaluate 3gpp --model qwen3-embedding-8b --embed-backend openrouter
```

### OpenRouter embeddings: measured behaviour and usage notes

Measured once, with `qwen3-embedding-8b` through OpenRouter on a 30-file ingest (these come from that one run; price and
rate limits can change):

- The vectors were 4096-dimensional and finite. The provider was DeepInfra, at 0.01 USD per million tokens.
- 19,100 chunks used 2,693,619 billed tokens (one special token per text more than a plain count of the chunk texts) and
  cost 0.027 USD. The hash of the chunk texts was identical to the one from the local embedders, so every embedder saw the
  same chunks.
- There were no retries and no 429 answers.
- The speed was about 26 chunks per second at the default settings, which projects to about 12 hours for the full corpus and
  1.4 to 1.5 USD at the measured price.

Usage notes:

- The key starts with `sk-or-v1-` and is read from the environment variable `OPENROUTER_API_KEY`.
- A 401 "Missing Authentication header" means the value is not an OpenRouter key (an empty variable, or another provider's
  key).
- A 402 "Insufficient credits" is about the **account** balance, whatever limit the key itself has.
- At the default `--batch-size` (500 for OpenRouter) one `embed()` call holds 500 texts, which the embedder sends as requests
  of 200: at most 3 in flight. `--embed-max-workers` (default 10) therefore has no effect unless you set it below 3. A
  `--batch-size` of 4000 makes 20 requests per call, so up to `--embed-max-workers` of them (10 by default, 20 if you raise
  it) can be in flight; I have not measured this.
- `ingest` prints no token counts or cost for OpenRouter embedders (the embedder keeps the totals and `evaluate` reports
  them); look at your OpenRouter account to see what a run cost.

**Known issue: `otel-300m`.** Loading `farbodtavakkoli/OTel-Embedding-300M` fails
with current sentence-transformers releases (checked on 5.5.0 and 5.7.0: `Dense.__init__()
missing 'in_features' and 'out_features'`) because of the model's saved layer
configuration. It fails the same way with plain `SentenceTransformer(...)`, so it
is not specific to this project; the other local models load normally.

**Precision on a GPU.** A registry entry may carry `"dtype": "float16" | "bfloat16" | "float32"`. On a GPU it sets both the
dtype the model is loaded in and the autocast dtype; `float32` runs without autocast. Without a `dtype`, nothing changed:
float16 on a GPU, nothing special on a CPU. Only `embeddinggemma-300m-fp32` has one. The OTel reranker entries in
`configs/reranker_models.json` accept the same field (default: float16 on a GPU, float32 on a CPU).

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

## Reranker models

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

## Adding an embedding model or a reranker

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
