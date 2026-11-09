# Evaluation

## Evaluate options

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

### How scoring works, per question type

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

### Per-format results

`CorpusResult.by_format` (and the "Per-format breakdown" section printed
after every run) reports metrics separately per `answer_type`, in addition
to the overall aggregate — needed for exactly the mixed `mc`+`tf`+`open`
questions-file case: run one `--questions-file` containing all three
formats, and get an accuracy figure for `mc`, a separate accuracy figure
for `tf`, and separate faithfulness/relevancy/context-recall/correctness
figures for `open`, rather than one flat number across everything. Also
saved in the leaderboard row (`by_format` column) and the JSON output file.

### Using your own LLM endpoint

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

### Optional: budgeted context

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

### Optional: retrieval-only runs

`--retrieval-only` skips answer generation and judging, so it is cheap and
needs no LLM key. Retrieval (reranker and budget respected) and every
retrieval metric are computed; accuracy and the four LLM-judged metrics stay
at their "not applicable" value. With `--include-baseline` the baseline pass
still runs normally (and then needs LLM access).

### Reproducibility record

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
otherwise deterministic for the same inputs: questions keep their order, and retrieval and metrics use no random choices.

### Metrics reference

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
