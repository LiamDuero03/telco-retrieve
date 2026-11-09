# TelcoRetrieve Challenge

The TelcoRetrieve Challenge asks you to answer telecom questions with a retrieval pipeline and a small generator model. The `open-telco-rag challenge` command group turns a question file into a submission file, checks it and scores it. It uses the same pipeline as the rest of TelcoRetrieve (see the [main README](../../README.md)).

Scoring parameters are set by the organisers and published on the official challenge page. This guide does not repeat them; check that page for the current rules.

## What the challenge commands use by default

The defaults come from the `challenge` profile, [`configs/profiles/challenge.json`](../../configs/profiles/challenge.json). Flags override them.

| Setting | Default | Flag |
|---|---|---|
| Embedder (queries) | `otel-0.6b` | `--model` |
| Reranker | `otel-reranker-0.6b`, on, keeps 5 passages | `--rerank`, `--no-rerank`, `--reranker-model`, `--rerank-top-k` |
| Retrieval | `hybrid` (dense and BM25), 10 passages before reranking | `--search-method`, `--top-k` |
| Generator | `google/gemma-3n-E4B-it`, served by your endpoint | `--generator-model`, `--llm-base-url` |
| Sampling | temperature 0, seed 1234 | `--temperature`, `--seed` |
| Answer length | 8 generated tokens | `--max-answer-tokens` |
| Prompt length limit | set by the profile | `--max-prompt-tokens` |

Run `open-telco-rag challenge export --help` for every flag with its current default.

## Set up

1. Install the package as described in the [main README](../../README.md#installation). Reranking and the local embedder need the `local-embed` extra.
2. Get the public questions: the 100 questions of [`GSMA/telco-retrieve-QnA`](https://huggingface.co/datasets/GSMA/telco-retrieve-QnA) are in `Test_public.csv`. The file has the columns `ID,question,options` and no answers. Load IDs as text; some start with a zero.

   ```bash
   python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('GSMA/telco-retrieve-QnA', 'Test_public.csv', repo_type='dataset', local_dir='.'))"
   ```

3. Get an index that matches the default embedder. The published fixed grid in [`GSMA/telco-retrieve-chunks`](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks) holds ready-made indexes under `telcoretrievechallenge/`. The cell for the defaults is `text_baseline/otel-0.6b`:

   ```bash
   open-telco-rag pull-hf GSMA/telco-retrieve-chunks \
       telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b --local-dir ./runs
   ```

   The database is then in `./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db` and its collection is `otel-0.6b_text_baseline_b512_v256_full`. The cell is large (about 10 GB).

4. Start an OpenAI-compatible server for the generator, for example with vLLM, and note its URL (it ends in `/v1`).

   ```bash
   export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
   export LOCAL_LLM_API_KEY=...        # only if your server needs one
   ```

   The key is read from `LOCAL_LLM_API_KEY` only. `--llm-base-url` does the same as the variable, and `--generator-model` (or `LOCAL_LLM_MODEL`) sets the model name sent to the server.

## Run

```bash
DB=./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db
open-telco-rag challenge export --questions Test_public.csv --out submission.csv \
    --db-path $DB --collection otel-0.6b_text_baseline_b512_v256_full
open-telco-rag challenge check --submission submission.csv --questions Test_public.csv
open-telco-rag challenge score --submission submission.csv --reference Reference.csv
```

- `export` retrieves, builds the prompt, calls the generator and writes the submission. It never reads reference answers. It prints the effective settings as one `[settings]` line on stderr, progress every 25 questions, and a last line with the number of rows, prompt tokens and empty answers (model output that could not be parsed).
- `check` validates the file against the Test CSV: header, one row per ID, answers that are valid for the question type, no empty prompts, a prompt within the length limit, UTF-8 without BOM and LF line endings. It exits with status 1 when it finds an error.
- `score` runs the scoring scripts that ship with this repository and prints accuracy, token efficiency and the overall score. The reference CSV has the columns `ID,answer`; you need the reference file for the question set you score.

If you build your own index instead of downloading one, pass the embedder and the metric explicitly:

```bash
open-telco-rag ingest 3gpp --data DATA --model otel-0.6b --distance-metric cosine \
    --db-path db --collection 3gpp_otel
```

`ingest` defaults to `minilm` and `l2` (see the [CLI reference](../reference/cli.md)). The published fixed grid was built with cosine distance, so use `--distance-metric cosine` if you want your index to behave like it. `export` works with whatever metric the index has.

## The submission file

Columns: `ID,input_prompt,answer`.

- `input_prompt` is the system message, a blank line, and the user message, exactly as sent to the generator.
- `answer` is the option letter for a multiple-choice question and `TRUE` or `FALSE` for a true/false question.

## How tokens are counted

The length of a prompt is the number of `cl100k_base` tokens (tiktoken) in the whole `input_prompt`: the system message, the retrieved passages with their labels, the question, the options and the answer instruction. If a prompt is longer than `--max-prompt-tokens`, `export` drops the lowest-ranked passages until it fits. Fewer or shorter passages give shorter prompts; `--rerank-top-k` and `--top-k` are the usual levers.

## Replace parts of the pipeline

- **Embedder.** A fine-tuned or custom embedder is allowed as long as the index was built with the same one. Add it to `configs/embedding_models.json` (see [embedding reference](../reference/embedding.md)), ingest with `--model <key>`, then export with `--model <key>`, or point to a folder with `--embedder-path` (or a Hugging Face id with `--embedder-id`). `--query-prefix` sets the query prefix. `export` stops if the embedder is not the one the index was built with; `--allow-embedder-mismatch` turns that into a warning.
- **Prompt builder.** `--prompt-builder module:function` replaces the prompt. The module must be importable (the current directory is).

  ```python
  # my_hooks.py
  from open_telco_rag.challenge.prompt import build_prompt

  def prompt(question, passages, max_prompt_tokens):
      return build_prompt(question, passages[:3], max_prompt_tokens)
  ```

  The builder returns a `Prompt(messages, input_prompt, n_tokens, n_passages)`. The CSV column is exactly `input_prompt`, and the generator gets exactly `messages`.

- **Answer parser.** `--answer-parser module:function` replaces the parsing of the model output.

  ```python
  # my_hooks.py
  from open_telco_rag.challenge.normalise import normalise_answer

  def answer(model_text, options):
      return normalise_answer(model_text.splitlines()[0] if model_text else "", options)
  ```

  `options` is `{}` for true/false questions.

Other tuning flags: `--profile`, `--seed`, `--temperature`, `--max-answer-tokens`, `--timeout` (seconds to wait for one generator request), `--resume`. For an index built with the `parent_child` strategy, `--resolve-parents` replaces child chunks by their parents. `--document-prefix` and `--embedder-dimensions` are recorded for consistency with ingest and do not change the queries. Chunking and distance metric are choices of `ingest`; see the [chunking reference](../reference/chunking.md).

## FAQ

**The run stopped halfway.** Run the same command with `--resume`. It continues from `<out>.partial` and refuses if the header or the IDs do not match the questions. Without `--resume`, an existing `.partial` file is replaced.

**`export` says the embedder does not match the index.** Pass the model the index was built with, for example `--model otel-0.6b`. The error names the key.

**`check` reports an answer that is not an option.** The parser returned text that is not a valid letter or `TRUE`/`FALSE`. Look at the empty-answer count in the last `export` line, or use `--answer-parser`.

**`check` says the file starts with a BOM or has CR characters.** Do not open and save the CSV in a spreadsheet program. Write it with LF line endings and no BOM.

**Where do I see all flags?** `open-telco-rag challenge <command> --help`.
