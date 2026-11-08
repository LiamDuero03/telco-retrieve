# `challenge` command group

Builds, checks and scores a submission CSV (`ID,input_prompt,answer`) for the challenge.
Defaults: OTel-Embedding-0.6B, OTel-Reranker-0.6B (keep 5), `google/gemma-3n-E4B-it`,
hybrid search, top-k 10, prompt cap 4096 tokens.

## Happy path

1. Build an index. `ingest` defaults to MiniLM and L2, so pass the model explicitly.
   L2 is the normal ingest default; the published fixed grid uses cosine. `export` reads
   whatever metric the index has.

       open-telco-rag ingest 3gpp --data DATA --model otel-0.6b \
           --db-path db --collection 3gpp_otel

2. Export answers from the Gemma endpoint (vLLM, include `/v1`). The API key, if any, comes
   from `LOCAL_LLM_API_KEY` only.

       export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
       open-telco-rag challenge export --questions Test.csv --out submission.csv \
           --db-path db --collection 3gpp_otel

   `--llm-base-url` does the same as the variable; `--generator-model` or
   `LOCAL_LLM_MODEL` changes the model name sent to the endpoint.

3. Check the file against the submission spec (stricter than the scorer):

       open-telco-rag challenge check --submission submission.csv --questions Test.csv

4. Score it with the official scripts (reference CSV has `ID,answer`):

       open-telco-rag challenge score --submission submission.csv --reference ref.csv

## Notes

- Export prints the effective settings as one `[settings]` line on stderr, and progress
  every 25 questions. The last line reports rows, prompt tokens and the number of empty
  answers (model output that could not be parsed).
- `--resume` continues from `<out>.partial` after a crash; it refuses if the header or IDs
  do not match the questions.
- A custom embedder: add an entry pointing to the folder in
  `configs/embedding_models.json`, ingest with `--model <key>`, then export with
  `--model <key>` or `--embedder-path <folder>` (optionally `--query-prefix`). Export
  aborts if the embedder is not the one the index was built with
  (`--allow-embedder-mismatch` turns that into a warning).

## Hooks

Both take `module:function` and the module must be importable (the current directory is).

    # my_hooks.py
    from open_telco_rag.challenge.prompt import build_prompt

    def prompt(question, passages, max_prompt_tokens):
        return build_prompt(question, passages[:3], max_prompt_tokens)

    --prompt-builder my_hooks:prompt

The builder returns a `Prompt(messages, input_prompt, n_tokens, n_passages)`; the CSV column
is exactly `input_prompt` and the generator gets exactly `messages`.

    # my_hooks.py
    def answer(model_text, options):
        return model_text.strip().split()[0].upper()[:1]

    --answer-parser my_hooks:answer

`options` is `{}` for true/false questions. Run `open-telco-rag challenge <command> --help`
for every flag.
