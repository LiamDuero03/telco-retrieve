# Challenge options: embedder, prompt builder, answer parser

These options belong to `open-telco-rag challenge export`. The [challenge guide](../challenge/README.md) covers the normal workflow.

## Use your own embedder

Use an embedder that matches the index. Add it to `configs/embedding_models.json` (see [embedding reference](embedding.md)), ingest with `--model <key>`, then export with the same key.

- `--embedder-path <folder>` loads a local sentence-transformers folder. `--embedder-id <id>` loads a Hugging Face model.
- `--query-prefix` sets the prefix added to queries.
- `export` stops if the embedder differs from the one that built the index. `--allow-embedder-mismatch` turns the stop into a warning.

## Prompt builder

`--prompt-builder module:function` replaces the prompt. The module must be importable; the current directory is.

```python
# my_hooks.py
from open_telco_rag.challenge.prompt import build_prompt

def prompt(question, passages, max_prompt_tokens):
    return build_prompt(question, passages[:3], max_prompt_tokens)
```

```bash
open-telco-rag challenge export ... --prompt-builder my_hooks:prompt
```

The builder returns a `Prompt(messages, input_prompt, n_tokens, n_passages)`. The CSV column is exactly `input_prompt`. The generator gets exactly `messages`.

## Answer parser

`--answer-parser module:function` replaces the parsing of the model output. `options` is `{}` for true/false questions.

```python
# my_hooks.py
from open_telco_rag.challenge.normalise import normalise_answer

def answer(model_text, options):
    return normalise_answer(model_text.splitlines()[0] if model_text else "", options)
```

## Other flags

`--profile`, `--seed`, `--temperature`, `--max-answer-tokens` and `--timeout` (seconds to wait for one generator request) are listed by `open-telco-rag challenge export --help`. `--resolve-parents` replaces child chunks by their parents for indexes built with the `parent_child` strategy. `--document-prefix` and `--embedder-dimensions` are recorded for consistency with ingest and do not change the queries.
