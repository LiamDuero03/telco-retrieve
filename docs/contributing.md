# Contributing

Contributions welcome. Please open an issue before submitting a pull request for significant changes.

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

The test suite needs these extras. They differ from the install extras (`local-embed,pdf,docx`, see [Installation](../README.md#installation)) by two:
`dev` (pytest, pytest-cov and ruff) and `cloud-embed` (httpx, which some tests import; the base dependency `anthropic`
already brings httpx in, so `cloud-embed` mainly makes it explicit). As with a normal
install, put the PyTorch install of "Installation" first. The suite needs no GPU and no API key. Some tests load the small
`all-MiniLM-L6-v2` model, which is downloaded on first use, so the first run needs network access (later runs use the
cache).

**Three tests are skipped by default**, so a normal run ends with "3 skipped":
- one in `tests/test_chunk_bound.py` and one in `tests/test_check_chunk_bound.py` fetch a real `tokenizer.json` from Hugging
  Face. To run them, set `RUN_REAL_TOKENIZER_TESTS=1` and have network access (`RUN_REAL_TOKENIZER_TESTS=1 pytest`; in
  PowerShell `$env:RUN_REAL_TOKENIZER_TESTS = 1` first);
- one in `tests/test_multi_collection_ondisk.py` repeats write-then-search on disk for about a minute. Set
  `RUN_ONDISK_STRESS=1` to run it; it needs no network.

`tests/test_default_behavior_golden.py` and `tests/test_ingest_metadata_golden.py`
pin the output of the default settings (chunking, retrieval, evaluation
results, stored chunk metadata) against files in `tests/golden/`. They should
only ever change when a default is changed on purpose; regenerate with
`python tests/test_default_behavior_golden.py --write` (and the ingest
equivalent) in that case.

## Adding a new source

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
