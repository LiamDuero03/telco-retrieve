# Corpora

## Supported corpora

| Corpus | Source | Format |
|--------|--------|--------|
| O-RAN Alliance | O-RAN R003–R005, all WGs | PDF, DOCX |
| 3GPP | TSpec-LLM dataset, Rel-8 through Rel-19 | Markdown |
| ETSI | ENI, ZSM, MEC, NFV, SAI, F5G, RIS | PDF, YAML |
| ITU-T | All 27 active series (G, Y, Q, X, M…) | PDF |
| CAMARA | Fall25, 75+ API repos | Markdown, YAML |
| GSMA | OPG, IG, AID, SGP, RCS and more | PDF, DOCX |
| TM Forum | TMF Open APIs, ODA Component specs ([tmforum-rand/tmf-otel](https://github.com/tmforum-rand/tmf-otel)) | YAML, JSON |

## Pre-built dataset

The indexed corpus (4.6M+ chunks, 1B+ tokens, 7 standards bodies) is on Hugging Face as [GSMA/telco-retrieve-chunks](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks). Download one corpus or all of it without running ingestion:

```python
from datasets import load_dataset

ds = load_dataset("GSMA/telco-retrieve-chunks", data_files="etsi.jsonl")   # one corpus
ds = load_dataset("GSMA/telco-retrieve-chunks")                            # all corpora
```

## Get the documents

Download O-RAN specifications from [o-ran.org](https://www.o-ran.org/specifications) into `data/oran/`.

**3GPP:** the CLI does not download specs; you supply the files. The Markdown conversions of the 3GPP
specifications are in the TSpec-LLM dataset on Hugging Face (`rasoul-nikbakht/TSpec-LLM`). Read its terms before use: the dataset is gated, so log in with
`huggingface-cli login` and accept them first. It holds a `.md` and a `.docx` for every spec, and the 3GPP
source ingests **both** extensions, so download only the Markdown:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('rasoul-nikbakht/TSpec-LLM', repo_type='dataset', allow_patterns=['3GPP-clean/*.md'], local_dir='data/3gpp')"
open-telco-rag ingest 3gpp --data data/3gpp/3GPP-clean --strategy hierarchical_markdown
```

The layout is `3GPP-clean/Rel-<N>/<NN>_series/<spec>.md`; the release, series, spec number and version come from
those folder and file names (`28312-i11.md` is TS 28.312, Rel-18). Files whose names do not follow that pattern
still ingest but may lack `spec_number`. All releases are millions of chunks: see "Resume" in the [CLI reference](cli.md) and ingest one
release at a time if you want the smallest possible restart unit.

For the staged GSMA/3GPP Rel-19 set, see [Preparing the GSMA/3GPP Rel-19 data](3gpp-data.md).
