# Documentation

TelcoRetrieve is part of the [GSMA Open Telco AI](https://www.open-telco.ai/).

| Page | Content |
|---|---|
| [Main README](../README.md) | Install and quick start |
| [Installation](reference/installation.md) | PyTorch builds, API keys |
| [Corpora](reference/corpora.md) | Supported corpora, pre-built dataset, getting documents |
| [Prepare 3GPP Rel-19 data](reference/3gpp-data.md) | Stage and ingest the Rel-19 specifications |
| [CLI reference](reference/cli.md) | Options of ingest, search, ask, info, delete, push-hf and pull-hf |
| [Chunking and enrichment](reference/chunking.md) | Chunk strategies, token bound, LLM enrichment, Marker |
| [Embedding and reranker models](reference/embedding.md) | Model registries, prefixes, how to add a model |
| [Evaluation](reference/evaluation.md) | `evaluate` options and metrics |
| [Python API](reference/python-api.md) | Ingest, search and generate from Python |
| [Architecture](architecture.md) | Pipeline stages and the project layout |
| [Contributing](contributing.md) | Helper scripts, tests, adding a source |
| [TelcoRetrieve Challenge](challenge/README.md) | Guide for challenge participants |
| [Challenge options](reference/challenge-options.md) | Own embedder, prompt builder, answer parser |
| [Graphics](graphics/README.md) | Pipeline and metrics figures |

## TelcoRetrieve Cookbook

The notebooks in [`notebooks/`](../notebooks/) are the TelcoRetrieve Cookbook, one per corpus. Each covers ingestion, search, question answering and evaluation. The PDF [Telco-Retrieve-Guidebook.pdf](../Telco-Retrieve-Guidebook.pdf) is the Guidebook.

| Notebook | Corpus |
|----------|--------|
| [`01_oran_quickstart.ipynb`](../notebooks/01_oran_quickstart.ipynb) | O-RAN Alliance R005 |
| [`02_3gpp_quickstart.ipynb`](../notebooks/02_3gpp_quickstart.ipynb) | 3GPP Release 8 through 19 |
| [`03_camara_quickstart.ipynb`](../notebooks/03_camara_quickstart.ipynb) | CAMARA Fall25 |
| [`04_tmforum_quickstart.ipynb`](../notebooks/04_tmforum_quickstart.ipynb) | TM Forum Open APIs & ODA Components |
| [`05_gsma_quickstart.ipynb`](../notebooks/05_gsma_quickstart.ipynb) | GSMA Open Gateway |
| [`06_itu_quickstart.ipynb`](../notebooks/06_itu_quickstart.ipynb) | ITU-T |
| [`07_etsi_quickstart.ipynb`](../notebooks/07_etsi_quickstart.ipynb) | ETSI ISG groups |



## Glossary

- **TelcoRetrieve**: the tool. An open RAG pipeline for telecom specifications.
- **Pipeline**: the stages ingest, chunk, embed, retrieve, rerank, generate.
- **CLI**: the `open-telco-rag` command. The Python package is `open_telco_rag`.
- **TelcoRetrieve Challenge**: a GSMA challenge that uses TelcoRetrieve. Its commands are in `open-telco-rag challenge`.
- **Profile**: a named preset of settings in `configs/profiles/`. The `challenge` profile holds the defaults of `challenge export`.
- **Index, collection**: the stored chunks and vectors of one corpus, built with one embedder.
