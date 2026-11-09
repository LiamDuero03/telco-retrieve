# Documentation

| Page | Content |
|---|---|
| [Main README](../README.md) | Install, quick start, commands |
| [CLI reference](reference/cli.md) | Options of ingest, search, ask, info, delete, push-hf and pull-hf |
| [Chunking and enrichment](reference/chunking.md) | Chunk strategies, token bound, LLM enrichment, Marker |
| [Embedding and reranker models](reference/embedding.md) | Model registries, prefixes, how to add a model |
| [Evaluation](reference/evaluation.md) | `evaluate` options and metrics |
| [Python API](reference/python-api.md) | Ingest, search and generate from Python |
| [Architecture](architecture.md) | Pipeline stages and the project layout |
| [Contributing](contributing.md) | Helper scripts, tests, adding a source |
| [TelcoRetrieve Challenge](challenge/README.md) | Guide for challenge participants |
| [Graphics](graphics/README.md) | Pipeline and metrics figures |

## Glossary

- **TelcoRetrieve**: the tool. An open RAG pipeline for telecom specifications.
- **Pipeline**: the stages ingest, chunk, embed, retrieve, rerank, generate.
- **CLI**: the `open-telco-rag` command. The Python package is `open_telco_rag`.
- **TelcoRetrieve Challenge**: a GSMA challenge that uses TelcoRetrieve. Its commands are in `open-telco-rag challenge`.
- **Profile**: a named preset of settings in `configs/profiles/`. The `challenge` profile holds the defaults of `challenge export`.
- **Index, collection**: the stored chunks and vectors of one corpus, built with one embedder.
