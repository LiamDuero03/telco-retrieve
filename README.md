# TelcoRetrieve

TelcoRetrieve answers questions about telecom standards. It finds the right passages in specifications such as 3GPP and O-RAN, then writes an answer that cites them.

<p align="center">
  <img src="docs/graphics/telcoretrieve-pipeline.png" alt="TelcoRetrieve pipeline: documents are chunked and embedded into an index, then retrieved, reranked and passed to the generator to produce an answer with sources" width="700">
</p>

## Installation

```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed,pdf,docx]"
```

For a GPU or CPU-only build of PyTorch, see [Installation](docs/reference/installation.md).

## Quick Start

Put the O-RAN specification files in `data/oran/`. Other corpora are covered in [Corpora](docs/reference/corpora.md).

Add a key for the answer step. Copy the example file, then set `OPENROUTER_API_KEY` in `.env`:

```bash
cp .env.example .env
```

Build an index. An index is a store you can search. The embedding model `bge` turns text into vectors:

```bash
open-telco-rag ingest oran --model bge --data data/oran/
```

Find passages:

```bash
open-telco-rag search "near-RT RIC control loop" --source-type oran --top-k 5
```

Ask a question. The answer lists its sources:

```bash
open-telco-rag ask "What interfaces does the Near-RT RIC expose?" --source-type oran
```

## Learn more

- [Documentation index](docs/README.md)
- [Reference](docs/reference/)
- [Architecture](docs/architecture.md)
- [TelcoRetrieve Challenge](docs/challenge/README.md)

## License

Apache 2.0. See [LICENSE](LICENSE).
