# Architecture


```
Documents (PDF / DOCX / YAML / Markdown)
        │
        ▼
   Source class          load() → clean() → metadata()
   (per SDO)             optional: Marker API for PDF
        │
        ▼
   Dynamic Chunker       Routes text to selected strategies (e.g. baseline, tokens, 
                         parent-child, lumberchunker) via configs
        │
        ▼
   Enrichment            Pre-embedding passes: Structural Tagging, Acronym Expansion 
                         or dynamic LLM Domain Expansion (metadata tracking)
        │
        ▼
   Embedder              SentenceTransformerEmbedder (local GPU/CPU)
                         OR OpenRouterEmbedder (Cloud API ThreadPool)
        |                 general-purpose or OTel telecom-domain models
        |
        ▼
   ChromaDB              Persistent vector store via configurable space metrics 
                         (L2 Euclidean geometric, Cosine angular, Inner Product)
        │
        ▼
   Search                Multi-method retrieval pipeline: Dense vector, Sparse keyword (BM25),
                         or Reciprocal Rank Fusion (Hybrid)
                         optional: cross-encoder or OTel reranking
        |
        ▼
   Generator             any LLM via OpenRouter or direct API
        │
        ▼
   Answer + sources
```

## Project Structure

```
open-telco-rag/
├── open_telco_rag/
│   ├── types.py
│   ├── core/
│   │   ├── pipeline.py          # Master orchestrator tracking stats and logging
│   │   ├── chunking.py          # Implements multi-strategy splitters (Token, etc.)
│   │   ├── enrichment.py        # Implements metadata tagging, glossary, and LLM-aided expansion
│   │   ├── retrieval.py         # Implements Multi-Method retrieval passes (Dense, BM25, and RRF)
│   │   ├── cleaning.py
│   │   └── marker_loader.py     # Datalab Marker API mixin
│   ├── sources/                 # one class per SDO
│   ├── embedders/
│   │   ├── sentense_transformers.py
│   │   ├── openrouter_embedder.py
│   │   ├── calls.py             # embed_query / embed_plain helpers
│   │   └── reranker.py          # CrossEncoderReranker, OTelReranker
│   ├── generators/
│   │   ├── claude.py
│   │   └── openrouter.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── questions.py
│   │   ├── results.py
│   │   └── metrics/
│   │       ├── generation.py
│   │       ├── retrieval.py
│   │       └── ground_truth.py
│   ├── challenge/               # challenge command group: export, check, score (see docs/challenge/)
│   └── stores/
│       └── chroma.py
├── cli/
│   └── main.py
├── scripts/                     # helper tools (embedder smoke test, cost estimate, throughput timing)
├── docs/graphics/               # pipeline diagrams and tables
├── tests/                       # pytest suite (incl. golden default-behavior tests)
├── notebooks/                   # quickstart cookbook
├── configs/
│   ├── embedding_models.json    # embedder registry
│   ├── reranker_models.json     # reranker registry
│   ├── profiles/                # named settings presets (challenge.json)
|   ├── chunking_strategies.json     # Central registry for character/token splitters
|   ├── enrichment_strategies.json   # Central registry for tagging, glossary, & LLM expansion
│   └── eval_questions/          # evaluation question sets per corpus
└── pyproject.toml
```
