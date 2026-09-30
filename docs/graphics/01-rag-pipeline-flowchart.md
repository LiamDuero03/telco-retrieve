# Figure 1 — Retrieval-Augmented Generation Pipeline

**Source:** `open_telco_rag/core/{pipeline,chunking,enrichment,retrieval}.py`

End-to-end path from a raw SDO document to a cited, grounded answer. Diamonds
are per-request configuration choices, not fixed branches — every
ingest/search call selects one path through each decision point.

```mermaid
flowchart TB
    classDef sourceLayer fill:#0E7C86,stroke:#0E7C86,color:#fff
    classDef chunkLayer fill:#EEF1F6,stroke:#C2C9D6,color:#131A2B
    classDef enrichLayer fill:#FDF1E7,stroke:#B75E1B,color:#131A2B
    classDef storeLayer fill:#131A2B,stroke:#131A2B,color:#fff
    classDef retrieveLayer fill:#EEF1F6,stroke:#0E7C86,color:#131A2B
    classDef genLayer fill:#0E7C86,stroke:#0E7C86,color:#fff
    classDef decision fill:none,stroke:#8891A3,color:#5B6478,stroke-dasharray:3 3

    SRC["Source Layer<br/>O-RAN · 3GPP · ETSI · ITU-T · CAMARA · GSMA · TM Forum"]:::sourceLayer
    LOAD["load() → clean() → metadata()<br/>+ optional Marker API for PDF/DOCX tables"]

    CDEC{"Chunking strategy?"}:::decision
    C1["hierarchical_markdown<br/>heading-tree, stateful breadcrumbs"]:::chunkLayer
    C2["sliding_window_tokens<br/>fixed token window, % overlap"]:::chunkLayer
    C3["parent_child<br/>large parent → small child vectors"]:::chunkLayer
    C4["lumber_chunker (LLM)<br/>rolling-window shift detection"]:::chunkLayer
    C5["text_baseline<br/>paragraph + char overlap, sentence-snap"]:::chunkLayer

    EDEC{"Enrichment strategy?<br/>(optional)"}:::decision
    E1["metadata_tagging<br/>prepends SDO/heading header text"]:::enrichLayer
    E2["acronym_expansion<br/>glossary regex, appends definitions"]:::enrichLayer
    E3["llm_metadata<br/>LLM extracts &amp; prepends term glosses"]:::enrichLayer
    E4["none<br/>pass through unchanged"]:::enrichLayer

    EMBED["Embedder<br/>SentenceTransformer (local, 7 models) or<br/>OpenRouter (cloud, 2 models)"]
    STORE["ChromaStore (dense HNSW)<br/>+ SQLite FTS5 (BM25) — parallel index"]:::storeLayer

    QUERY(["Query"])
    QEMBED["embed(query)"]
    RDEC{"search_method?"}:::decision
    RD["dense<br/>ANN over embeddings"]:::retrieveLayer
    RB["bm25<br/>FTS5 keyword match"]:::retrieveLayer
    RH["hybrid<br/>reciprocal rank fusion"]:::retrieveLayer
    RERANK{"reranker configured?"}:::decision
    CE["cross-encoder rerank<br/>ms-marco / bge / OTel, top-k'"]:::retrieveLayer
    PARENT{"resolve_parents?"}:::decision
    PEXP["expand child → full parent text"]:::retrieveLayer

    GEN["Generator<br/>Claude · OpenRouter · Local (vLLM)"]:::genLayer
    ANSWER(["Grounded answer<br/>+ source citations"])

    SRC --> LOAD --> CDEC
    CDEC --> C1 & C2 & C3 & C4 & C5
    C1 & C2 & C3 & C4 & C5 --> EDEC
    EDEC --> E1 & E2 & E3 & E4
    E1 & E2 & E3 & E4 --> EMBED --> STORE

    QUERY --> QEMBED --> RDEC
    RDEC --> RD & RB & RH
    STORE -.retrieves from.-> RD
    STORE -.retrieves from.-> RB
    RD & RB & RH --> RERANK
    RERANK -->|yes| CE --> PARENT
    RERANK -->|no| PARENT
    PARENT -->|yes, parent_child only| PEXP --> GEN
    PARENT -->|no| GEN
    GEN --> ANSWER
```
