# Chunking & Enrichment Strategy Breakdown

**Source:** `configs/{chunking,enrichment}_strategies.json`, `open_telco_rag/core/{chunking,enrichment}.py`

Chunking strategy is selected per ingestion run; enrichment is layered on top
and is independent of which chunker produced the chunk.

## Chunking

### hierarchical_markdown — rule-based
- **Mechanism:** Splits on Markdown heading lines (`#`–`#####`), maintaining a stateful breadcrumb stack per depth so every chunk carries its full heading path (e.g. "§5 > §5.2 > §5.2.1"). Flushes a chunk when either a new heading starts or the size ceiling (default 4000 chars) is hit.
- **Trigger / use case:** Well-structured Markdown corpora (3GPP TSpec-LLM exports) where heading depth is a reliable proxy for topic boundaries.

### sliding_window_tokens — rule-based
- **Mechanism:** Tokenizes with `tiktoken` (cl100k_base), then slides a fixed-size window (default 512 tokens) forward by `window − overlap` tokens (default overlap 50) until the document is exhausted.
- **Trigger / use case:** Unstructured or heading-poor documents where a model-native token budget matters more than document layout.

### parent_child — rule-based
- **Mechanism:** Runs `hierarchical_markdown` to produce large parent segments (default 2048 chars), then re-splits each parent into small child chunks (default 512 chars) by paragraph. Each child stores `parent_id` and the full parent text in its metadata for retrieval-time expansion.
- **Trigger / use case:** Two-tier retrieval — embed and match on precise child text, but hand the generator the full parent context (see Figure 1's `resolve_parents` branch).

### lumber_chunker — LLM-aided
- **Mechanism:** First splits on top-level Markdown headers to bound parallel work, then runs the LumberChunker paper's rolling-window algorithm inside each section: an LLM is shown a window of θ tokens (default 550) of ID-tagged paragraphs and asked to name the first paragraph where topic content shifts. On LLM failure, falls back to a fixed 3-paragraph block.
- **Trigger / use case:** Corpora where structural cues (headings, fixed sizes) don't track topic boundaries closely enough — trades ingestion-time LLM cost for boundary quality.

### text_baseline — fallback
- **Mechanism:** Splits on blank-line paragraph breaks, accumulating up to a character ceiling (default 1024). On overflow, keeps the last `overlap` characters (default 200) and snaps backward to the nearest sentence terminator so chunks don't open mid-sentence.
- **Trigger / use case:** Generic fallback for any format without a more specific strategy — the only strategy every source can always fall back to.

## Enrichment

### none — rule-based
- **Mechanism:** Identity pass — chunk text reaches the embedder unmodified.
- **Trigger / use case:** Baseline for isolating chunking-strategy effects without enrichment as a confound.

### metadata_tagging — rule-based
- **Mechanism:** Writes each provided context key (SDO, working group, document heading, …) into the chunk's stored metadata *and* prepends a `[Document Context → …]` header string directly into the embedded text.
- **Trigger / use case:** When structural provenance (which spec, which working group) should itself be part of what the embedding captures, not just sidecar metadata.

### acronym_expansion — rule-based
- **Mechanism:** Matches chunk text against a compiled, word-boundary-safe regex built from a telecom acronym glossary (cached per process); appends an `[Acronym Expansions]` block listing every match found.
- **Trigger / use case:** Corpora dense with unexpanded acronyms (O-RAN, 3GPP) where retrieval on the spelled-out term would otherwise miss the chunk.

### llm_metadata — LLM-aided
- **Mechanism:** Sends each chunk to an LLM (local vLLM server or OpenRouter, concurrency-capped) asking for a JSON array of term definitions; prepends a `Technical Context & Definitions` block built from the response. A truncated or malformed response leaves the chunk unenriched rather than failing the run.
- **Trigger / use case:** Domain concepts a fixed glossary can't cover (implicit context, multi-word technical phrases) — trades ingestion-time LLM cost for richer grounding text.
