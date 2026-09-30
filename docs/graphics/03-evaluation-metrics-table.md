# Evaluation Metrics

**Source:** `open_telco_rag/evaluation/{evaluate.py,metrics/*.py}`

| Metric | Evaluation Stage | What it Measures | Ground Truth Required? | Description / Formula Intent |
|---|---|---|---|---|
| `faithfulness` | Generation | Hallucination | No | LLM decomposes the answer into atomic claims, then checks each against the retrieved chunks. Score = supported claims ÷ total claims. |
| `answer_relevancy` | Generation | On-topic-ness | No | LLM reverse-engineers *N* questions the answer could be responding to; score is mean cosine similarity between those and the original question. |
| `context_recall` | Generation | Retrieval sufficiency | Yes | LLM breaks the reference answer into statements and checks how many are attributable to the retrieved chunks. 0 = retrieval missed the relevant content entirely. |
| `answer_correctness` | Generation | Factual accuracy | Yes | Weighted blend (default 0.5/0.5) of an LLM-judged factual F1 (correct / incorrect / missing facts vs. reference) and answer↔reference embedding cosine similarity. |
| `accuracy` (mc / tf) | Generation | Deterministic correctness | Yes | Structured verdict (preferred, via schema-constrained generation) or prose-fallback parsing, classified three ways: correct / incorrect / abstained — abstention counts as 0 but is tracked separately from a confident wrong answer. |
| `cosine_sim_stats` | Retrieval | Match strength | No | Mean / min / max similarity (`exp(−distance)`) across the retrieved set. |
| `coverage_ratio` | Retrieval | Retrieval confidence | No | Fraction of retrieved chunks meeting a similarity threshold — low values suggest retrieval settled for weak matches to fill top-k. |
| `source_coverage` | Retrieval | Source diversity | No | Count of distinct source documents among retrieved chunks — 1 means single-document, no cross-document synthesis. |
| `canonical_source_hit` | Retrieval | Right-document recall | Yes | Binary: does the question's expected canonical source (e.g. spec number) appear among retrieved chunks? −1 when no canonical source is defined for the question. |
