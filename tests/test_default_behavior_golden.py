"""Golden regression check: DEFAULT-settings behaviour must not change.

Captures, on a fixed sample input, what the pipeline does with default settings:
  * chunking output for every strategy (defaults, and with explicit char/token sizes)
  * retrieval (dense, threshold, rerank, parent resolution) against a fake store
  * what evaluate_question() passes to search()/generate() and the serialized
    QuestionResult / aggregate() output (so no new fields leak into old output)

The golden file (tests/golden/default_behavior.json) was generated from pristine
upstream/main. Any change to default behaviour makes this test fail; opt-in
features must leave these outputs byte-identical.

Regenerate ONLY when a default change is deliberate and approved:
    python tests/test_default_behavior_golden.py --write
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

GOLDEN = Path(__file__).parent / "golden" / "default_behavior.json"

# Keys whose values depend on wall-clock time / machine and are not behaviour.
_VOLATILE = ("latency", "elapsed", "timestamp", "duration", "seconds", "_ms")


def _sample_markdown() -> str:
    parts = ["# Sample Spec\n\nIntroduction paragraph describing the AMF and SMF roles."]
    for i in range(1, 7):
        parts.append(f"## Section {i}\n")
        for j in range(1, 5):
            parts.append(
                f"### Clause {i}.{j}\n\n"
                + " ".join(
                    f"The network function NF-{i}-{j}-{k} exchanges N{k} messages "
                    f"with peer {i * j + k} during procedure P{i}{j}{k}."
                    for k in range(1, 9)
                )
                + "\n\n"
                + f"Second paragraph of clause {i}.{j}: the UE shall retry after timer T{i}{j} expires, "
                f"and the AMF shall not release the context until acknowledgement {j} arrives."
            )
    return "\n\n".join(parts)


def _chunk_dump(chunks) -> list[dict]:
    out = []
    for c in chunks:
        d = asdict(c) if is_dataclass(c) else dict(vars(c))
        d["_sha256"] = hashlib.sha256(d["text"].encode("utf8")).hexdigest()[:16]
        out.append(d)
    return out


class _FakeLLMClient:
    """Deterministic stand-in for the LumberChunker LLM: always names paragraph 2."""

    def __init__(self):
        msg = SimpleNamespace(content="Answer: ID 2")
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        self.completion = SimpleNamespace(create=lambda **kw: resp)


def _chunking_golden() -> dict:
    from open_telco_rag.core.chunking import ChunkingStrategy, chunk_document

    text = _sample_markdown()
    out: dict = {}
    for strat in ChunkingStrategy:
        extra = {"llm_client": _FakeLLMClient()} if strat == ChunkingStrategy.LUMBER_CHUNKER else {}
        out[f"{strat.value}/defaults"] = _chunk_dump(chunk_document(text, strat, **extra))
        out[f"{strat.value}/source_type"] = _chunk_dump(
            chunk_document(text, strat, source_type="3gpp", **extra))
    # explicit char sizes (how cli/pipeline pass sizes today)
    out["text_baseline/chunk_size=1024,overlap=200"] = _chunk_dump(
        chunk_document(text, "text_baseline", chunk_size=1024, overlap=200))
    out["hierarchical_markdown/chunk_size=2000"] = _chunk_dump(
        chunk_document(text, "hierarchical_markdown", chunk_size=2000))
    out["parent_child/parent=1500,child=300"] = _chunk_dump(
        chunk_document(text, "parent_child", parent_size=1500, child_size=300))
    out["sliding_window_tokens/chunk_size=256,overlap=32"] = _chunk_dump(
        chunk_document(text, "sliding_window_tokens", chunk_size=256, overlap=32))
    out["sliding_window_tokens/metadata_tagging"] = _chunk_dump(
        chunk_document(text, "sliding_window_tokens", chunk_size=256, overlap=32,
                       enrichment="metadata_tagging"))
    return out


class _FakeEmbedder:
    def embed(self, texts):
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf8")).digest()
            v = [b / 255.0 for b in h[:8]]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs


class _FakeStore:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self):
        from open_telco_rag.types import SearchResult

        emb = _FakeEmbedder()
        self._rows = []
        for i in range(20):
            text = f"Chunk {i}: AMF procedure {i} covers N{i % 5} registration."
            meta = {"source": f"doc{i % 3}.md"}
            if i % 4 == 0:
                meta.update(parent_id=f"p{i // 8}", full_parent_text=f"PARENT {i // 8} full text")
            self._rows.append((f"c{i}", text, meta, emb.embed([text])[0]))
        self._SR = SearchResult

    def search(self, vector, n_results=5):
        def dist(v):
            return 1.0 - sum(a * b for a, b in zip(v, vector))

        ranked = sorted(self._rows, key=lambda r: (dist(r[3]), r[0]))[:n_results]
        return [self._SR(text=r[1], distance=round(dist(r[3]), 12), metadata=r[2], doc_id=r[0])
                for r in ranked]


class _FakeReranker:
    def rerank(self, query, candidates, top_k=5):
        return list(reversed(candidates))[:top_k]


def _sr_dump(results) -> list[dict]:
    return [asdict(r) for r in results]


def _retrieval_golden() -> dict:
    from open_telco_rag.core.pipeline import search
    from open_telco_rag.core.retrieval import execute_retrieval

    store, emb = _FakeStore(), _FakeEmbedder()
    q = "AMF registration procedure"
    out = {
        "execute_retrieval/defaults": _sr_dump(execute_retrieval(q, store, emb)),
        "execute_retrieval/top_k=3": _sr_dump(execute_retrieval(q, store, emb, top_k=3)),
        "execute_retrieval/threshold=0.5": _sr_dump(
            execute_retrieval(q, store, emb, top_k=10, threshold=0.5)),
        "execute_retrieval/rerank": _sr_dump(
            execute_retrieval(q, store, emb, top_k=4, reranker=_FakeReranker(), rerank_top_k=2)),
        "execute_retrieval/resolve_parents": _sr_dump(
            execute_retrieval(q, store, emb, top_k=8, resolve_parents=True)),
        "pipeline.search/defaults": _sr_dump(search(q, store, emb)),
    }
    return out


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()
                if not any(m in k.lower() for m in _VOLATILE)}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _evaluate_golden() -> dict:
    import open_telco_rag.core.pipeline as pipeline_module
    from open_telco_rag.evaluation.evaluate import evaluate_question
    from open_telco_rag.evaluation.questions import EvalQuestion
    from open_telco_rag.evaluation.results import aggregate, question_to_dict
    from open_telco_rag.types import SearchResult

    calls: dict = {"search": [], "generate": []}
    chunks = [
        SearchResult(text="The AMF handles registration.", distance=0.2,
                     metadata={"source": "23501-j00.md"}, doc_id="c1"),
        SearchResult(text="The SMF manages sessions.", distance=0.3,
                     metadata={"source": "23502-j00.md"}, doc_id="c2"),
    ]

    def fake_search(*a, **kw):
        calls["search"].append({"args": len(a), "kwargs": {k: repr(v) for k, v in sorted(kw.items())
                                                            if k not in ("store", "embedder")}})
        return list(chunks)

    def fake_generate(question, results, *a, **kw):
        calls["generate"].append({"question": question, "n_results": len(results),
                                  "texts": [r.text for r in results],
                                  "kwargs": sorted(kw)})
        return {"answer": "B", "sources": [], "chunks_used": len(results)}

    orig_search, orig_generate = pipeline_module.search, pipeline_module.generate
    pipeline_module.search, pipeline_module.generate = fake_search, fake_generate
    try:
        mc = EvalQuestion(question="Which interface?", keywords=[], corpus="3gpp",
                          answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B")
        tf = EvalQuestion(question="The AMF requires 3 PLMN IDs.", keywords=[], corpus="3gpp",
                          answer_type="tf", reference_answer="False")
        results = [
            evaluate_question(mc, store=object(), embedder=object(),
                              rag_generator_name="local", rag_model_name="local-model"),
            evaluate_question(mc, store=object(), embedder=object(), baseline=True,
                              rag_generator_name="local", rag_model_name="local-model"),
            evaluate_question(tf, store=object(), embedder=object(), top_k=3,
                              rag_generator_name="local", rag_model_name="local-model"),
        ]
    finally:
        pipeline_module.search, pipeline_module.generate = orig_search, orig_generate

    return {
        "calls": calls,
        "question_results": [_scrub(question_to_dict(r)) for r in results],
        "aggregate": _scrub(aggregate(results)),
    }


def compute() -> dict:
    return json.loads(json.dumps({
        "chunking": _chunking_golden(),
        "retrieval": _retrieval_golden(),
        "evaluate": _evaluate_golden(),
    }, sort_keys=True, default=str))


def test_default_behavior_matches_golden():
    assert GOLDEN.exists(), f"missing {GOLDEN}; generate with --write on a known-good tree"
    expected = json.loads(GOLDEN.read_text(encoding="utf8"))
    actual = compute()
    assert actual.keys() == expected.keys()
    for section in expected:
        assert actual[section].keys() == expected[section].keys(), section
        for key in expected[section]:
            assert actual[section][key] == expected[section][key], f"{section}/{key} changed"


def test_golden_is_deterministic():
    assert compute() == compute()


if __name__ == "__main__":
    if "--write" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n", encoding="utf8")
        print(f"wrote {GOLDEN}")
    else:
        print("pass --write to regenerate the golden file")
