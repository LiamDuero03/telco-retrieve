"""Span-based hit metric sits alongside exact_chunk_hit and is OPT-IN.

New keys appear only when the question data carries an evidence span; existing
metric names, values and aggregates are unchanged either way.
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.core.pipeline import _process_file_worker
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.evaluation.results import (
    CorpusResult, aggregate, print_summary, question_to_dict, result_to_dict,
)
from open_telco_rag.types import SearchResult

SPAN_KEYS = {"span_hit_at_k", "span_hit_in_context"}
SPAN_CORPUS_KEYS = {"span_hit_at_k_rate", "span_hit_in_context_rate"}


def _chunks(with_offsets=True):
    def meta(start, end):
        m = {"source": "doc.md"}
        if with_offsets:
            m.update(char_start=start, char_end=end)
        return m

    return [
        SearchResult(text="alpha", distance=0.1, metadata=meta(0, 100), doc_id="doc.md::m::0"),
        SearchResult(text="beta", distance=0.2, metadata=meta(100, 200), doc_id="doc.md::m::1"),
    ]


def _question(**over):
    base = dict(question="Which interface?", keywords=[], corpus="3gpp", answer_type="mc",
                options={"A": "N1", "B": "E2"}, reference_answer="B")
    base.update(over)
    return EvalQuestion(**base)


def _run(monkeypatch, question, chunks):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: list(chunks))
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": len(chunks)})
    return evaluate_question(question, store=object(), embedder=object(),
                             rag_generator_name="local", rag_model_name="local-model")


SPAN = dict(evidence_doc_id="doc.md", evidence_char_start=120, evidence_char_end=180)


def test_question_without_span_leaves_span_fields_unset(monkeypatch):
    result = _run(monkeypatch, _question(), _chunks())
    assert result.span_hit_at_k is None and result.span_hit_in_context is None
    assert SPAN_KEYS.isdisjoint(question_to_dict(result))


def test_aggregate_and_summary_have_no_span_output_without_spans(monkeypatch):
    result = _run(monkeypatch, _question(), _chunks())
    agg = aggregate([result])
    assert SPAN_CORPUS_KEYS.isdisjoint(agg)
    assert all(SPAN_CORPUS_KEYS.isdisjoint(v) for v in agg["by_format"].values())
    corpus = CorpusResult(corpus="3gpp", model="m", top_k=5,
                          question_results=[result], **agg)
    assert SPAN_CORPUS_KEYS.isdisjoint(result_to_dict(corpus))
    out = io.StringIO()
    with redirect_stdout(out):
        print_summary(corpus)
    assert "Span" not in out.getvalue()


def test_question_with_span_gets_span_hits(monkeypatch):
    result = _run(monkeypatch, _question(**SPAN), _chunks())
    assert result.span_hit_at_k == 1 and result.span_hit_in_context == 1  # overlaps chunk 1
    d = question_to_dict(result)
    assert d["span_hit_at_k"] == 1 and d["span_hit_in_context"] == 1


def test_span_miss_and_not_applicable(monkeypatch):
    miss = _run(monkeypatch, _question(evidence_doc_id="doc.md", evidence_char_start=500,
                                       evidence_char_end=600), _chunks())
    assert miss.span_hit_at_k == 0
    legacy = _run(monkeypatch, _question(**SPAN), _chunks(with_offsets=False))
    assert legacy.span_hit_at_k == -1  # collection has no offsets


def test_aggregates_gain_span_keys_only_with_spans(monkeypatch):
    with_span = _run(monkeypatch, _question(**SPAN), _chunks())
    agg = aggregate([with_span])
    assert agg["span_hit_at_k_rate"] == 1.0
    assert agg["by_format"]["mc"]["span_hit_at_k_rate"] == 1.0
    corpus = CorpusResult(corpus="3gpp", model="m", top_k=5,
                          question_results=[with_span], **agg)
    out = io.StringIO()
    with redirect_stdout(out):
        print_summary(corpus)
    assert "Span hit@k rate" in out.getvalue()


def test_existing_metrics_identical_with_or_without_span(monkeypatch):
    plain = question_to_dict(_run(monkeypatch, _question(), _chunks()))
    spanned = question_to_dict(_run(monkeypatch, _question(**SPAN), _chunks()))
    for key in SPAN_KEYS:
        spanned.pop(key)
    volatile = ("latency", "_ms")
    plain = {k: v for k, v in plain.items() if not any(m in k for m in volatile)}
    spanned = {k: v for k, v in spanned.items() if not any(m in k for m in volatile)}
    assert plain == spanned


# ── stored chunk metadata ─────────────────────────────────────────────────────

DOC = "\n\n".join(f"Paragraph {i}: the AMF handles registration procedure {i}." for i in range(30))


def _ingest_metadata(**chunk_kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        doc = data_dir / "Rel-18" / "28312-i11.md"
        doc.parent.mkdir()
        doc.write_text(DOC, encoding="utf8")
        kwargs = {"strategy": "text_baseline", **chunk_kwargs}
        result = _process_file_worker((doc, data_dir, "3gpp", False, kwargs, 10, "minilm"))
    assert result["status"] == "success", result["error"]
    return result["metadatas"]


def test_token_sized_ingest_stores_offsets():
    metas = _ingest_metadata(target_tokens=60)
    assert metas and all(m["char_start"] >= 0 and m["char_end"] > m["char_start"] and m["n_tokens"] > 0
                         for m in metas)


def test_default_ingest_stores_no_offsets():
    metas = _ingest_metadata(chunk_size=400, overlap=50)
    assert metas and all(not ({"char_start", "char_end", "n_tokens"} & set(m)) for m in metas)


# ── difficulty passthrough (opt-in) ───────────────────────────────────────────

def test_difficulty_absent_changes_no_output(monkeypatch):
    result = _run(monkeypatch, _question(), _chunks())
    assert result.difficulty is None
    assert "difficulty" not in question_to_dict(result)
    agg = aggregate([result])
    assert not any("difficult" in key for key in agg)
    assert all(not any("difficult" in key for key in v) for v in agg["by_format"].values())
    corpus = CorpusResult(corpus="3gpp", model="m", top_k=5, question_results=[result], **agg)
    assert "difficult" not in json.dumps(result_to_dict(corpus))
    out = io.StringIO()
    with redirect_stdout(out):
        print_summary(corpus)
    assert "ifficult" not in out.getvalue()


def test_difficulty_present_is_carried_into_the_question_result(monkeypatch):
    result = _run(monkeypatch, _question(difficulty="intermediate"), _chunks())
    assert question_to_dict(result)["difficulty"] == "intermediate"
    # carried per question only: no new aggregate or summary output
    assert not any("difficult" in key for key in aggregate([result]))
