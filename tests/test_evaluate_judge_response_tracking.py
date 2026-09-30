"""Tests for evaluate_question()'s opt-in judge-response persistence.

With record_judge_calls=True, QuestionResult.judge_calls captures every
(metric, prompt, raw_response) triple in call order; off by default the field
stays None and is omitted from saved output.

Same fixture pattern as test_evaluate_efficiency_tracking.py: "[]" short-
circuits every judge metric to 0.0 without a real LLM, and pipeline_module's
search()/generate() are patched at the module the evaluate_question()
imports from (called locally, so patching the attribute affects the import
at call time).
"""
from __future__ import annotations

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult


def _fake_chunks():
    return [SearchResult(text="The AMF handles registration.", distance=0.2,
                          metadata={"source": "23501-j00.md"}, doc_id="c1")]


def _open_question() -> EvalQuestion:
    return EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
    )


class _FakeEmbedder:
    def embed(self, texts, is_query=False):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _judge_call_returning_empty_json():
    def _call(prompt, max_tokens=1024):
        return "[]"
    return _call


def test_open_question_records_one_judge_call_per_metric(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model",
        record_judge_calls=True,
    )

    assert result.error == ""
    metrics_called = [c["metric"] for c in result.judge_calls]
    # reference_answer is set -> all four judge metrics run.
    assert metrics_called == ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]
    for call in result.judge_calls:
        assert call["prompt"]  # the real prompt text, not empty
        assert call["raw_response"] == "[]"  # the real raw response, verbatim


def test_mc_tf_questions_record_no_judge_calls(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )
    result = evaluate_question(
        q, store=object(), embedder=object(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model",
        record_judge_calls=True,
    )

    assert result.judge_calls == []  # recording on, but mc/tf never call a judge


def test_judge_call_recording_does_not_disturb_real_usage_tracking(monkeypatch):
    """The recording wrapper must delegate to the same underlying llm_call
    object -- _snapshot_llm_usage() reads usage off THAT object, not the
    wrapper, so judge_prompt_tokens/judge_completion_tokens must still work
    exactly as before this feature existed."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    def _tracked_call(prompt, max_tokens=1024):
        _tracked_call.total_prompt_tokens += 10
        _tracked_call.total_completion_tokens += 2
        return "[]"
    _tracked_call.total_prompt_tokens = 0
    _tracked_call.total_completion_tokens = 0

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_tracked_call,
        rag_generator_name="local", rag_model_name="local-model",
        record_judge_calls=True,
    )

    assert len(result.judge_calls) == 4
    assert result.judge_prompt_tokens == 40
    assert result.judge_completion_tokens == 8


def test_self_consistency_records_a_judge_call_per_generation(monkeypatch):
    """n_generations>1 for open questions averages judge scores across
    generations -- judge_calls must reflect every one of those calls, not
    just the first generation's."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    result = evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model",
        n_generations=3, generation_temperature=0.7, record_judge_calls=True,
    )

    assert result.self_consistency_n == 3
    # 4 metrics x 3 generations.
    assert len(result.judge_calls) == 12


# ── opt-in: off by default ────────────────────────────────────────────────────

def _run_open(monkeypatch, **kw):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw_: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw_: {
        "answer": "The AMF handles registration.", "sources": [], "chunks_used": 1})
    return evaluate_question(
        _open_question(), store=object(), embedder=_FakeEmbedder(),
        llm_call=_judge_call_returning_empty_json(),
        rag_generator_name="local", rag_model_name="local-model", **kw)


def test_recording_is_off_by_default_and_omitted_from_output(monkeypatch):
    from open_telco_rag.evaluation.results import question_to_dict

    result = _run_open(monkeypatch)
    assert result.judge_calls is None
    assert "judge_calls" not in question_to_dict(result)


def test_recording_does_not_change_any_score(monkeypatch):
    from open_telco_rag.evaluation.results import question_to_dict

    def scores(**kw):
        d = question_to_dict(_run_open(monkeypatch, **kw))
        d.pop("judge_calls", None)
        return {k: v for k, v in d.items() if not any(m in k for m in ("latency", "_ms"))}

    assert scores() == scores(record_judge_calls=True)


def test_recorded_calls_are_saved_when_enabled(monkeypatch):
    from open_telco_rag.evaluation.results import question_to_dict

    d = question_to_dict(_run_open(monkeypatch, record_judge_calls=True))
    assert next(c["metric"] for c in d["judge_calls"]) == "faithfulness"
