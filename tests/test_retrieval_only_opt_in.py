"""retrieval_only is OPT-IN: off by default, unchanged output when unset."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli.main as cli_main
import open_telco_rag.core.pipeline as pipeline_module
import open_telco_rag.evaluation.evaluate as evaluate_module
from open_telco_rag.evaluation.evaluate import evaluate_question
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.evaluation.results import question_to_dict
from open_telco_rag.types import SearchResult


def _question():
    return EvalQuestion(question="Which interface?", keywords=[], corpus="3gpp", answer_type="mc",
                        options={"A": "N1", "B": "E2"}, reference_answer="B")


def _patch_pipeline(monkeypatch, calls):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: [
        SearchResult(text="alpha", distance=0.1, metadata={}, doc_id="c1")])

    def generate(*a, **kw):
        calls.append("generate")
        return {"answer": "B", "sources": [], "chunks_used": 1}

    monkeypatch.setattr(pipeline_module, "generate", generate)


def _volatile(d):
    return {k: v for k, v in d.items() if not any(m in k for m in ("latency", "_ms"))}


def test_default_and_explicit_false_are_identical(monkeypatch):
    _patch_pipeline(monkeypatch, [])
    kw = dict(store=object(), embedder=object(), rag_generator_name="local", rag_model_name="m")
    omitted = question_to_dict(evaluate_question(_question(), **kw))
    explicit = question_to_dict(evaluate_question(_question(), retrieval_only=False, **kw))
    assert _volatile(omitted) == _volatile(explicit)
    assert omitted["accuracy"] == 1.0  # generation ran


def test_retrieval_only_never_calls_the_generator_or_judge(monkeypatch):
    calls = []
    _patch_pipeline(monkeypatch, calls)

    def judge(prompt, max_tokens):
        calls.append("judge")
        return ""

    result = evaluate_question(_question(), store=object(), embedder=object(), llm_call=judge,
                               retrieval_only=True, rag_generator_name="local", rag_model_name="m")
    assert calls == [] and result.error == "" and result.answer == ""
    assert result.chunks_used == 1 and result.used_chunk_ids == ["c1"]


def test_retrieval_only_with_baseline_is_rejected():
    result = None
    with pytest.raises(ValueError, match="mutually exclusive"):
        result = evaluate_question(_question(), store=object(), embedder=object(),
                                   baseline=True, retrieval_only=True)
    assert result is None


# ── CLI ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def cli_evaluate(monkeypatch, tmp_path):
    seen = {"corpus_kwargs": None, "leaderboard": 0, "llm_built": 0}

    def fake_corpus(**kw):
        seen["corpus_kwargs"] = kw
        return SimpleNamespace(top_k=5, question_results=[]), None

    def build_llm(*a, **k):
        seen["llm_built"] += 1
        return lambda prompt, max_tokens: ""

    import open_telco_rag.evaluation._claude as claude_module

    monkeypatch.setattr(evaluate_module, "evaluate_corpus", fake_corpus)
    monkeypatch.setattr(claude_module, "make_llm_call", build_llm)
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(count=lambda: 3))
    monkeypatch.setattr(cli_main, "_update_leaderboard",
                        lambda *a, **k: seen.__setitem__("leaderboard", seen["leaderboard"] + 1))
    return seen, tmp_path


def _run(args, tmp_path):
    cli_main.main(["evaluate", "3gpp", "--no-save", "--leaderboard", str(tmp_path / "lb.json")] + args)


def test_cli_without_flag_is_unchanged(cli_evaluate):
    seen, tmp = cli_evaluate
    _run([], tmp)
    assert "retrieval_only" not in seen["corpus_kwargs"]
    assert seen["llm_built"] == 2 and seen["leaderboard"] == 1


def test_cli_retrieval_only_builds_no_llm_and_skips_the_leaderboard(cli_evaluate):
    seen, tmp = cli_evaluate
    _run(["--retrieval-only"], tmp)
    assert seen["corpus_kwargs"]["retrieval_only"] is True
    assert seen["llm_built"] == 0 and seen["leaderboard"] == 0
    with pytest.raises(RuntimeError, match="retrieval-only"):
        seen["corpus_kwargs"]["llm_call"]("p", 1)


def test_cli_retrieval_only_with_baseline_still_builds_llms(cli_evaluate):
    seen, tmp = cli_evaluate
    _run(["--retrieval-only", "--include-baseline"], tmp)
    assert seen["llm_built"] == 2
