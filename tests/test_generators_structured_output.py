"""Tests for LocalGenerator/OpenRouterGenerator's response_schema wiring.

Part of the mc/tf root-cause fix: instead of inferring a verdict from
free-form prose (three regex bugs found in one session), request
schema-constrained output via response_format: json_schema — the same field
enrich_llm_metadata() already uses successfully, just with a real schema.
"""
from __future__ import annotations

from open_telco_rag.evaluation.evaluate import _match_deterministic_answer
from open_telco_rag.generators.claude import ClaudeGenerator
from open_telco_rag.generators.local import LocalGenerator
from open_telco_rag.generators.openrouter import OpenRouterGenerator
from open_telco_rag.types import SearchResult


def _fake_chunks():
    return [SearchResult(text="The AMF handles registration.", distance=0.2,
                          metadata={"source": "23501-j00.md"}, doc_id="c1")]


_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["answered", "abstained"]},
        "option": {"type": ["string", "null"], "enum": ["A", "B", None]},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "option", "justification"],
    "additionalProperties": False,
}


class _FakeResponse:
    def __init__(self, content, status=200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": {}}


# ── LocalGenerator ───────────────────────────────────────────────────────

def test_local_generator_includes_response_format_json_schema_when_schema_given(monkeypatch):
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json=json)
        return _FakeResponse('{"verdict": "answered", "option": "B", "justification": "x"}')

    monkeypatch.setattr(requests, "post", _fake_post)

    gen = LocalGenerator(model="test-model", base_url="http://localhost:8000/v1")
    result = gen.generate("Which interface?", _fake_chunks(), response_schema=_SCHEMA)

    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["schema"] == _SCHEMA
    assert result["answer"] == '{"verdict": "answered", "option": "B", "justification": "x"}'


def test_local_generator_omits_response_format_when_no_schema(monkeypatch):
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json=json)
        return _FakeResponse("The AMF handles registration and mobility management.")

    monkeypatch.setattr(requests, "post", _fake_post)

    gen = LocalGenerator(model="test-model", base_url="http://localhost:8000/v1")
    gen.generate("What does the AMF do?", _fake_chunks())

    assert "response_format" not in captured["json"]


def test_local_generator_coerces_null_content_to_empty_string_even_with_schema(monkeypatch):
    """Same reasoning-model-exhausts-budget failure mode as _make_local_call's fix;
    must not crash on None just because a schema was requested."""
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(None))

    gen = LocalGenerator(model="test-model", base_url="http://localhost:8000/v1")
    result = gen.generate("Which interface?", _fake_chunks(), response_schema=_SCHEMA)

    assert result["answer"] == ""


# ── OpenRouterGenerator ──────────────────────────────────────────────────

def test_openrouter_generator_includes_response_format_json_schema_when_schema_given(monkeypatch):
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json=json)
        return _FakeResponse('{"verdict": "abstained", "option": null, "justification": "no info"}')

    monkeypatch.setattr(requests, "post", _fake_post)

    gen = OpenRouterGenerator(model="meta-llama/llama-3.1-8b-instruct", api_key="fake-key")
    result = gen.generate("Which interface?", _fake_chunks(), response_schema=_SCHEMA)

    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert captured["json"]["response_format"]["json_schema"]["schema"] == _SCHEMA
    assert "abstained" in result["answer"]


def test_openrouter_generator_omits_response_format_when_no_schema(monkeypatch):
    import requests
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update(json=json)
        return _FakeResponse("The AMF handles registration.")

    monkeypatch.setattr(requests, "post", _fake_post)

    gen = OpenRouterGenerator(model="meta-llama/llama-3.1-8b-instruct", api_key="fake-key")
    gen.generate("What does the AMF do?", _fake_chunks())

    assert "response_format" not in captured["json"]


def test_openrouter_generator_coerces_null_content_to_empty_string(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResponse(None))

    gen = OpenRouterGenerator(model="meta-llama/llama-3.1-8b-instruct", api_key="fake-key")
    result = gen.generate("Which interface?", _fake_chunks(), response_schema=_SCHEMA)

    assert result["answer"] == ""


# ── Zero-retrieval mc/tf must score "abstained", not "incorrect" ───────────
#
# Regression: all three generators used to return the plain text "No
# relevant chunks found." on zero retrieval -- not valid structured-verdict
# JSON, and not recognized by the prose fallback's not-found phrase list
# either, so it silently scored "incorrect" instead of "abstained". None of
# these tests need to mock the HTTP/SDK call at all: zero retrieval is an
# early return before any request is made.

def test_local_generator_zero_retrieval_scores_abstained_not_incorrect():
    gen = LocalGenerator(model="test-model", base_url="http://localhost:8000/v1")
    result = gen.generate("Which interface?", [], response_schema=_SCHEMA)

    assert _match_deterministic_answer(result["answer"], "B", "mc") == "abstained"


def test_openrouter_generator_zero_retrieval_scores_abstained_not_incorrect():
    gen = OpenRouterGenerator(model="meta-llama/llama-3.1-8b-instruct", api_key="fake-key")
    result = gen.generate("Which interface?", [], response_schema=_SCHEMA)

    assert _match_deterministic_answer(result["answer"], "B", "mc") == "abstained"


def test_claude_generator_zero_retrieval_scores_abstained_not_incorrect():
    """ClaudeGenerator never emits structured output (see its own
    docstring), so this exercises the prose-fallback path specifically --
    confirms the zero-retrieval text itself is recognized as not-found."""
    gen = ClaudeGenerator(api_key="fake-key")
    result = gen.generate("Which interface?", [], response_schema=_SCHEMA)

    assert _match_deterministic_answer(result["answer"], "B", "mc") == "abstained"


def test_claude_generator_zero_retrieval_tf_also_scores_abstained():
    gen = ClaudeGenerator(api_key="fake-key")
    result = gen.generate("Is this true?", [])

    assert _match_deterministic_answer(result["answer"], "True", "tf") == "abstained"
