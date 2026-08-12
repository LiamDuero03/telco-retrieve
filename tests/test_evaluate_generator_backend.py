"""Tests for open_telco_rag.evaluation.evaluate._build_generator() — a real,
confirmed bug: this function used to accept an llm_call callable and
silently ignore it entirely, ALWAYS constructing an OpenRouterGenerator
regardless of what --rag-generator actually asked for. --rag-generator
claude never worked: RAG answers were always generated via OpenRouter's
API, using whatever model string happened to be passed (which may not even
be OpenRouter-compatible).

Fixed by taking an explicit generator_name ("claude"/"openrouter"/"local")
instead of a discarded callable — these tests confirm each of the three
names actually constructs the matching Generator class, with the right
model/base_url/api_key threaded through.
"""
from __future__ import annotations

from open_telco_rag.evaluation.evaluate import _build_generator
from open_telco_rag.generators.claude import ClaudeGenerator
from open_telco_rag.generators.openrouter import OpenRouterGenerator
from open_telco_rag.generators.local import LocalGenerator


def test_claude_generator_name_builds_a_claude_generator(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    generator = _build_generator("claude", model_name="claude-sonnet-4-6")

    assert isinstance(generator, ClaudeGenerator)
    assert generator.model == "claude-sonnet-4-6"


def test_openrouter_generator_name_builds_an_openrouter_generator(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-test")

    generator = _build_generator("openrouter", model_name="meta-llama/llama-3.3-70b-instruct")

    assert isinstance(generator, OpenRouterGenerator)
    assert generator.model == "meta-llama/llama-3.3-70b-instruct"


def test_local_generator_name_builds_a_local_generator_not_openrouter():
    """The exact regression this fix targets: previously this would have
    silently returned an OpenRouterGenerator here too, requiring
    OPENROUTER_API_KEY even for a fully local, no-cloud-API-key run."""
    generator = _build_generator(
        "local", model_name="Qwen/Qwen3-32B",
        base_url="http://localhost:8001/v1", api_key="local-key",
    )

    assert isinstance(generator, LocalGenerator)
    assert not isinstance(generator, OpenRouterGenerator)
    assert generator.model == "Qwen/Qwen3-32B"
    assert generator.base_url == "http://localhost:8001/v1"
    assert generator.api_key == "local-key"


def test_local_generator_requires_no_cloud_api_key(monkeypatch):
    """Confirms a local run genuinely needs zero cloud credentials --
    OPENROUTER_API_KEY/ANTHROPIC_API_KEY deliberately left unset."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    generator = _build_generator("local", model_name="local-model", base_url="http://localhost:8000/v1")

    assert isinstance(generator, LocalGenerator)


def test_different_generator_names_produce_different_classes():
    """The core of the bug: three different generator_name values used to
    all collapse onto the same OpenRouterGenerator. Confirm they're
    genuinely distinct now."""
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-for-test")
    os.environ.setdefault("OPENROUTER_API_KEY", "fake-key-for-test")

    claude_gen     = _build_generator("claude")
    openrouter_gen = _build_generator("openrouter")
    local_gen      = _build_generator("local", model_name="local-model")

    classes = {type(claude_gen), type(openrouter_gen), type(local_gen)}
    assert len(classes) == 3, f"expected 3 distinct generator classes, got {classes}"
