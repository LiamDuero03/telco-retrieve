"""
Shared LLM helper for evaluation metrics.

Centralises client creation and JSON response parsing.
Provides a default Claude implementation and a factory for
constructing any llm_call-compatible callable for use in metrics.

An llm_call is simply:  (prompt: str, max_tokens: int) -> str

Usage:
    # Default — Claude (used automatically if you pass nothing)
    from open_telco_rag.evaluation._claude import claude_call, parse_json

    # Custom — any model via factory
    from open_telco_rag.evaluation._claude import make_llm_call
    llm_call = make_llm_call("openrouter", model="meta-llama/llama-3.3-70b-instruct")
    llm_call = make_llm_call("claude",     model="claude-sonnet-4-20250514")
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

_DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_OR_MODEL     = "meta-llama/llama-3.3-70b-instruct"
_OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"

# ── Default Claude implementation ─────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def claude_call(prompt: str, max_tokens: int = 1024) -> str:
    """Default llm_call implementation — Claude Sonnet via Anthropic API."""
    resp = _get_client().messages.create(
        model    = _DEFAULT_CLAUDE_MODEL,
        max_tokens = max_tokens,
        messages = [{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── OpenRouter implementation ─────────────────────────────────────────────────

def _make_openrouter_call(model: str) -> Callable[[str, int], str]:
    """Return an llm_call using OpenRouter with the given model."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
        )

    def _call(prompt: str, max_tokens: int = 1024) -> str:
        response = requests.post(
            f"{_OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/gsma-open-telco-ai/open-telco-rag",
                "X-Title":       "Open Telco RAG",
            },
            json={
                "model":       model,
                "max_tokens":  max_tokens,
                "temperature": 0.0,
                "messages":    [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    return _call


# ── Factory ───────────────────────────────────────────────────────────────────

def make_llm_call(
    generator: str = "claude",
    model: str | None = None,
) -> Callable[[str, int], str]:
    """
    Return an llm_call callable for use in evaluation metrics.

    Args:
        generator: "claude" or "openrouter"
        model:     model identifier. Defaults to claude-sonnet-4-20250514
                   (Claude) or meta-llama/llama-3.3-70b-instruct (OpenRouter).

    Returns:
        Callable[[str, int], str] — takes (prompt, max_tokens) returns str.

    Examples:
        llm_call = make_llm_call("claude")
        llm_call = make_llm_call("openrouter", "meta-llama/llama-3.3-70b-instruct")
        llm_call = make_llm_call("openrouter", "google/gemini-2.0-flash-001")
    """
    if generator == "openrouter":
        resolved = model or _DEFAULT_OR_MODEL
        print(f"  Eval LLM : OpenRouter ({resolved})")
        return _make_openrouter_call(resolved)

    # Claude — return a fresh callable with a potentially overridden model
    resolved = model or _DEFAULT_CLAUDE_MODEL
    print(f"  Eval LLM : Claude ({resolved})")

    if resolved == _DEFAULT_CLAUDE_MODEL:
        return claude_call   # reuse the default singleton-backed function

    # Custom Claude model — create a dedicated closure
    def _custom_claude_call(prompt: str, max_tokens: int = 1024) -> str:
        resp = _get_client().messages.create(
            model      = resolved,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    return _custom_claude_call


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json(raw: str) -> Any:
    """Strip markdown fences if present and parse JSON."""
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(cleaned)