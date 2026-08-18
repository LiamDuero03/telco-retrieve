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
    """Default llm_call implementation — Claude Sonnet via Anthropic API.

    Tracks cumulative usage as plain attributes on this function object
    (total_prompt_tokens/total_completion_tokens) rather than widening the
    return type — every caller expects a bare string back. evaluate_question()
    takes before/after snapshots of these attributes to attribute a batch of
    judge calls to one question. See _make_local_call/_make_openrouter_call
    and make_llm_call()'s _custom_claude_call for the same pattern.
    """
    resp = _get_client().messages.create(
        model    = _DEFAULT_CLAUDE_MODEL,
        max_tokens = max_tokens,
        messages = [{"role": "user", "content": prompt}],
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        claude_call.total_prompt_tokens     += getattr(usage, "input_tokens", 0) or 0
        claude_call.total_completion_tokens += getattr(usage, "output_tokens", 0) or 0
    return resp.content[0].text.strip()


claude_call.total_prompt_tokens = 0
claude_call.total_completion_tokens = 0


# ── Local implementation (any OpenAI-compatible server) ────────────────────────

def _make_local_call(
    model: str, base_url: str | None = None, api_key: str | None = None,
    timeout: int = 60,
) -> Callable[[str, int], str]:
    """Return an llm_call using a local OpenAI-compatible server (vLLM,
    Ollama, llama.cpp, etc.) — same convention as generators/local_llm.py
    and generators/local.py: base_url/api_key fall back to
    LOCAL_LLM_BASE_URL/LOCAL_LLM_API_KEY env vars, and the Authorization
    header is omitted entirely when no key is configured so unauthenticated
    local servers keep working unchanged.
    """
    import requests

    resolved_base_url = (base_url or os.environ.get(
        "LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"
    )).rstrip("/")
    resolved_api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")

    def _call(prompt: str, max_tokens: int = 1024) -> str:
        headers = {"Content-Type": "application/json"}
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        response = requests.post(
            f"{resolved_base_url}/chat/completions",
            headers=headers,
            json={
                "model":       model,
                "max_tokens":  max_tokens,
                "temperature": 0.0,
                "messages":    [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        # Reasoning models can spend their whole max_tokens budget on hidden
        # reasoning and return a well-formed "content": null (valid HTTP 200).
        # Coerce to "" instead of crashing on .strip().
        usage = data.get("usage") or {}
        _call.total_prompt_tokens     += usage.get("prompt_tokens", 0) or 0
        _call.total_completion_tokens += usage.get("completion_tokens", 0) or 0
        return (content or "").strip()

    _call.total_prompt_tokens = 0
    _call.total_completion_tokens = 0
    return _call


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
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        # Same defensive coercion as _make_local_call — OpenRouter can also
        # route to reasoning models returning "content": null.
        usage = data.get("usage") or {}
        _call.total_prompt_tokens     += usage.get("prompt_tokens", 0) or 0
        _call.total_completion_tokens += usage.get("completion_tokens", 0) or 0
        return (content or "").strip()

    _call.total_prompt_tokens = 0
    _call.total_completion_tokens = 0
    return _call


# ── Factory ───────────────────────────────────────────────────────────────────

def make_llm_call(
    generator: str = "claude",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Callable[[str, int], str]:
    """
    Return an llm_call callable for use in evaluation metrics.

    Args:
        generator: "claude", "openrouter", or "local"
        model:     model identifier. Defaults to claude-sonnet-4-20250514
                   (Claude) or meta-llama/llama-3.3-70b-instruct (OpenRouter).
                   Required in practice for "local" — a local server only
                   ever serves one model, under whatever name it was
                   started with; there's no aliasing to fall back on.
        base_url:  only used for generator="local" — server URL, e.g.
                   "http://localhost:8000/v1". Falls back to
                   LOCAL_LLM_BASE_URL env var, then to
                   "http://localhost:8000/v1". Ignored otherwise.
        api_key:   only used for generator="local" — Bearer token for
                   servers that require authentication. Falls back to
                   LOCAL_LLM_API_KEY env var, then to no Authorization
                   header at all. Ignored otherwise.

    Returns:
        Callable[[str, int], str] — takes (prompt, max_tokens) returns str.

    Examples:
        llm_call = make_llm_call("claude")
        llm_call = make_llm_call("openrouter", "meta-llama/llama-3.3-70b-instruct")
        llm_call = make_llm_call("openrouter", "google/gemini-2.0-flash-001")
        llm_call = make_llm_call("local", "Qwen/Qwen3-32B", base_url="http://localhost:8001/v1")
    """
    if generator == "local":
        resolved_url = base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        print(f"  Eval LLM : Local ({resolved_url}) — model: {model or 'unset!'}")
        return _make_local_call(model or "local-model", base_url=base_url, api_key=api_key)

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
        usage = getattr(resp, "usage", None)
        if usage is not None:
            _custom_claude_call.total_prompt_tokens     += getattr(usage, "input_tokens", 0) or 0
            _custom_claude_call.total_completion_tokens += getattr(usage, "output_tokens", 0) or 0
        return resp.content[0].text.strip()

    _custom_claude_call.total_prompt_tokens = 0
    _custom_claude_call.total_completion_tokens = 0
    return _custom_claude_call


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json(raw: str) -> Any:
    """Extract and parse a JSON array/object from an LLM response,
    tolerating markdown fences, a <think>...</think> block, and
    leading/trailing prose around the JSON.

    Two production bugs motivate this: (1) gpt-oss-120b prefaces JSON with
    prose that the old str.lstrip("```json")-based parser couldn't
    handle — lstrip() strips a character set, not a literal prefix, and
    does nothing for text before the fence, so json.loads() broke
    silently (answer_relevancy/context_recall fell back to 0.0 for every
    question while the other metrics scored fine). (2) Qwen3-32B emits
    raw <think>...</think> reasoning inline in "content" when the server
    has no reasoning-parser configured — since these prompts ask for a
    JSON array, the reasoning often mentions its own brackets/braces,
    which corrupts the greedy bracket-matching fallback below unless the
    <think> block is stripped first.

    An unclosed <think> tag means the response was truncated mid-reasoning
    (max_tokens exhausted) — raised loudly here rather than falling
    through to a generic "no JSON found" error, so the fix (a bigger
    max_tokens) is obvious from the exception alone.
    """
    import re

    raw = raw.strip()

    if "<think>" in raw:
        close_idx = raw.find("</think>")
        if close_idx != -1:
            # Discard the reasoning entirely, keep whatever follows it.
            raw = raw[close_idx + len("</think>"):].strip()
        else:
            # Truncated mid-reasoning: fail loudly here rather than falling
            # through to the generic "could not find JSON" error below.
            raise ValueError(
                "LLM response was truncated mid-reasoning (<think> block never "
                "closed) -- max_tokens is too small for this model to finish "
                f"reasoning and emit JSON. Raw (first 200 chars): {raw[:200]!r}"
            )

    # Strip a fenced code block wherever it appears, not just a literal prefix.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: no fence, but prose may still surround the JSON — greedily
    # grab the first '[' or '{' through the last matching ']' or '}'.
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not find valid JSON in LLM response: {raw[:200]!r}")