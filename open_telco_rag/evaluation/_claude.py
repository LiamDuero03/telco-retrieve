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

    Tracks cumulative usage as PLAIN ATTRIBUTES on this function object
    (total_prompt_tokens/total_completion_tokens, set right below this
    def) rather than changing the return type to something richer than
    `str` — every caller in metrics/generation.py and metrics/ground_truth.py
    calls llm_call(prompt, max_tokens) expecting a bare string back, and
    changing that signature everywhere would be invasive for no benefit.
    evaluate_question() instead takes before/after snapshots of these
    attributes (see its own judge-usage comment) to attribute a batch of
    judge calls' token usage to one question, without this function's call
    sites ever needing to change. See _make_local_call/_make_openrouter_call
    below and make_llm_call()'s _custom_claude_call for the same pattern.
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
        # Reasoning models (e.g. openai/gpt-oss-120b) can spend their
        # entire max_tokens budget on hidden reasoning tokens and return a
        # well-formed response with "content": null when nothing is left
        # over for visible text -- a valid HTTP 200, so raise_for_status()
        # doesn't catch it. Coerce to "" instead of crashing on
        # `.strip()`; a genuinely-thin token budget is the caller's
        # problem to notice from an empty string, not this function's to
        # explode over.
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
        # Same defensive coercion as _make_local_call's _call() above --
        # OpenRouter can also route to reasoning models that return
        # "content": null on a too-small max_tokens budget.
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
    tolerating markdown code fences, a <think>...</think> reasoning block,
    AND leading/trailing prose around the JSON — e.g. a reasoning-style
    model's chain-of-thought preamble before its final answer (confirmed
    in production: openai/gpt-oss-120b, used as both the eval-metrics
    judge and RAG generator, produced exactly this pattern —
    faithfulness()/answer_correctness() scored normally while
    answer_relevancy()/context_recall() came back exactly 0.0 for every
    question, the literal fallback both return on any parse failure here).

    The previous implementation only handled a response that literally
    STARTED with a code fence, via str.lstrip("```json") — which has two
    real problems: (1) str.lstrip()'s argument is a CHARACTER SET, not a
    literal prefix, so it strips any leading run of the individual
    characters `,j,s,o,n rather than the literal 7-character fence
    (works by accident when the fence is followed by a character outside
    that set, silently wrong otherwise); (2) it does nothing at all for
    text BEFORE a fence or before bare JSON — the moment a model prefaces
    its answer with any explanatory text at all, json.loads() fails
    immediately on that prose, none of these metric prompts request
    response_format=json_object (unlike enrich_llm_metadata's ingestion-
    side enrichment call, which does), so nothing prevents a model from
    doing exactly that.

    <think>...</think> is a SEPARATE failure mode from gpt-oss-120b's
    plain-prose preamble, confirmed in production with Qwen/Qwen3-32B as
    the judge: unlike gpt-oss-120b (served with reasoning content kept out
    of the "content" field entirely, or the field coming back null on a
    thin budget — see _make_local_call's own None -> "" coercion), Qwen3
    commonly emits its reasoning INLINE in "content" as literal
    <think>...</think> text when the server has no reasoning-parser
    configured to split it out server-side. The generic "greedily grab the
    first bracket through the last bracket" fallback below is NOT safe
    against this: every one of these prompts explicitly asks the model to
    "return a JSON array/object", so the model's own reasoning very often
    talks about that literal format (e.g. "I need to return something like
    [...]") — the <think> block ends up containing its OWN incidental
    brackets/braces, and greedy .*/DOTALL then spans from the FIRST
    bracket INSIDE the reasoning through the LAST bracket at the very end
    (the real JSON's own closing bracket), producing a mashed-together
    blob that raises JSONDecodeError("Extra data: ...") instead of ever
    reaching the actual JSON. Reproduced directly: a <think> block reading
    "I need to return a JSON array like [\"q1\", \"q2\"]. The answer talks
    about {AMF} registration." immediately corrupts the fallback below
    this way, even though the real JSON immediately follows it cleanly.
    Stripping the whole <think>...</think> block FIRST removes those
    incidental brackets from consideration entirely.
    """
    import re

    raw = raw.strip()

    if "<think>" in raw:
        close_idx = raw.find("</think>")
        if close_idx != -1:
            # Discard the reasoning entirely, keep whatever follows it.
            raw = raw[close_idx + len("</think>"):].strip()
        else:
            # <think> opened but never closed -- the response was
            # truncated mid-reasoning (max_tokens exhausted before the
            # model ever got to emit any JSON at all). This is NOT a
            # parsing problem to work around here -- falling through to
            # the generic "could not find JSON" a few lines below would
            # bury the actual cause. Fail loudly and specifically instead,
            # so it's obvious from the exception message alone that the
            # fix is a bigger max_tokens budget for this call, not a
            # parser change.
            raise ValueError(
                "LLM response was truncated mid-reasoning (<think> block never "
                "closed) -- max_tokens is too small for this model to finish "
                f"reasoning and emit JSON. Raw (first 200 chars): {raw[:200]!r}"
            )

    # Strip a fenced code block wherever it appears (not just a literal
    # prefix) — most direct case: ```json ... ``` or ``` ... ``` with
    # nothing meaningful outside the fence.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: no fence, but there may still be prose around the JSON
    # (a reasoning model's preamble, or trailing commentary) — greedily
    # grab the first '[' or '{' through the last matching ']' or '}' and
    # parse just that substring.
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not find valid JSON in LLM response: {raw[:200]!r}")