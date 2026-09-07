"""
Shared LLM helper for evaluation metrics: client creation and JSON parsing.

An llm_call is (prompt: str, max_tokens: int) -> str. `claude_call` is the
default; `make_llm_call(generator, model, ...)` builds one for claude,
openrouter, or local.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

_DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_OR_MODEL     = "meta-llama/llama-3.3-70b-instruct"
_OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"


class _TrackedLLMCall:
    """Callable that tracks token usage (and cost) per thread via
    threading.local(), so evaluate.py's before/after delta per question
    isn't corrupted by concurrently-running calls.

    fn does the request and returns (content, prompt_tokens,
    completion_tokens, cost); cost is a USD amount where the backend reports
    one, else None (tracked separately from an accumulated $0.00).
    """

    def __init__(self, fn: Callable[[str, int], tuple[str, int, int, float | None]]):
        self._fn = fn
        self._local = threading.local()

    def __call__(self, prompt: str, max_tokens: int = 1024) -> str:
        content, prompt_tokens, completion_tokens, cost = self._fn(prompt, max_tokens)
        self._local.prompt = getattr(self._local, "prompt", 0) + prompt_tokens
        self._local.completion = getattr(self._local, "completion", 0) + completion_tokens
        if cost is not None:
            self._local.cost = getattr(self._local, "cost", 0.0) + cost
            self._local.cost_known = True
        return content

    @property
    def total_prompt_tokens(self) -> int:
        return getattr(self._local, "prompt", 0)

    @property
    def total_completion_tokens(self) -> int:
        return getattr(self._local, "completion", 0)

    @property
    def total_cost(self) -> float | None:
        return getattr(self._local, "cost", 0.0) if getattr(self._local, "cost_known", False) else None


# ── Default Claude implementation ─────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _claude_request(prompt: str, max_tokens: int = 1024) -> tuple[str, int, int, float | None]:
    resp = _get_client().messages.create(
        model    = _DEFAULT_CLAUDE_MODEL,
        max_tokens = max_tokens,
        messages = [{"role": "user", "content": prompt}],
    )
    usage = getattr(resp, "usage", None)
    prompt_tokens = (getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
    completion_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
    # None, not 0.0: no Claude pricing table here.
    return resp.content[0].text.strip(), prompt_tokens, completion_tokens, None


# Default llm_call: Claude via Anthropic. Module-level singleton (used as a
# default param value, e.g. evaluate_question(llm_call=claude_call)).
claude_call = _TrackedLLMCall(_claude_request)


# ── Local implementation (any OpenAI-compatible server) ────────────────────────

def _make_local_call(
    model: str, base_url: str | None = None, api_key: str | None = None,
    timeout: int = 60,
) -> Callable[[str, int], str]:
    """llm_call against a local OpenAI-compatible server (vLLM, Ollama, ...).
    base_url/api_key fall back to LOCAL_LLM_* env vars; the Authorization
    header is omitted when no key is set. Uses a pooled requests.Session for
    concurrent use.
    """
    import requests
    from requests.adapters import HTTPAdapter

    resolved_base_url = (base_url or os.environ.get(
        "LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"
    )).rstrip("/")
    resolved_api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _request(prompt: str, max_tokens: int = 1024) -> tuple[str, int, int, float | None]:
        headers = {"Content-Type": "application/json"}
        if resolved_api_key:
            headers["Authorization"] = f"Bearer {resolved_api_key}"

        response = session.post(
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
        # Reasoning models can return content: null; coerce to "".
        usage = data.get("usage") or {}
        # Self-hosted: no $ cost, so None.
        return (content or "").strip(), usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0, None

    return _TrackedLLMCall(_request)


# ── OpenRouter implementation ─────────────────────────────────────────────────

# Per-model OpenRouter provider-routing overrides for models seen returning
# empty message.content on some backing providers. Scoped to models
# diagnosed against - don't widen without a live repro.
_PROVIDER_ROUTING: dict[str, dict] = {
    "qwen/qwen3.8-27b": {
        "order": ["chutes"],                   # confirmed clean
        "allow_fallbacks": True,               # don't hard-fail if Chutes is down
        "ignore": ["Io Net", "Cloudflare"],    # degraded on OpenRouter's listing
    },
    "z-ai/glm-5.3-flash": {
        # No clean provider confirmed yet; just exclude the two degraded ones.
        "ignore": ["Io Net", "Cloudflare"],
    },
}


def _post_with_retry(poster, url: str, *, headers: dict, json_payload: dict, timeout: int, max_retries: int = 5):
    """POST with retry on 429/5xx (transient OpenRouter failures under load).

    poster is a requests.Session or the bare `requests` module (both
    duck-type). Honours Retry-After, else exponential backoff with jitter
    capped at 30s; raises the final HTTPError once retries are exhausted.
    """
    import random
    import time as _time

    response = None
    for attempt in range(max_retries + 1):
        response = poster.post(url, headers=headers, json=json_payload, timeout=timeout)
        if response.status_code not in (429, 502, 503, 504):
            response.raise_for_status()
            return response
        if attempt == max_retries:
            break
        retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after is not None else 2.0 * (2 ** attempt)
        except ValueError:
            delay = 2.0 * (2 ** attempt)
        delay = min(delay, 30.0) + random.uniform(0, 1.0)
        print(f"    [retry] {response.status_code} from OpenRouter -- attempt {attempt + 1}/{max_retries}, "
              f"waiting {delay:.1f}s")
        _time.sleep(delay)

    response.raise_for_status()  # retries exhausted
    return response  # unreachable


def _make_openrouter_call(model: str) -> Callable[[str, int], str]:
    """llm_call using OpenRouter. Same Session-reuse and thread-local usage
    tracking as _make_local_call."""
    import requests
    from requests.adapters import HTTPAdapter

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
        )

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    provider_routing = _PROVIDER_ROUTING.get(model)

    def _request(prompt: str, max_tokens: int = 1024) -> tuple[str, int, int, float | None]:
        payload = {
            "model":       model,
            "max_tokens":  max_tokens,
            "temperature": 0.0,
            "messages":    [{"role": "user", "content": prompt}],
            # Ask OpenRouter to return the real billed cost in usage.cost.
            "usage": {"include": True},
        }
        if provider_routing:
            payload["provider"] = provider_routing
        response = _post_with_retry(
            session,
            f"{_OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/gsma-open-telco-ai/open-telco-rag",
                "X-Title":       "Open Telco RAG",
            },
            json_payload=payload,
            timeout=60,
        )
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        # Same content: null coercion as _make_local_call.
        usage = data.get("usage") or {}
        return (content or "").strip(), usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0, usage.get("cost")

    return _TrackedLLMCall(_request)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_llm_call(
    generator: str = "claude",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
) -> Callable[[str, int], str]:
    """
    Return an llm_call callable for use in evaluation metrics.

    Args:
        generator: "claude", "openrouter", or "local"
        model:     model identifier. Defaults to claude-sonnet-4-20250514
                   (Claude) or meta-llama/llama-3.3-70b-instruct (OpenRouter).
                   Required in practice for "local" - a local server only
                   ever serves one model, under whatever name it was
                   started with; there's no aliasing to fall back on.
        base_url:  only used for generator="local" - server URL, e.g.
                   "http://localhost:8000/v1". Falls back to
                   LOCAL_LLM_BASE_URL env var, then to
                   "http://localhost:8000/v1". Ignored otherwise.
        api_key:   only used for generator="local" - Bearer token for
                   servers that require authentication. Falls back to
                   LOCAL_LLM_API_KEY env var, then to no Authorization
                   header at all. Ignored otherwise.
        timeout:   only used for generator="local" (60s default). Per-request
                   read timeout - raise it for large-prompt strategies (e.g.
                   parent_child) under high concurrency.

    Returns:
        Callable[[str, int], str] - takes (prompt, max_tokens) returns str.

    Examples:
        llm_call = make_llm_call("claude")
        llm_call = make_llm_call("openrouter", "meta-llama/llama-3.3-70b-instruct")
        llm_call = make_llm_call("openrouter", "google/gemini-2.0-flash-001")
        llm_call = make_llm_call("local", "Qwen/Qwen3-32B", base_url="http://localhost:8001/v1")
        llm_call = make_llm_call("local", "google/gemma-4-31b-it", base_url="...", timeout=180)
    """
    if generator == "local":
        resolved_url = base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        print(f"  Eval LLM : Local ({resolved_url}) — model: {model or 'unset!'} (timeout={timeout}s)")
        return _make_local_call(model or "local-model", base_url=base_url, api_key=api_key, timeout=timeout)

    if generator == "openrouter":
        resolved = model or _DEFAULT_OR_MODEL
        print(f"  Eval LLM : OpenRouter ({resolved})")
        return _make_openrouter_call(resolved)

    # Claude - return a fresh callable with a potentially overridden model
    resolved = model or _DEFAULT_CLAUDE_MODEL
    print(f"  Eval LLM : Claude ({resolved})")

    if resolved == _DEFAULT_CLAUDE_MODEL:
        return claude_call   # reuse the default singleton-backed instance

    # Custom Claude model; same thread-local tracking.
    def _custom_claude_request(prompt: str, max_tokens: int = 1024) -> tuple[str, int, int, float | None]:
        resp = _get_client().messages.create(
            model      = resolved,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = (getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        return resp.content[0].text.strip(), prompt_tokens, completion_tokens, None

    return _TrackedLLMCall(_custom_claude_request)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json(raw: str) -> Any:
    """Parse a JSON array/object from an LLM response, tolerating markdown
    fences, a <think>...</think> block, and surrounding prose. An unclosed
    <think> tag (truncated mid-reasoning) raises loudly.
    """
    import re

    raw = raw.strip()

    if "<think>" in raw:
        close_idx = raw.find("</think>")
        if close_idx != -1:
            # Discard the reasoning entirely, keep whatever follows it.
            raw = raw[close_idx + len("</think>"):].strip()
        else:
            # Truncated mid-reasoning: fail loudly.
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

    # Fallback: grab from the first bracket/brace to the last matching one.
    match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not find valid JSON in LLM response: {raw[:200]!r}")