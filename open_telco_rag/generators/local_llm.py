"""LLM clients for ingestion-time stages (chunk_lumber_chunker,
enrich_llm_metadata) - a different interface from the RAG-answer Generators.

    llm_client.completion.create(model=..., messages=[...])
    -> response.choices[0].message.content

.base_url/.api_key are also exposed at the top level for enrich_llm_metadata's
own async httpx requests.

    LocalLLMClient       - any OpenAI-compatible server, no API cost.
    OpenRouterLLMClient  - reuses OPENROUTER_API_KEY; validate without a GPU.

Both are stateless so they can be pickled into ProcessPoolExecutor workers.
"""
from __future__ import annotations

import os
import time
from typing import Any


# ── Response shape mimicking the OpenAI SDK ──

class _Message:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message",)

    def __init__(self, content: str):
        self.message = _Message(content)


class _CompletionResponse:
    __slots__ = ("choices",)

    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class LLMConnectionError(RuntimeError):
    """Raised when the LLM endpoint can't be reached or returns a bad response."""


# ── Local backend (vLLM / Ollama / any OpenAI-compatible server) ──────────────

class _LocalCompletionAPI:
    def __init__(self, base_url: str, timeout: int, max_retries: int, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

    def create(self, model: str, messages: list[dict], **kwargs: Any) -> _CompletionResponse:
        import requests

        payload = {
            "model": model,
            "messages": messages,
            # lumber_chunker only needs a short answer.
            "max_tokens": kwargs.get("max_tokens", 20),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # Omitted when no key is configured.
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        last_error: Exception | None = None
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _CompletionResponse(content)

            except requests.exceptions.ConnectionError as e:
                last_error = LLMConnectionError(
                    f"Could not reach local LLM server at {self.base_url}. "
                    f"Is it running? (e.g. `vllm serve {model} --port 8000`, "
                    f"or check `curl {self.base_url}/models`). Original error: {e}"
                )
            except requests.exceptions.Timeout as e:
                last_error = LLMConnectionError(
                    f"Local LLM server at {self.base_url} timed out after "
                    f"{self.timeout}s. It may still be loading the model, or "
                    f"be overloaded with concurrent requests. Original error: {e}"
                )
            except (KeyError, IndexError, ValueError) as e:
                last_error = LLMConnectionError(
                    f"Local LLM server at {self.base_url} returned an unexpected "
                    f"response shape (not OpenAI-compatible?). Original error: {e}"
                )
            except Exception as e:  # requests.HTTPError etc.
                last_error = LLMConnectionError(
                    f"Local LLM server at {self.base_url} returned an error: {e}"
                )

            if attempt < self.max_retries - 1:
                time.sleep(backoff)
                backoff *= 2

        assert last_error is not None
        raise last_error


class LocalLLMClient:
    """Talks to any local OpenAI-compatible chat completions server (vLLM,
    Ollama, llama.cpp, ...).

    base_url falls back to LOCAL_LLM_BASE_URL then a default; api_key falls
    back to LOCAL_LLM_API_KEY then no Authorization header.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: str | None = None,
    ):
        self.base_url = base_url or os.environ.get(
            "LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")
        self.completion = _LocalCompletionAPI(self.base_url, timeout, max_retries, api_key=self.api_key)


# ── OpenRouter backend (no local GPU required) ─────────────────────────────────

class _OpenRouterCompletionAPI:
    def __init__(self, api_key: str, timeout: int, max_retries: int):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def create(self, model: str, messages: list[dict], **kwargs: Any) -> _CompletionResponse:
        import requests
        from open_telco_rag.generators.openrouter import OPENROUTER_BASE_URL

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 20),
            "temperature": kwargs.get("temperature", 0.0),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    last_error = LLMConnectionError(
                        f"OpenRouter rate-limited the request (429). Retrying..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return _CompletionResponse(content)

            except (KeyError, IndexError, ValueError) as e:
                last_error = LLMConnectionError(
                    f"OpenRouter returned an unexpected response shape: {e}"
                )
            except Exception as e:
                last_error = LLMConnectionError(f"OpenRouter request failed: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(backoff)
                backoff *= 2

        assert last_error is not None
        raise last_error


class OpenRouterLLMClient:
    """Same interface as LocalLLMClient, backed by OpenRouter. api_key falls
    back to OPENROUTER_API_KEY."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Pass api_key= or set the "
                "environment variable. Get a free key at https://openrouter.ai/keys"
            )
        # Top-level so callers building their own requests treat both backends alike.
        from open_telco_rag.generators.openrouter import OPENROUTER_BASE_URL
        self.base_url = OPENROUTER_BASE_URL
        self.api_key = resolved_key
        self.completion = _OpenRouterCompletionAPI(resolved_key, timeout, max_retries)