"""OpenRouter generator - any LLM via the OpenRouter API.

Model ids are passed straight through, or use a short key from
RECOMMENDED_MODELS. Needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os

from open_telco_rag.generators._prompts import DEFAULT_PROMPT_MODE, resolve_system_prompt
from open_telco_rag.generators._zero_retrieval import zero_retrieval_answer

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RECOMMENDED_MODELS = {
    "llama-3.1-8b":   "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b":  "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "mistral-7b":     "mistralai/mistral-7b-instruct",
    "mixtral-8x7b":   "mistralai/mixtral-8x7b-instruct",
    "gpt-4o":         "openai/gpt-4o",
    "gpt-4o-mini":    "openai/gpt-4o-mini",
    "gemini-flash":   "google/gemini-flash-1.5",
    "gemini-pro":     "google/gemini-pro-1.5",
    "command-r-plus": "cohere/command-r-plus",
}

# per-1M-tokens (input, output) USD. Fallback only - OpenRouter's own
# returned cost is authoritative (see generate()).
MODEL_PRICING = {
    "meta-llama/llama-3.1-8b-instruct":   (0.055, 0.055),
    "meta-llama/llama-3.1-70b-instruct":  (0.35,  0.40),
    "meta-llama/llama-3.1-405b-instruct": (2.66,  2.66),
    "openai/gpt-4o":                      (2.50,  10.00),
    "openai/gpt-4o-mini":                 (0.150, 0.600),
    "google/gemini-flash-1.5":            (0.075, 0.300),
    "google/gemini-pro-1.5":              (1.25,  5.00),
}


class OpenRouterGenerator:
    """Generate answers via OpenRouter.

    model is a full id or a RECOMMENDED_MODELS short key. api_key falls back
    to OPENROUTER_API_KEY. site_url/site_name are optional attribution headers.
    """

    def __init__(
        self,
        model:       str   = "meta-llama/llama-3.1-8b-instruct",
        max_tokens:  int   = 1000,
        temperature: float = 0.0,
        top_p:       float | None = None,
        api_key:     str | None = None,
        site_url:    str   = "",
        site_name:   str   = "Open Telco RAG",
        timeout:     int   = 60,
        prompt_mode: str   = DEFAULT_PROMPT_MODE,
    ):
        self.model       = RECOMMENDED_MODELS.get(model, model)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.top_p       = top_p
        self._api_key    = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.site_url    = site_url
        self.site_name   = site_name
        self.timeout     = timeout
        self.prompt_mode = prompt_mode

        self.last_usage  = {}
        self.last_cost   = 0.0

        if not self._api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Pass api_key= or set the environment variable. "
                "Get a free key at https://openrouter.ai/keys"
            )

    def generate(
        self,
        question:    str,
        results:     list,
        source_type: str = "telecom",
        response_schema: dict | None = None,
    ) -> dict:
        """Generate a grounded answer from retrieved chunks.

        response_schema, when given, requests structured output; enforcement
        varies per upstream provider and a non-conformant response falls back
        to prose parsing.

        Returns a dict: answer, sources, chunks_used, retrieved_chunks.
        """
        import requests

        if not results:
            # Structured abstain verdict when response_schema requests one; see _zero_retrieval.py.
            return {
                "answer":           zero_retrieval_answer(response_schema),
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }

        context = "\n\n---\n\n".join(r.text for r in results)
        sources = list(dict.fromkeys(r.metadata.get("source", "") for r in results))

        system_prompt = resolve_system_prompt(source_type, self.prompt_mode)

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        payload = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            # Ask OpenRouter to return the real billed cost in usage.cost.
            "usage": {"include": True},
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": response_schema, "strict": True},
            }
        # Per-model provider-routing override; shared table with evaluation/_claude.py.
        from open_telco_rag.evaluation._claude import _PROVIDER_ROUTING, _post_with_retry
        provider_routing = _PROVIDER_ROUTING.get(self.model)
        if provider_routing:
            payload["provider"] = provider_routing

        # _post_with_retry: retry on 429/5xx. `requests` duck-types as a poster.
        response = _post_with_retry(
            requests,
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization":  f"Bearer {self._api_key}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   self.site_url,
                "X-Title":        self.site_name,
            },
            json_payload=payload,
            timeout=self.timeout,
        )
        data = response.json()

        content = data["choices"][0]["message"].get("content")
        # Same None -> "" coercion as LocalGenerator.
        answer = (content or "").strip()

        self.last_usage = data.get("usage", {})
        prompt_tokens = self.last_usage.get("prompt_tokens", 0)
        completion_tokens = self.last_usage.get("completion_tokens", 0)

        # Prefer OpenRouter's real billed cost (usage.include=True); table is a fallback.
        api_cost = self.last_usage.get("cost")
        if api_cost is not None:
            self.last_cost = api_cost
        else:
            price_in, price_out = MODEL_PRICING.get(self.model, (0.35, 0.40))
            self.last_cost = ((prompt_tokens / 1_000_000) * price_in) + ((completion_tokens / 1_000_000) * price_out)

        return {
            "answer":           answer,
            "sources":          sources,
            "chunks_used":      len(results),
            "retrieved_chunks": [r.text for r in results],
            "model":            self.model,
            "usage":            self.last_usage,
            "cost":             self.last_cost,
        }