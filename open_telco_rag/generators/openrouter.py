"""OpenRouter generator — access any LLM via OpenRouter API.

OpenRouter provides a unified OpenAI-compatible API for hundreds of models:
  - GPT-4o, GPT-4o-mini          (OpenAI)
  - Llama 3.1 8B, 70B, 405B     (Meta, via Together/Fireworks/Groq)
  - Mistral 7B, Mixtral 8x7B    (Mistral AI)
  - Gemini 1.5 Pro, Flash        (Google)
  - Command R+                   (Cohere)
  - Claude (via OpenRouter)      (Anthropic)
  - Any model at openrouter.ai/models

This is the recommended backend for:
  - Running open-source models without vendor lock-in
  - Cost-sensitive workloads (Llama 3.1 8B is ~50x cheaper than Claude Sonnet)
  - AMD GPU validation (KR2.2) — use Ollama locally with the same interface

Get a free API key at: https://openrouter.ai/keys
"""

from __future__ import annotations

import os

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

# per-1M-tokens (input, output) USD; 2026 baseline rates
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
    """Generate answers using any model via OpenRouter.

    Args:
        model:       Model ID or short key from RECOMMENDED_MODELS.
                     Default: meta-llama/llama-3.1-8b-instruct
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        api_key:     OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        site_url:    Optional — your site URL for OpenRouter attribution.
        site_name:   Optional — your app name for OpenRouter attribution.

    Example:
        # Using a short key
        generator = OpenRouterGenerator(model="llama-3.1-70b")

        # Using a full model ID
        generator = OpenRouterGenerator(model="meta-llama/llama-3.1-70b-instruct")

        response = generator.generate(question, results, source_type="oran")
        print(response["answer"])
    """

    def __init__(
        self,
        model:       str   = "meta-llama/llama-3.1-8b-instruct",
        max_tokens:  int   = 1000,
        temperature: float = 0.0,
        api_key:     str | None = None,
        site_url:    str   = "",
        site_name:   str   = "Open Telco RAG",
    ):
        self.model       = RECOMMENDED_MODELS.get(model, model)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.site_url    = site_url
        self.site_name   = site_name

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

        response_schema, when given, requests schema-constrained structured output
        (see LocalGenerator.generate()'s docstring for full rationale). Whether a
        specific model routed through OpenRouter actually enforces the schema varies
        per upstream provider — a non-conformant response just isn't valid JSON,
        which evaluate.py's _parse_structured_verdict() falls back to prose parsing for.

        Returns:
            dict with keys: answer, sources, chunks_used, retrieved_chunks
        """
        import requests

        if not results:
            # Structured abstain verdict when response_schema requests one
            # (mc/tf) so this scores "abstained", not "incorrect" -- see
            # generators/_zero_retrieval.py.
            return {
                "answer":           zero_retrieval_answer(response_schema),
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }

        context = "\n\n---\n\n".join(r.text for r in results)
        sources = list({r.metadata.get("source", "") for r in results})

        system_prompt = (
            f"You are a {source_type} technical expert. "
            "Answer questions using only the context provided. "
            "If the answer is not in the context, say so clearly. "
            "Be precise and technical."
        )

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
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": response_schema, "strict": True},
            }

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization":  f"Bearer {self._api_key}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   self.site_url,
                "X-Title":        self.site_name,
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"].get("content")
        # same None -> "" coercion as LocalGenerator.generate() -- reasoning models
        # can also return content: null on a thin token budget
        answer = (content or "").strip()

        self.last_usage = data.get("usage", {})
        prompt_tokens = self.last_usage.get("prompt_tokens", 0)
        completion_tokens = self.last_usage.get("completion_tokens", 0)

        # default to llama-3.1-70b rates if model is untracked
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