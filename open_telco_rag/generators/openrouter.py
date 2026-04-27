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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Recommended models for telecom RAG generation
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
        site_url:    str   = "https://github.com/gsma-open-telco-ai/open-telco-rag",
        site_name:   str   = "Open Telco RAG",
    ):
        # Resolve short key to full model ID
        self.model       = RECOMMENDED_MODELS.get(model, model)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.site_url    = site_url
        self.site_name   = site_name
        self._api_key    = api_key or os.environ.get("OPENROUTER_API_KEY")

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
    ) -> dict:
        """Generate a grounded answer from retrieved chunks.

        Returns:
            dict with keys: answer, sources, chunks_used, retrieved_chunks
        """
        import requests

        if not results:
            return {
                "answer":           "No relevant chunks found.",
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

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization":  f"Bearer {self._api_key}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   self.site_url,
                "X-Title":        self.site_name,
            },
            json={
                "model":       self.model,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        answer = data["choices"][0]["message"]["content"].strip()

        return {
            "answer":           answer,
            "sources":          sources,
            "chunks_used":      len(results),
            "retrieved_chunks": [r.text for r in results],
            "model":            self.model,
            "usage":            data.get("usage", {}),
        }