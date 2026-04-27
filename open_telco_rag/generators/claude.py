"""Claude generator — Anthropic API backend."""

from __future__ import annotations

import os
import time

import anthropic


class ClaudeGenerator:
    """Generate answers using Anthropic Claude.

    Args:
        model:       Claude model ID. Default: claude-sonnet-4-6.
                     Options: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        max_retries: Retries on API overload with exponential backoff.
        api_key:     Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Example:
        generator = ClaudeGenerator(model="claude-sonnet-4-6")
        response  = generator.generate(question, results, source_type="oran")
        print(response["answer"])
    """

    def __init__(
        self,
        model:       str   = "claude-sonnet-4-6",
        max_tokens:  int   = 1000,
        temperature: float = 0.0,
        max_retries: int   = 3,
        api_key:     str | None = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Pass api_key= or set the environment variable. "
                "Get a key at https://console.anthropic.com"
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
        if not results:
            return {
                "answer":           "No relevant chunks found.",
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }

        context = "\n\n---\n\n".join(r.text for r in results)
        sources = list({r.metadata.get("source", "") for r in results})

        prompt = (
            f"You are a {source_type} technical expert. "
            "Answer the question using only the context provided. "
            "If the answer is not in the context, say so clearly. "
            "Be precise and technical.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        client = anthropic.Anthropic(api_key=self._api_key)

        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(
                    model       = self.model,
                    max_tokens  = self.max_tokens,
                    temperature = self.temperature,
                    messages    = [{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if "overloaded" in str(e).lower() and attempt < self.max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  Claude overloaded — retrying in {wait}s "
                          f"({attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    raise

        return {
            "answer":           response.content[0].text,
            "sources":          sources,
            "chunks_used":      len(results),
            "retrieved_chunks": [r.text for r in results],
        }