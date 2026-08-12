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
        response_schema: dict | None = None,
    ) -> dict:
        """Generate a grounded answer from retrieved chunks.

        response_schema is accepted for interface parity with
        LocalGenerator/OpenRouterGenerator (see LocalGenerator.generate()'s
        own docstring for what it's for) but NOT wired up here — Claude's
        structured-output mechanism is tool-use-based, not the OpenAI-
        style response_format: json_schema LocalGenerator/OpenRouterGenerator
        both use, and RAG_GENERATOR="claude" isn't part of this session's
        actual grid config (RAG_GENERATOR="local" is). A caller that
        passes response_schema here still gets a normal free-text
        response — evaluate.py's _parse_structured_verdict() correctly
        treats that as "not structured" and falls back to prose parsing,
        so this is a silent no-op, not a broken code path, but it means
        this backend doesn't yet get the accuracy/reliability benefit of
        schema-constrained mc/tf generation. Wiring Claude tool-use here
        is a real follow-up if ClaudeGenerator is ever used as
        RAG_GENERATOR for the grid.

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

        # Anthropic SDK usage shape differs from the OpenAI-compatible
        # "usage" dict local.py/openrouter.py return (input_tokens/
        # output_tokens, not prompt_tokens/completion_tokens) -- normalized
        # to the same prompt_tokens/completion_tokens/total_tokens keys
        # here so evaluate.py's token-accounting block can read gen["usage"]
        # identically regardless of which generator produced it.
        usage = getattr(response, "usage", None)
        prompt_tokens     = getattr(usage, "input_tokens", 0) or 0 if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0

        return {
            "answer":           response.content[0].text,
            "sources":          sources,
            "chunks_used":      len(results),
            "retrieved_chunks": [r.text for r in results],
            "model":            self.model,
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      prompt_tokens + completion_tokens,
            },
            # No pricing table wired up for this backend yet (unlike
            # openrouter.py's MODEL_PRICING) -- None ("not computed"), not
            # 0.0, so this doesn't get misread as "genuinely free" the way
            # local.py's real self-hosted 0.0 is (see evaluate.py's
            # token-accounting block / QuestionResult.cost's own comment).
            "cost": None,
        }