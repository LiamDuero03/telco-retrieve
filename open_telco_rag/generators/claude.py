"""Claude generator - Anthropic API backend."""

from __future__ import annotations

import os
import time

import anthropic

from open_telco_rag.generators._prompts import DEFAULT_PROMPT_MODE, resolve_system_prompt
from open_telco_rag.generators._zero_retrieval import zero_retrieval_answer


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
        prompt_mode: str   = DEFAULT_PROMPT_MODE,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.prompt_mode = prompt_mode

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

        response_schema is accepted for interface parity but NOT wired up -
        Claude's structured output is tool-use-based. Falls back to free text.
        TODO: wire Claude tool-use if this backend is ever used as RAG_GENERATOR.

        Returns a dict: answer, sources, chunks_used, retrieved_chunks.
        """
        if not results:
            # Always prose here (Claude never emits structured output); see _zero_retrieval.py.
            return {
                "answer":           zero_retrieval_answer(None),
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }

        context = "\n\n---\n\n".join(r.text for r in results)
        sources = list({r.metadata.get("source", "") for r in results})

        prompt = (
            f"{resolve_system_prompt(source_type, self.prompt_mode)}\n\n"
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

        # Anthropic SDK uses input/output_tokens; normalise to prompt/completion.
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
            # None, not 0.0: no pricing table here yet.
            "cost": None,
        }