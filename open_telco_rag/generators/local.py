"""Local generator — any OpenAI-compatible server (vLLM, Ollama, llama.cpp
server, LM Studio, etc.) for RAG answer generation during evaluation.

Same Generator protocol as ClaudeGenerator/OpenRouterGenerator
(.generate(question, results, source_type) -> dict). This is distinct from
generators/local_llm.py's LocalLLMClient/OpenRouterLLMClient, which implement
a different interface (.completion.create) built for ingestion's
chunk_lumber_chunker/enrich_llm_metadata, not RAG evaluation.

Mirrors qna-generation's --generator local convention (base_url/api_key via
LOCAL_LLM_BASE_URL/LOCAL_LLM_API_KEY) so the same self-hosted server can
answer RAG evaluation questions without a cloud API key.
"""
from __future__ import annotations

import os

from open_telco_rag.generators._zero_retrieval import zero_retrieval_answer


class LocalGenerator:
    """Generate answers using any local OpenAI-compatible chat completions
    server.

    Args:
        model:       Model name your server is serving (no short-key resolution
                     like OpenRouterGenerator — a local server only serves one model).
        base_url:    Server URL, e.g. "http://localhost:8000/v1". Falls back to
                     LOCAL_LLM_BASE_URL env var, then to "http://localhost:8000/v1".
        api_key:     Bearer token for authenticated servers (e.g. vLLM --api-key).
                     Falls back to LOCAL_LLM_API_KEY, then no Authorization header.
        max_tokens:  Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        timeout:     Seconds to wait for a response.

    Example:
        generator = LocalGenerator(model="Qwen/Qwen3-32B", base_url="http://localhost:8001/v1")
        response  = generator.generate(question, results, source_type="oran")
        print(response["answer"])
    """

    def __init__(
        self,
        model:       str,
        base_url:    str | None = None,
        api_key:     str | None = None,
        max_tokens:  int   = 1000,
        temperature: float = 0.0,
        timeout:     int   = 60,
    ):
        self.model       = model
        self.base_url    = (base_url or os.environ.get(
            "LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"
        )).rstrip("/")
        self.api_key     = api_key or os.environ.get("LOCAL_LLM_API_KEY")
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

    def generate(
        self,
        question:    str,
        results:     list,
        source_type: str = "telecom",
        response_schema: dict | None = None,
    ) -> dict:
        """Generate a grounded answer from retrieved chunks.

        response_schema, when given, requests schema-constrained structured output
        (sent as response_format: json_schema, same field enrich_llm_metadata() uses)
        for reliable mc/tf scoring instead of parsing free text after the fact. Actual
        enforcement depends on the server having a guided-decoding backend configured
        (e.g. vLLM --guided-decoding-backend outlines/xgrammar) — not detected or
        required here; a non-conformant response just fails to parse as the expected
        shape and evaluate.py's _parse_structured_verdict() falls back to prose parsing.
        None (default) omits response_format entirely.

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
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # omitted when no key configured, so unauthenticated servers keep working
        # (matches _LocalCompletionAPI's header behavior in generators/local_llm.py)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content")
        # reasoning models can burn max_tokens on hidden reasoning and return content: null;
        # coerce to "" so .strip() doesn't crash and prose-fallback parsing handles it
        answer = (content or "").strip()

        return {
            "answer":           answer,
            "sources":          sources,
            "chunks_used":      len(results),
            "retrieved_chunks": [r.text for r in results],
            "model":            self.model,
            "usage":            data.get("usage", {}),
            "cost":             0.0,  # self-hosted — no per-token API cost
        }
