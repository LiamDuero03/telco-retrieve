"""Local generator — any OpenAI-compatible server (vLLM, Ollama, llama.cpp
server, LM Studio, etc.) for RAG answer generation during evaluation.

Same Generator protocol as ClaudeGenerator/OpenRouterGenerator
(.generate(question, results, source_type) -> dict) — this is the missing
third backend for evaluate: generators/local_llm.py's LocalLLMClient/
OpenRouterLLMClient already exist, but they implement a *different*
interface (.completion.create(model, messages) -> response, built for
ingestion's chunk_lumber_chunker/enrich_llm_metadata), not this one.

Mirrors the qna-generation side's --generator local convention (base_url/
api_key, falling back to LOCAL_LLM_BASE_URL/LOCAL_LLM_API_KEY env vars) so
the same self-hosted server used there can also answer RAG evaluation
questions, without needing a cloud API key at all — the point of this
class existing.
"""
from __future__ import annotations

import os


class LocalGenerator:
    """Generate answers using any local OpenAI-compatible chat completions
    server.

    Args:
        model:       Model name your server is serving (no short-key
                     resolution the way OpenRouterGenerator/
                     RECOMMENDED_MODELS does — a local server only ever
                     serves one model under whatever name it was started
                     with).
        base_url:    Server URL, e.g. "http://localhost:8000/v1". Falls
                     back to LOCAL_LLM_BASE_URL env var, then to
                     "http://localhost:8000/v1".
        api_key:     Bearer token for servers that require authentication
                     (e.g. vLLM started with --api-key). Falls back to the
                     LOCAL_LLM_API_KEY env var, then to no Authorization
                     header at all — unauthenticated local servers keep
                     working unchanged.
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

        response_schema, when given, is a JSON Schema object requesting
        SCHEMA-CONSTRAINED structured output — root-cause fix for mc/tf
        scoring instead of parsing free text after the fact (see
        evaluation/evaluate.py's _mc_response_schema()/_tf_response_
        schema()/_match_deterministic_answer()). Sent as response_format:
        {"type": "json_schema", "json_schema": {...}} — the SAME
        response_format field core/enrichment.py's enrich_llm_metadata()
        already sends successfully to both OpenRouter and this project's
        vLLM server (there with the generic {"type": "json_object"} mode,
        confirmed working against real infra this session), just with an
        actual schema instead of "any valid JSON" this time. Whether the
        server actually ENFORCES the schema (vs. silently ignoring an
        unrecognized field and returning free text, or honoring
        response_format generically but not the enum/schema constraint)
        depends on the serving stack having a guided-decoding backend
        configured (e.g. vLLM with --guided-decoding-backend
        outlines/xgrammar, or a recent enough vLLM version with xgrammar
        as the default) — this function doesn't detect or require that;
        a non-conformant response just isn't valid JSON matching the
        expected shape, which _parse_structured_verdict() on the
        evaluate.py side treats as a signal to fall back to prose
        parsing, not an error here. None (the default, unchanged from
        before) omits response_format entirely — every non-mc/tf caller
        is unaffected.

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
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Most local servers (vLLM in particular, started with --api-key)
        # require this — omitted entirely when no key is configured, so
        # unauthenticated local servers keep working unchanged. Matches
        # _LocalCompletionAPI's own conditional-header behavior in
        # generators/local_llm.py.
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
        # Same None -> "" coercion as _make_local_call in evaluation/
        # _claude.py -- a reasoning model can spend its whole max_tokens
        # budget on hidden reasoning and return "content": null. An empty
        # string here isn't valid structured-verdict JSON either, so it
        # correctly falls through to the prose fallback (which then also
        # finds nothing to match and returns "incorrect"/"abstained" as
        # appropriate) rather than crashing on .strip() against None.
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
