"""Text chunk enrichment strategies for telecom specifications."""

from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from open_telco_rag.types import Chunk

# Define the directory of the current Python script
SCRIPT_DIR = Path(__file__).resolve().parent


class EnrichmentStrategy(str, Enum):
    """Supported strategies for chunk metadata/content enrichment."""

    NONE = "none"
    LLM_METADATA = "llm_metadata"
    ACRONYM_EXPANSION = "acronym_expansion"
    METADATA_TAGGING = "metadata_tagging"


async def _enrich_chunks_async(
    chunks: list[Chunk],
    resolved_model: str,
    base_url: str,
    api_key: str | None,
    system_prompt: str,
    max_concurrency: int,
    max_tokens: int,
) -> list[Chunk]:
    """Async worker for enrich_llm_metadata — one httpx.AsyncClient shared
    across every request (connection pooling/keep-alive), concurrency capped
    by a semaphore rather than a fixed thread count. A single failed chunk
    is caught and logged without cancelling the rest, matching the original
    ThreadPoolExecutor version's per-chunk fault isolation — asyncio.gather's
    default (return_exceptions=False) would otherwise cancel every other
    in-flight request the moment one chunk raised.

    base_url/api_key come from whichever LLM client enrich_llm_metadata was
    given (LocalLLMClient or OpenRouterLLMClient — see local_llm.py) rather
    than being hardcoded to OpenRouter, so this works against a local vLLM
    server exactly the same way chunk_lumber_chunker already does. api_key
    is optional and only sent as a Bearer token when present — unauthenticated
    local servers (the common case for a self-hosted vLLM instance) keep
    working with no Authorization header at all, matching
    _LocalCompletionAPI's own conditional-header behavior in local_llm.py.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def process_single_chunk(client: "httpx.AsyncClient", chunk: Chunk) -> Chunk:
        if not chunk.text.strip():
            return chunk
        async with semaphore:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json={
                        "model": resolved_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": f"Extract and explain terms for this chunk:\n\n{chunk.text}",
                            },
                        ],
                        # Supported by OpenRouter and by vLLM's OpenAI-compatible
                        # server (subject to that model/server having a guided-
                        # decoding backend configured — if not, some models may
                        # ignore this or return non-JSON text, which the
                        # json.loads() below already handles via the except
                        # clause: a malformed response just skips enrichment
                        # for that one chunk rather than failing the run).
                        "response_format": {"type": "json_object"},
                        # Previously uncapped -- the model had no incentive to
                        # stop early, and unlike chunking's calls (capped at
                        # 20 tokens, ~95-100 req/s observed), an unbounded
                        # multi-definition JSON completion can run for
                        # hundreds of tokens. Capping this is still the
                        # single largest lever on enrichment throughput, but
                        # cap it too low and completions get cut off
                        # mid-string -- syntactically broken JSON, not just
                        # slow. See the finish_reason=="length" check below,
                        # which turns that failure mode into a clear,
                        # diagnosable warning instead of a generic parse
                        # error, so a too-low cap is obvious from the logs
                        # rather than looking like a model/prompt problem.
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                response_data = response.json()
                choice = response_data["choices"][0]
                raw_content = choice["message"]["content"]

                if choice.get("finish_reason") == "length":
                    print(
                        f"[Warning] LLM Enrichment for chunk {chunk.chunk_index} was "
                        f"truncated at max_tokens={max_tokens} before finishing -- "
                        f"raise the max_tokens kwarg (e.g. to {max_tokens * 2}) if this "
                        f"recurs often; a truncated response can't be parsed as valid "
                        f"JSON, so this chunk is left unenriched."
                    )

                explanations = json.loads(raw_content)
                if isinstance(explanations, dict):
                    key = list(explanations.keys())[0]
                    explanations = explanations[key]

                if explanations and isinstance(explanations, list):
                    enrichment_prefix = (
                        "Technical Context & Definitions:\n"
                        + "\n".join(f"- {e}" for e in explanations)
                    )
                    chunk.text = (
                        f"{enrichment_prefix}\n\nOriginal Content:\n{chunk.text}"
                    )
            except Exception as e:
                print(
                    f"[Warning] LLM Enrichment failed for chunk {chunk.chunk_index}: {e}"
                )
        return chunk

    import httpx

    async with httpx.AsyncClient(timeout=60) as client:
        await asyncio.gather(*(process_single_chunk(client, c) for c in chunks))

    return chunks


def enrich_llm_metadata(
    chunks: list[Chunk], llm_client: Any = None, **kwargs: Any
) -> list[Chunk]:
    """Strategy 1: Async, semaphore-limited concurrent metadata enrichment.

    llm_client is the SAME kind of object chunk_lumber_chunker takes —
    LocalLLMClient (any OpenAI-compatible server, e.g. vLLM) or
    OpenRouterLLMClient (see open_telco_rag/generators/local_llm.py) — not
    a RAG-answer Generator. This function reads .base_url/.api_key directly
    off it rather than calling .completion.create() (which is synchronous),
    since it needs async concurrency across potentially hundreds of
    thousands of chunks; both client classes expose those two attributes at
    the top level for exactly this purpose.

    Bridges this module's synchronous enrich_chunks()/enrich_* contract to
    an async implementation (asyncio.run) — callers (_process_file_worker,
    ingest_multi_enrichment's main-process loop) are all plain synchronous
    code with no event loop of their own, so this is safe. httpx.AsyncClient
    + asyncio.gather replace the previous requests + ThreadPoolExecutor
    approach: coroutines waiting on network I/O don't hold an OS thread
    each, so concurrency can scale well past a thread pool's practical
    ceiling — capped by max_concurrency (kwarg "llm_concurrency", default
    50) rather than a fixed worker count, to avoid hammering the API/server
    past its actual rate limit or vLLM's configured max_num_seqs.

    max_tokens (kwarg, default 1024): caps each completion's generation
    length. Previously unset entirely, which measured at ~14x slower
    throughput than chunking's capped calls on real hardware (MI350X,
    Qwen3-30B-A3B) — an uncapped multi-definition JSON completion has no
    incentive to stop early. First attempt at a cap (400) was too
    aggressive: empirically it truncated a large fraction of real
    completions mid-string on that same hardware/model (observed
    truncation points up to ~550 tokens in), which isn't just slower, it's
    a parse failure — a chunk whose response got cut off can't be parsed
    as JSON at all, so it's silently left unenriched. 1024 leaves
    comfortable headroom above what was actually observed. A
    finish_reason=="length" response (see below) prints a specific warning
    naming the cap itself as the cause, rather than a generic parse error,
    so a still-too-low cap is diagnosable from the logs directly rather
    than by guessing.
    """
    if not llm_client:
        print(
            "[Warning] No LLM client provided for LLM_METADATA enrichment. Skipping pass."
        )
        return chunks

    model_name = kwargs.get("model_name", "google/qwen-2.5-coder-32b")
    max_concurrency = kwargs.get("llm_concurrency", 50)
    max_tokens = kwargs.get("max_tokens", 1500)

    system_prompt = (
        "You are a telecom expert assistant. Analyze the provided technical text chunk. "
        "Identify key domain abbreviations, acronyms, or complex concepts present, and "
        "provide brief, highly accurate definitions or explanations for them. "
        "Return your analysis strictly as a flat JSON array of definition strings."
    )

    # Resolve model name once. Only meaningful for OpenRouter's model-name
    # aliasing — a local vLLM server is serving exactly one model under
    # whatever name it was started with, so this is a harmless no-op (a
    # miss on RECOMMENDED_MODELS just returns model_name unchanged) rather
    # than something that needs to be skipped for the local backend.
    from open_telco_rag.generators import RECOMMENDED_MODELS

    resolved_model = RECOMMENDED_MODELS.get(model_name, model_name)

    base_url = getattr(llm_client, "base_url", None)
    if not base_url:
        print(
            f"[Warning] LLM client {type(llm_client).__name__} has no base_url "
            f"attribute — cannot run LLM_METADATA enrichment. Expected a "
            f"LocalLLMClient or OpenRouterLLMClient (see "
            f"open_telco_rag.generators.local_llm). Skipping pass."
        )
        return chunks
    api_key = getattr(llm_client, "api_key", None)

    return asyncio.run(_enrich_chunks_async(
        chunks, resolved_model, base_url, api_key, system_prompt, max_concurrency, max_tokens,
    ))


def load_glossary(file_path: str | Path) -> dict[str, str]:
    """Loads and returns the glossary dictionary from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: Glossary file '{file_path}' not found. Using empty dictionary."
        )
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from '{file_path}': {e}")
        return {}


# Process-local cache: glossary path -> (dictionary, compiled regex).
# enrich_acronym_expansion used to reload the JSON file AND rebuild/recompile
# the master alternation regex (the more expensive part, for a glossary this
# size) on every single file. That's the same file every time within a run,
# so we build it once per process and reuse it. Doesn't change behavior or
# the function's inputs/outputs — same glossary_path/glossary_dict kwargs,
# same matching logic — purely an internal cache.
_GLOSSARY_CACHE: dict[str, tuple[dict[str, str], "re.Pattern | None"]] = {}


def _compile_glossary_pattern(acronym_dictionary: dict[str, str]) -> "re.Pattern | None":
    if not acronym_dictionary:
        return None
    # Sort by length descending to match longest acronyms first
    sorted_acronyms = sorted(acronym_dictionary.keys(), key=len, reverse=True)
    escaped_acronyms = [re.escape(a) for a in sorted_acronyms]
    # Lookbehinds/Lookaheads to prevent false matches on hyphenated terms (e.g., RAN inside O-RAN)
    return re.compile(r"(?<![\w\-])(" + "|".join(escaped_acronyms) + r")(?![\w\-])")


def _get_compiled_glossary(
    glossary_path: str | Path, glossary_dict: dict[str, str] | None
) -> tuple[dict[str, str], "re.Pattern | None"]:
    # An explicitly-passed dict is treated as ad hoc (caller-owned, may differ
    # call to call) and isn't cached by path — build it fresh, same as before.
    if glossary_dict:
        return glossary_dict, _compile_glossary_pattern(glossary_dict)

    key = str(glossary_path)
    if key not in _GLOSSARY_CACHE:
        acronym_dictionary = load_glossary(glossary_path)
        _GLOSSARY_CACHE[key] = (acronym_dictionary, _compile_glossary_pattern(acronym_dictionary))
    return _GLOSSARY_CACHE[key]


def enrich_acronym_expansion(
    chunks: list[Chunk], **kwargs: Any
) -> list[Chunk]:
    """Appends acronym expansions utilizing strict word-boundary evaluations.

    Optimized for large glossaries by compiling a single regex pattern once
    per glossary path and reusing it across calls (see _get_compiled_glossary).
    """
    # Accept a massive parsed dictionary via kwargs, fallback to file relative to script
    default_glossary_path = SCRIPT_DIR / "3gpp_glossary.json"
    glossary_path = kwargs.get("glossary_path", default_glossary_path)
    glossary_dict = kwargs.get("glossary_dict")

    acronym_dictionary, master_pattern = _get_compiled_glossary(glossary_path, glossary_dict)

    if not acronym_dictionary or master_pattern is None:
        return chunks

    for chunk in chunks:
        # Find all unique matches in the chunk text in one sweep
        matches = set(master_pattern.findall(chunk.text))

        if matches:
            # Map the found acronyms back to their definitions
            found_expansions = [
                f"{acronym}: {acronym_dictionary[acronym]}"
                for acronym in matches
                if acronym in acronym_dictionary
            ]

            if found_expansions:
                # Append the structured suffix to the chunk payload
                suffix_string = "\n\n[Acronym Expansions]:\n* " + "\n* ".join(
                    sorted(found_expansions)
                )
                chunk.text += suffix_string

            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["acronyms_expanded"] = list(matches)

    return chunks


def enrich_metadata_tagging(chunks: list[Chunk], **kwargs: Any) -> list[Chunk]:
    """Strategy 3: Structural metadata ingestion. Updates the chunk's
    database metadata dictionary (SQL) AND injects a structural text header
    directly into the text tokens before embedding.
    """
    for chunk in chunks:
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}

        header_parts = []

        # Pull any useful context keys dynamically passed down from the source parser
        for key, value in kwargs.items():
            if key in ["llm_client", "model_name"]:
                continue

            # 1. Write directly to metadata
            chunk.metadata[key] = value

            # 2. Format the data into a text string fragment
            clean_key = key.replace("_", " ").title()
            header_parts.append(f"{clean_key}: {value}")

        chunk.metadata["enriched"] = True

        # 3. Prepend the structured text block to raw document content string
        if header_parts:
            metadata_header = " | ".join(header_parts)
            chunk.text = (
                f"[Document Context -> {metadata_header}]\n\n{chunk.text}"
            )

    return chunks


def enrich_chunks(
    chunks: list[Chunk],
    strategy: EnrichmentStrategy | str,
    **kwargs: Any,
) -> list[Chunk]:
    """Master orchestration function to route chunks to the chosen enrichment strategy."""
    _STRATEGY_MAP = {
        EnrichmentStrategy.NONE: lambda c, **kw: c,
        EnrichmentStrategy.LLM_METADATA: enrich_llm_metadata,
        EnrichmentStrategy.ACRONYM_EXPANSION: enrich_acronym_expansion,
        EnrichmentStrategy.METADATA_TAGGING: enrich_metadata_tagging,
    }

    if isinstance(strategy, str):
        try:
            strategy = EnrichmentStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Unknown enrichment strategy '{strategy}'. "
                f"Supported choices: {[s.value for s in EnrichmentStrategy]}"
            )

    enricher_function = _STRATEGY_MAP[strategy]
    return enricher_function(chunks, **kwargs)