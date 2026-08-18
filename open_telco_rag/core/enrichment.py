"""Text chunk enrichment strategies for telecom specifications."""

from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from open_telco_rag.types import Chunk

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
    timeout: float,
    frequency_penalty: float,
) -> list[Chunk]:
    """Async worker for enrich_llm_metadata. Shares one httpx.AsyncClient
    across requests; concurrency capped by a semaphore, not a thread count.
    Each chunk's exceptions are caught individually — asyncio.gather's
    default would otherwise cancel every in-flight request when one chunk raised.

    base_url/api_key come from the caller's LLM client (LocalLLMClient or
    OpenRouterLLMClient, see local_llm.py), so this works against local
    vLLM the same way chunk_lumber_chunker does. api_key is optional; sent
    as a Bearer token only when present, so unauthenticated local servers
    keep working with no Authorization header.
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
                        # Not all servers honor this (needs a guided-decoding backend);
                        # a malformed response is caught below and just skips that chunk.
                        "response_format": {"type": "json_object"},
                        # Biggest lever on enrichment throughput, but too low truncates
                        # completions mid-string (broken JSON). The finish_reason=="length"
                        # check below turns that into a clear warning, not a silent parse failure.
                        "max_tokens": max_tokens,
                        # Without this, some chunks send the model into a degenerate
                        # repetition loop (BPE compresses repeated tokens efficiently, so a
                        # loop can emit 100k+ characters well inside a 3000-token budget --
                        # raising max_tokens then just lets the loop run longer, it doesn't
                        # fix the truncation). Standard OpenAI field, so it round-trips through
                        # both vLLM and OpenRouter unlike vLLM-only params (e.g. repetition_penalty).
                        "frequency_penalty": frequency_penalty,
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
                    # A model asked for "nothing notable here" sometimes returns {}
                    # instead of the requested [] -- keys()[0] on that raised an
                    # unhandled IndexError ("list index out of range"), masking a
                    # legitimate empty result as a parse failure. Treat it the same
                    # as an empty list: nothing to prepend, chunk left unenriched.
                    keys = list(explanations.keys())
                    explanations = explanations[keys[0]] if keys else []

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

    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*(process_single_chunk(client, c) for c in chunks))

    return chunks


def enrich_llm_metadata(
    chunks: list[Chunk], llm_client: Any = None, **kwargs: Any
) -> list[Chunk]:
    """Strategy 1: async, semaphore-limited concurrent metadata enrichment.

    llm_client is the same object type chunk_lumber_chunker takes
    (LocalLLMClient or OpenRouterLLMClient, see generators/local_llm.py),
    not a RAG-answer Generator — reads .base_url/.api_key directly rather
    than calling the synchronous .completion.create(), since this needs
    async concurrency across potentially hundreds of thousands of chunks.

    Bridges this module's synchronous enrich_* contract to an async impl
    via asyncio.run() — safe since callers are plain synchronous code with
    no event loop of their own. httpx.AsyncClient + gather replace the
    previous requests + ThreadPoolExecutor approach: coroutines waiting on
    I/O don't hold an OS thread each, so concurrency scales well past a
    thread pool's ceiling — capped by max_concurrency (kwarg
    "llm_concurrency", default 50) to avoid exceeding the API/server's rate limit.

    max_tokens (kwarg, default 1500): caps each completion's length. Left
    uncapped, throughput measured ~14x slower on real hardware (uncapped
    JSON completions have no incentive to stop early). A first cap of 400
    was too aggressive — completions truncated mid-string, which is a
    parse failure, not just slower (unparsable chunks are silently left
    unenriched). Observed truncation points vary by model/corpus though —
    1500 has since recurred often enough on some chunks to be worth raising
    further (the runtime warning below suggests doubling it); there's no
    single safe constant here, only "raise it if the warning keeps firing."
    A finish_reason=="length" response prints a warning naming the cap as
    the cause, instead of a generic parse error.

    timeout (kwarg, default 180s): per-request httpx timeout (connect+read+
    write+pool, since a single float applies to all four). At high
    max_concurrency against one vLLM instance, requests can queue behind
    each other for longer than a short timeout allows -- those show up as
    a bare "[Warning] LLM Enrichment failed for chunk N: ..." with no
    truncation warning, i.e. work the server already spent GPU time on,
    thrown away. Raise this before raising max_concurrency further.

    frequency_penalty (kwarg, default 0.4): passed straight to the API to
    discourage repeated tokens. A truncation whose JSONDecodeError reports
    a char offset far beyond what max_tokens should produce (e.g. 150k+
    chars from a 3000-token cap) is a repetition loop, not genuinely long
    output -- raising max_tokens for those just delays the same failure.
    """
    if not llm_client:
        print(
            "[Warning] No LLM client provided for LLM_METADATA enrichment. Skipping pass."
        )
        return chunks

    model_name = kwargs.get("model_name", "google/qwen-2.5-coder-32b")
    max_concurrency = kwargs.get("llm_concurrency", 50)
    max_tokens = kwargs.get("max_tokens", 1500)
    timeout = kwargs.get("timeout", 180)
    frequency_penalty = kwargs.get("frequency_penalty", 0.4)

    system_prompt = (
        "You are a telecom expert assistant. Analyze the provided technical text chunk. "
        "Identify key domain abbreviations, acronyms, or complex concepts present, and "
        "provide brief, highly accurate definitions or explanations for them. "
        "Return your analysis strictly as a flat JSON array of definition strings."
    )

    # OpenRouter model-name aliasing; no-op for local vLLM (returns model_name unchanged)
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
        chunks, resolved_model, base_url, api_key, system_prompt,
        max_concurrency, max_tokens, timeout, frequency_penalty,
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


# Process-local cache: glossary path -> (dict, compiled regex). Avoids
# rebuilding the alternation regex (expensive) on every file; same
# glossary_path/glossary_dict inputs and matching logic, purely internal.
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
    # Explicit glossary_dict is caller-owned/ad hoc — not cached by path
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
    # glossary_dict overrides; otherwise fall back to the bundled glossary file
    default_glossary_path = SCRIPT_DIR / "3gpp_glossary.json"
    glossary_path = kwargs.get("glossary_path", default_glossary_path)
    glossary_dict = kwargs.get("glossary_dict")

    acronym_dictionary, master_pattern = _get_compiled_glossary(glossary_path, glossary_dict)

    if not acronym_dictionary or master_pattern is None:
        return chunks

    for chunk in chunks:
        matches = set(master_pattern.findall(chunk.text))

        if matches:
            found_expansions = [
                f"{acronym}: {acronym_dictionary[acronym]}"
                for acronym in matches
                if acronym in acronym_dictionary
            ]

            if found_expansions:
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

        # Any extra kwargs are treated as document context to tag
        for key, value in kwargs.items():
            if key in ["llm_client", "model_name"]:
                continue

            chunk.metadata[key] = value

            clean_key = key.replace("_", " ").title()
            header_parts.append(f"{clean_key}: {value}")

        chunk.metadata["enriched"] = True

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