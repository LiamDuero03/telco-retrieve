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


def _strip_lone_surrogates(text: str) -> str:
    """Replace unpaired UTF-16 surrogates with U+FFFD so the text encodes to
    UTF-8. A truncated LLM completion can leave one, which HF tokenizers reject."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


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
    """Async worker for enrich_llm_metadata: one shared httpx client,
    concurrency capped by a semaphore, per-chunk exceptions caught
    individually. base_url/api_key come from the caller's LLM client;
    api_key sent as a Bearer token only when present.
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
                        # Not all servers honor this; malformed responses skip the chunk.
                        "response_format": {"type": "json_object"},
                        # Too low truncates completions into broken JSON (warned below).
                        "max_tokens": max_tokens,
                        # Suppresses degenerate repetition loops; works on vLLM + OpenRouter.
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
                    # Model sometimes returns {} instead of []; treat as empty.
                    keys = list(explanations.keys())
                    explanations = explanations[keys[0]] if keys else []

                if explanations and isinstance(explanations, list):
                    enrichment_prefix = (
                        "Technical Context & Definitions:\n"
                        + "\n".join(f"- {e}" for e in explanations)
                    )
                    chunk.text = _strip_lone_surrogates(
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

    llm_client is a LocalLLMClient / OpenRouterLLMClient (see
    generators/local_llm.py); .base_url/.api_key are read directly.
    Bridges the synchronous enrich_* contract to an async impl via
    asyncio.run().

    kwargs: llm_concurrency (default 50), max_tokens (default 1500; raise if
    the truncation warning keeps firing), timeout (default 180s),
    frequency_penalty (default 0.4).
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

    # OpenRouter alias resolution; no-op for local vLLM.
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


# Process-local cache: glossary path -> (dict, compiled regex).
_GLOSSARY_CACHE: dict[str, tuple[dict[str, str], "re.Pattern | None"]] = {}


def _compile_glossary_pattern(acronym_dictionary: dict[str, str]) -> "re.Pattern | None":
    if not acronym_dictionary:
        return None
    # Sort by length descending to match longest acronyms first
    sorted_acronyms = sorted(acronym_dictionary.keys(), key=len, reverse=True)
    escaped_acronyms = [re.escape(a) for a in sorted_acronyms]
    # Word-boundary guards so "RAN" doesn't match inside "O-RAN".
    return re.compile(r"(?<![\w\-])(" + "|".join(escaped_acronyms) + r")(?![\w\-])")


def _get_compiled_glossary(
    glossary_path: str | Path, glossary_dict: dict[str, str] | None
) -> tuple[dict[str, str], "re.Pattern | None"]:
    # An explicit glossary_dict is not cached by path.
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
    """Append glossary acronym expansions, matched on strict word boundaries.
    The alternation regex is compiled once per glossary path and cached.
    """
    # glossary_dict overrides; else the bundled glossary file.
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
    """Strategy 3: write chunk metadata and inject a structural text header
    before embedding. Header values are deduplicated (3GPP metadata fields
    are often all "raw.md").
    """
    for chunk in chunks:
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}

        header_parts = []
        seen_values: set[str] = set()

        # Any extra kwargs are treated as document context to tag
        for key, value in kwargs.items():
            if key in ["llm_client", "model_name"]:
                continue

            chunk.metadata[key] = value  # full dict always recorded, even if deduped from the header

            value_str = str(value)
            if value_str in seen_values:
                continue
            seen_values.add(value_str)

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
    """Route chunks to the chosen enrichment strategy."""
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