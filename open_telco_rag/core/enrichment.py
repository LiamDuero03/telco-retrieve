"""Text chunk enrichment strategies for telecom specifications."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any
from open_telco_rag.types import Chunk
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict

class EnrichmentStrategy(str, Enum):
    """Supported strategies for chunk metadata/content enrichment."""
    NONE = "none"
    LLM_METADATA = "llm_metadata"
    ACRONYM_EXPANSION = "acronym_expansion"
    METADATA_TAGGING = "metadata_tagging"

def enrich_llm_metadata(chunks: list[Chunk], llm_client: Any = None, **kwargs: Any) -> list[Chunk]:
    """Strategy 1: Parallelized metadata enrichment via thread execution."""
    if not llm_client:
        print("[Warning] No LLM client provided for LLM_METADATA enrichment. Skipping pass.")
        return chunks

    model_name = kwargs.get("model_name", "google/qwen-2.5-coder-32b")
    
    system_prompt = (
        "You are a telecom expert assistant. Analyze the provided technical text chunk. "
        "Identify key domain abbreviations, acronyms, or complex concepts present, and "
        "provide brief, highly accurate definitions or explanations for them. "
        "Return your analysis strictly as a flat JSON array of definition strings."
    )

    # Resolve model name once
    from open_telco_rag.generators import RECOMMENDED_MODELS
    resolved_model = RECOMMENDED_MODELS.get(model_name, model_name)

    def process_single_chunk(chunk: Chunk):
        if not chunk.text.strip():
            return chunk
        try:
            import requests
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {llm_client._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract and explain terms for this chunk:\n\n{chunk.text}"}
                    ],
                    "response_format": {"type": "json_object"}
                },
                timeout=60,
            )
            response.raise_for_status()
            raw_content = response.json()["choices"][0]["message"]["content"]
            
            explanations = json.loads(raw_content)
            if isinstance(explanations, dict):
                key = list(explanations.keys())[0]
                explanations = explanations[key]

            if explanations and isinstance(explanations, list):
                enrichment_prefix = "Technical Context & Definitions:\n" + "\n".join(f"- {e}" for e in explanations)
                chunk.text = f"{enrichment_prefix}\n\nOriginal Content:\n{chunk.text}"
        except Exception as e:
            print(f"[Warning] LLM Enrichment failed for chunk {chunk.chunk_index}: {e}")
        return chunk

    # Execute all requests concurrently using up to 20 parallel threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(process_single_chunk, chunks))

    return chunks


def load_glossary(file_path: str) -> Dict[str, str]:
    """Loads and returns the glossary dictionary from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Glossary file '{file_path}' not found. Using empty dictionary.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from '{file_path}': {e}")
        return {}


def enrich_acronym_expansion(chunks: List[Chunk], **kwargs: Any) -> List[Chunk]:
    """Appends acronym expansions utilizing strict word-boundary evaluations.

    Optimized for large glossaries by compiling a single regex pattern.
    """
    # Accept a massive parsed dictionary via kwargs, fallback to loading from file
    glossary_path = kwargs.get("glossary_path", "3gpp:glossary.json")
    acronym_dictionary = kwargs.get("glossary_dict") or load_glossary(glossary_path)

    if not acronym_dictionary:
        return chunks

    # Pre-compilation: Sort by length descending to match longest acronyms first
    # This prevents shorter acronyms (like "RAN") from prematurely matching inside "O-RAN"
    sorted_acronyms = sorted(acronym_dictionary.keys(), key=len, reverse=True)
    escaped_acronyms = [re.escape(a) for a in sorted_acronyms]

    # Create a single regex pattern to find ANY of the acronyms in O(1) passes per chunk
    master_pattern = re.compile(r"\b(" + "|".join(escaped_acronyms) + r")\b")

    for chunk in chunks:
        # Find all unique matches in the chunk text in one sweep
        matches = set(master_pattern.findall(chunk.text))

        if matches:
            # Map the found acronyms back to their definitions
            found_expansions = [
                f"{acronym}: {acronym_dictionary[acronym]}" for acronym in matches
            ]

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
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}

        header_parts = []
        
        # Pull any useful context keys dynamically passed down from the source parser
        for key, value in kwargs.items():
            if key in ["llm_client", "model_name"]:
                continue
            
            # 1. This writes directly to the Chroma/SQLite Relational Database Engine
            chunk.metadata[key] = value
            
            # 2. Format the data into a text string fragment
            clean_key = key.replace("_", " ").title()
            header_parts.append(f"{clean_key}: {value}")

        chunk.metadata["enriched"] = True

        # 3. Prepend the structured text block to the raw document content string
        if header_parts:
            metadata_header = " | ".join(header_parts)
            chunk.text = f"[Document Context -> {metadata_header}]\n\n{chunk.text}"

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