"""Text chunk enrichment strategies for telecom specifications."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from open_telco_rag.types import Chunk

class EnrichmentStrategy(str, Enum):
    """Supported strategies for chunk metadata/content enrichment."""
    NONE = "none"
    LLM_METADATA = "llm_metadata"
    ACRONYM_EXPANSION = "acronym_expansion"
    METADATA_TAGGING = "metadata_tagging"


def enrich_llm_metadata(chunks: list[Chunk], llm_client: Any = None, **kwargs: Any) -> list[Chunk]:
    """Strategy 1: Dynamically injects LLM-generated abbreviations and technical explanations 
    directly into the chunk body text to give the embedder more semantic context.
    """
    if not llm_client:
        print("[Warning] No LLM client provided for LLM_METADATA enrichment. Skipping pass.")
        return chunks

    model_name = kwargs.get("model_name", "google/qwen-2.5-coder-32b")
    
    system_prompt = (
        "You are a telecom expert assistant. Analyze the provided technical text chunk. "
        "Identify key domain abbreviations, acronyms, or complex concepts present, and "
        "provide brief, highly accurate definitions or explanations for them. "
        "Return your analysis strictly as a flat JSON array of definition strings. "
        "Example output: [\"gNB: Next Generation NodeB (5G Base Station)\", \"AMF: Core Access and Mobility Management Function\"]"
    )

    for chunk in chunks:
        if not chunk.text.strip():
            continue
        try:
            response = llm_client.completion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract and explain terms for this chunk:\n\n{chunk.text}"}
                ],
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            explanations = json.loads(raw_content)
            
            if isinstance(explanations, dict):
                key = list(explanations.keys())[0]
                explanations = explanations[key]

            if explanations and isinstance(explanations, list):
                # Prepend the semantic explanations right into the chunk text
                enrichment_prefix = "Technical Context & Definitions:\n" + "\n".join(f"- {e}" for e in explanations)
                chunk.text = f"{enrichment_prefix}\n\nOriginal Content:\n{chunk.text}"

        except Exception as e:
            print(f"[Warning] LLM Enrichment failed for chunk {chunk.chunk_index}: {e}")
            continue

    return chunks


def enrich_acronym_expansion(chunks: list[Chunk], **kwargs: Any) -> list[Chunk]:
    """Strategy 2: Scans chunk text against a fixed dictionary of telecom acronyms 
    and appends their explicit expansions to maximize token-matching accuracy.
    """
    # Sample dictionary (Expand this with your core thesis telecom acronyms)
    TELCO_ACRONYMS = {
        "RAN": "Radio Access Network",
        "RIC": "RAN Intelligent Controller",
        "O-RAN": "Open Radio Access Network",
        "3GPP": "3rd Generation Partnership Project",
        "CU": "Centralized Unit",
        "DU": "Distributed Unit",
        "RU": "Radio Unit",
    }

    for chunk in chunks:
        found_expansions = []
        # Basic boundary check to see if acronym matches a standalone word
        for acronym, expansion in TELCO_ACRONYMS.items():
            if repr(acronym)[1:-1] in chunk.text:  # Quick membership check
                found_expansions.append(f"{acronym} ({expansion})")

        if found_expansions:
            # Append dictionary definitions to the bottom of the text chunk
            expansion_suffix = "\n\n[Acronym Expansions]: " + ", ".join(found_expansions)
            chunk.text = f"{chunk.text}{expansion_suffix}"

    return chunks


def enrich_metadata_tagging(chunks: list[Chunk], **kwargs: Any) -> list[Chunk]:
    """Strategy 3: Structural metadata ingestion. Explicitly updates the chunk's 
    metadata payload dictionary with variables passed during extraction context.
    """
    # Grab structural parameters safely out of keyword arguments passed down by the source
    doc_name = kwargs.get("source_document", "unspecified_doc")
    release  = kwargs.get("release", "unknown_release")
    series   = kwargs.get("series", "unknown_series")

    for chunk in chunks:
        if not hasattr(chunk, 'metadata') or chunk.metadata is None:
            chunk.metadata = {}

        # Set specific structural tracking variables for downstream vector store storage
        chunk.metadata["document_name"] = doc_name
        chunk.metadata["release"]       = release
        chunk.metadata["series"]        = series
        chunk.metadata["enriched"]      = True

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