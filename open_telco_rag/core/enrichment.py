"""Text chunk enrichment strategies for telecom specifications."""

from __future__ import annotations

import json
import re
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

def enrich_acronym_expansion(chunks: list[Chunk], **kwargs) -> list[Chunk]:
    """
    Appends acronym expansions utilizing strict word-boundary evaluations.
    """
    acronym_dictionary = {
        "RAN": "Radio Access Network",
        "RIC": "RAN Intelligent Controller",
        "O-RAN": "Open Radio Access Network",
        "3GPP": "3rd Generation Partnership Project",
        "SMO": "Service Management and Orchestration"
    }
    
    for chunk in chunks:
        found_expansions = []
        for acronym, expansion in acronym_dictionary.items():
            # Strict word boundary regex pattern check
            pattern = r"\b" + re.escape(acronym) + r"\b"
            if re.search(pattern, chunk.text):
                found_expansions.append(f"{acronym}: {expansion}")
                
        if found_expansions:
            suffix_string = "\n\n[Acronym Expansions]:\n* " + "\n* ".join(found_expansions)
            chunk.text += suffix_string
            
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