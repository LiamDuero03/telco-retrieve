"""Text chunking strategies."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any
import tiktoken

from open_telco_rag.types import Chunk


class ChunkingStrategy(str, Enum):
    """Supported strategies for document chunking."""
    MARKDOWN = "markdown"
    TEXT = "text"
    FIXED_TOKENS = "fixed_tokens"
    HIERARCHICAL = "hierarchical"
    LLM_AIDED = "llm_aided"
    INFORMATION_NUGGETS = "information_nuggets"


def _accumulate_chunks(
    paragraphs: list[str],
    chunk_size: int,
    heading: str | None = None,
    source_type: str = "",
) -> list[Chunk]:
    """Group paragraphs into chunks up to chunk_size characters."""
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(Chunk(
                text=current.strip(),
                heading=heading,
                chunk_index=idx,
                source_type=source_type
            ))
            idx += 1
            current = ""
        current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            heading=heading,
            chunk_index=idx,
            source_type=source_type
        ))

    return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = 4000,
    source_type: str = "",
) -> list[Chunk]:
    """Split markdown on heading boundaries then by paragraph."""
    if not text or not text.strip():
        return []

    sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    all_chunks: list[Chunk] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"^#{1,3}\s+(.+)$", section, re.MULTILINE)
        heading = heading_match.group(1).strip() if heading_match else None

        if len(section) <= chunk_size:
            all_chunks.append(Chunk(
                text=section,
                heading=heading,
                chunk_index=len(all_chunks),
                source_type=source_type
            ))
            continue

        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        all_chunks.extend(_accumulate_chunks(
            paragraphs, chunk_size, heading, source_type
        ))

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
    source_type: str = "",
) -> list[Chunk]:
    """Generic paragraph-based chunker with overlap for PDFs and plain text."""
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(Chunk(
                text=current.strip(),
                heading=None,
                chunk_index=idx,
                source_type=source_type
            ))
            idx += 1
            words = current.split()
            overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current
            current = overlap_text
        current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            heading=None,
            chunk_index=idx,
            source_type=source_type
        ))

    return chunks


def chunk_fixed_tokens(
    text: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
    source_type: str = "",
) -> list[Chunk]:
    """Strictly segments text by character-agnostic model token lengths."""
    if not text or not text.strip():
        return []

    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    
    chunks: list[Chunk] = []
    start_idx = 0
    chunk_index = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_tokens, len(tokens))
        chunk_token_slice = tokens[start_idx:end_idx]
        chunk_text_decoded = tokenizer.decode(chunk_token_slice)

        chunks.append(Chunk(
            text=chunk_text_decoded.strip(),
            heading=None,
            chunk_index=chunk_index,
            source_type=source_type
        ))
        
        chunk_index += 1
        start_idx += (chunk_tokens - overlap_tokens)
        
        if start_idx >= len(tokens) or (chunk_tokens <= overlap_tokens):
            break

    return chunks


def chunk_hierarchical(
    text: str,
    parent_size: int = 2048,
    child_size: int = 512,
    source_type: str = "",
) -> list[Chunk]:
    """Generates localized parent-child structures to preserve global technical context."""
    parent_chunks = chunk_markdown(text, chunk_size=parent_size, source_type=source_type)
    hierarchical_chunks: list[Chunk] = []
    
    global_child_idx = 0
    for p_chunk in parent_chunks:
        child_paragraphs = [p.strip() for p in p_chunk.text.split("\n\n") if p.strip()]
        current_child_text = ""
        parent_chunk_id = f"{source_type}_P_{p_chunk.chunk_index}"

        for para in child_paragraphs:
            if len(current_child_text) + len(para) + 2 > child_size and current_child_text:
                c_chunk = Chunk(
                    text=current_child_text.strip(),
                    heading=p_chunk.heading,
                    chunk_index=global_child_idx,
                    source_type=source_type
                )
                c_chunk.metadata = {
                    "parent_id": parent_chunk_id,
                    "parent_context": p_chunk.text[:200] + "..."
                }
                hierarchical_chunks.append(c_chunk)
                global_child_idx += 1
                current_child_text = ""
            current_child_text = f"{current_child_text}\n\n{para}" if current_child_text else para

        if current_child_text.strip():
            c_chunk = Chunk(
                text=current_child_text.strip(),
                heading=p_chunk.heading,
                chunk_index=global_child_idx,
                source_type=source_type
            )
            c_chunk.metadata = {
                "parent_id": parent_chunk_id,
                "parent_context": p_chunk.text[:200] + "..."
            }
            hierarchical_chunks.append(c_chunk)
            global_child_idx += 1

    return hierarchical_chunks


def chunk_llm_aided(
    text: str,
    llm_client: Any,
    model_name: str = "google/qwen-2.5-coder-32b",
    source_type: str = "",
) -> list[Chunk]:
    """Leverages LLM structural comprehension to break text on precise semantic shifts."""
    if not text or not text.strip():
        return []

    system_prompt = (
        "You are an advanced text analysis utility. Your task is to look over a document text "
        "and determine logical breaking points where a semantic shift or clear topical change occurs. "
        "Return the content exactly as formatted arrays inside a clean valid JSON list structure. "
        "Example output: [\"semantic chunk 1 text\", \"semantic chunk 2 text\"]"
    )

    try:
        response = llm_client.completion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze and split this document text semantically:\n\n{text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        extracted_strings = json.loads(raw_content)
        
        if isinstance(extracted_strings, dict):
            key = list(extracted_strings.keys())[0]
            extracted_strings = extracted_strings[key]

        return [
            Chunk(text=segment.strip(), heading=None, chunk_index=i, source_type=source_type)
            for i, segment in enumerate(extracted_strings) if segment.strip()
        ]
    except Exception as e:
        print(f"[Warning] LLM Chunker failed: {e}. Falling back to chunk_text baseline.")
        return chunk_text(text, chunk_size=1500, source_type=source_type)


def chunk_information_nuggets(
    text: str,
    llm_client: Any,
    model_name: str = "google/qwen-2.5-coder-32b",
    source_type: str = "",
) -> list[Chunk]:
    """Deserializes text into atomic, self-contained semantic propositions ("nuggets")."""
    if not text or not text.strip():
        return []

    system_prompt = (
        "You are a factual extraction engine. Dissect the target text into self-contained independent factual "
        "assertions ('Information Nuggets'). Each nugget must be a single standalone declarative sentence "
        "fully containing its context (substitute anaphoric reference pronouns with exact proper nouns). "
        "Return your analysis as a flat JSON array of clean strings."
    )

    try:
        response = llm_client.completion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract information nuggets from:\n\n{text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        nuggets = json.loads(raw_content)
        if isinstance(nuggets, dict):
            key = list(nuggets.keys())[0]
            nuggets = nuggets[key]

        chunks: list[Chunk] = []
        current_block = ""
        chunk_idx = 0
        
        for nugget in nuggets:
            if len(current_block) + len(nugget) + 2 > 1200 and current_block:
                chunks.append(Chunk(
                    text=current_block.strip(),
                    heading=None,
                    chunk_index=chunk_idx,
                    source_type=source_type
                ))
                chunk_idx += 1
                current_block = ""
            current_block = f"{current_block} {nugget}" if current_block else nugget

        if current_block.strip():
            chunks.append(Chunk(
                text=current_block.strip(),
                heading=None,
                chunk_index=chunk_idx,
                source_type=source_type
            ))

        return chunks
    except Exception as e:
        print(f"[Warning] Nugget Chunker failed: {e}. Falling back to chunk_text baseline.")
        return chunk_text(text, chunk_size=1000, source_type=source_type)


# --- Infrastructure Interface Entry Point ---

def chunk_document(
    text: str,
    strategy: ChunkingStrategy | str,
    **kwargs: Any,
) -> list[Chunk]:
    """Master orchestration function to route text data to selected chunking strategy.

    Args:
        text: Target document raw string.
        strategy: Chosen pipeline method (Enum or matching raw string name).
        **kwargs: Flexible keyword configuration configurations tailored to individual strategies 
                  (e.g., llm_client, chunk_size, parent_size).
    """
    # Map strategies cleanly to their underlying programmatic implementation
    _STRATEGY_MAP = {
        ChunkingStrategy.MARKDOWN: chunk_markdown,
        ChunkingStrategy.TEXT: chunk_text,
        ChunkingStrategy.FIXED_TOKENS: chunk_fixed_tokens,
        ChunkingStrategy.HIERARCHICAL: chunk_hierarchical,
        ChunkingStrategy.LLM_AIDED: chunk_llm_aided,
        ChunkingStrategy.INFORMATION_NUGGETS: chunk_information_nuggets,
    }

    # Normalize input string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = ChunkingStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Supported choices: {[s.value for s in ChunkingStrategy]}"
            )

    chunker_function = _STRATEGY_MAP[strategy]
    return chunker_function(text, **kwargs)
