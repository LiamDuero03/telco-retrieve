"""Text chunking strategies aligned with thesis benchmark requirements."""

from __future__ import annotations

import concurrent.futures
import json
import re
from enum import Enum
from typing import Any
import tiktoken

from open_telco_rag.types import Chunk


class ChunkingStrategy(str, Enum):
    """Supported strategies for document chunking."""
    HIERARCHICAL_MARKDOWN = "hierarchical_markdown"  # Markdown heading boundaries (###)
    SLIDING_WINDOW_TOKENS = "sliding_window_tokens"  # Fixed token size with sliding window % overlap
    PARENT_CHILD = "parent_child"                    # Two-tier context tracking structures
    LUMBER_CHUNKER = "lumber_chunker"                # LLM-aided boundary mapping
    TEXT_BASELINE = "text_baseline"                  # Plain paragraph baseline fall-back


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


def chunk_hierarchical_markdown(
    text: str,
    chunk_size: int = 4000,
    source_type: str = "",
    **kwargs: Any,
) -> list[Chunk]:
    """Strategy 1: splits on Markdown headings (# to #####), tracking a
    breadcrumb trail across nested sub-sections.
    """
    if not text or not text.strip():
        return []

    # Capturing group keeps heading lines in the split stream
    heading_pattern = re.compile(r"(^#{1,5}\s+.*$)", re.MULTILINE)
    parts = heading_pattern.split(text)

    chunks: list[Chunk] = []
    current_headers = {1: None, 2: None, 3: None, 4: None, 5: None}
    
    current_text_block = ""
    active_heading_attr = "None"
    global_idx = 0

    for part in parts:
        if not part.strip():
            continue

        header_match = re.match(r"^(#{1,5})\s+(.*)$", part.strip())
        if header_match:
            # Must flush under the PREVIOUS heading before updating active_heading_attr,
            # or this flush mislabels content with the heading about to start.
            if current_text_block.strip():
                chunks.append(Chunk(
                    text=current_text_block.strip(),
                    heading=active_heading_attr,
                    chunk_index=global_idx,
                    source_type=source_type
                ))
                global_idx += 1
                current_text_block = ""

            level = len(header_match.group(1))
            heading_text = header_match.group(2).strip()

            # Clear deeper sub-levels so the breadcrumb reflects current hierarchy
            current_headers[level] = heading_text
            for l in range(level + 1, 6):
                current_headers[l] = None

            active_breadcrumbs = [current_headers[i] for i in range(1, 6) if current_headers[i]]
            active_heading_attr = " > ".join(active_breadcrumbs) if active_breadcrumbs else "None"

            # Keep header line inline in the chunk text, not stripped out
            current_text_block += part + "\n"
        else:
            if len(current_text_block) + len(part) > chunk_size and current_text_block.strip():
                chunks.append(Chunk(
                    text=current_text_block.strip(),
                    heading=active_heading_attr,
                    chunk_index=global_idx,
                    source_type=source_type
                ))
                global_idx += 1
                current_text_block = part + "\n"
            else:
                current_text_block += part + "\n"
                
    if current_text_block.strip():
        chunks.append(Chunk(
            text=current_text_block.strip(),
            heading=active_heading_attr,
            chunk_index=global_idx,
            source_type=source_type
        ))
        
    return chunks


def chunk_text_baseline(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
    source_type: str = "",
    **kwargs: Any,
) -> list[Chunk]:
    """Baseline paragraph-based chunker with character overlap, snapped
    backward to the nearest sentence boundary.
    """
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

            if len(current) > overlap:
                lookback_window = current[-overlap:]
                # Find sentence-ending punctuation within the overlap window
                split_match = list(re.finditer(r"[\.\!\?\n]\s*", lookback_window))
                if split_match:
                    # Start overlap right after the last punctuation match
                    snap_point = split_match[-1].end()
                    current = lookback_window[snap_point:]
                else:
                    current = lookback_window
            else:
                current = current
            
        current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            heading=None,
            chunk_index=idx,
            source_type=source_type
        ))

    return chunks

def chunk_sliding_window_tokens(
    text: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
    source_type: str = "",
) -> list[Chunk]:
    """Strategy 2: fixed-size token windows with sliding overlap."""
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


def chunk_parent_child(
    text: str,
    parent_size: int = 2048,
    child_size: int = 512,
    source_type: str = "",
    **kwargs: Any,
) -> list[Chunk]:
    """Strategy 3: two-tier retrieval — large parent segments broken down
    into small, targeted child vectors.
    """
    parent_chunks = chunk_hierarchical_markdown(text, chunk_size=parent_size, source_type=source_type)
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
                    "parent_context": p_chunk.text[:200] + "...",
                    "full_parent_text": p_chunk.text  # Cached for two-tier parent retrieval routines
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
                "parent_context": p_chunk.text[:200] + "...",
                "full_parent_text": p_chunk.text
            }
            hierarchical_chunks.append(c_chunk)
            global_child_idx += 1

    return hierarchical_chunks


def chunk_lumber_chunker(
    text: str,
    llm_client: Any,
    model_name: str = "openai/gpt-4o-mini",
    source_type: str = "",
    theta: int = 550,  # Optimal token context window from the paper
    **kwargs: Any,  # Absorbs chunk_size/overlap from the CLI; unused — sized via theta instead
) -> list[Chunk]:
    """Strategy 4: LumberChunker adapted for 3GPP Technical Specs.
    Splits by major Markdown headers to process sections concurrently,
    then runs the paper's rolling-window shift-detection inside each section.
    """
    if not text or not text.strip():
        return []

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split on top-level headers first, so the LLM never crosses a major clause boundary
    section_pattern = re.compile(r"(^##?\s+.*$)", re.MULTILINE)
    parts = section_pattern.split(text)
    
    sections = []
    current_section = ""
    for part in parts:
        if section_pattern.match(part):
            if current_section.strip():
                sections.append(current_section)
            current_section = part + "\n"
        else:
            current_section += part + "\n"
    if current_section.strip():
        sections.append(current_section)

    # Rolling-window algorithm from the paper, applied per section
    def process_section(section_text: str, section_idx: int) -> list[str]:
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        section_chunks = []
        p_idx = 0
        N = len(paragraphs)

        while p_idx < N:
            # Build Group (G_i) up to theta tokens
            group_paras = []
            group_tokens = 0
            curr_slice_idx = p_idx
            
            while curr_slice_idx < N and group_tokens < theta:
                para_text = f"ID {curr_slice_idx}: {paragraphs[curr_slice_idx]}"
                group_paras.append(para_text)
                group_tokens += len(tokenizer.encode(para_text))
                curr_slice_idx += 1

            # Last paragraph or end reached: consume the remaining group whole
            if len(group_paras) <= 1 or p_idx == N - 1:
                section_chunks.append("\n\n".join([p.split(": ", 1)[1] for p in group_paras]))
                break

            # Prompt exactly as specified in the LumberChunker paper (Appendix B)
            document_context = "\n".join(group_paras)
            system_prompt = (
                "You will receive as input an English document with paragraphs identified by 'ID XXXX: <text>'.\n"
                "Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.\n"
                "Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'."
            )
            
            try:
                response = llm_client.completion.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Document:\n{document_context}\n\nAnswer:"}
                    ]
                )

                llm_output = response.choices[0].message.content
                match = re.search(r"ID\s*(\d+)", llm_output, re.IGNORECASE)

                if match:
                    shift_id = int(match.group(1))
                    # Guard against the LLM returning an out-of-window ID
                    if p_idx < shift_id < curr_slice_idx:
                        # Chunk spans p_idx up to (but excluding) shift_id
                        chunk_text = "\n\n".join(paragraphs[p_idx:shift_id])
                        section_chunks.append(chunk_text)
                        p_idx = shift_id  # Next group starts where the shift occurred
                        continue

                raise ValueError("Invalid or out-of-bounds ID returned.")

            except Exception:
                # LLM call failed or returned garbage: fall back to a fixed-size block
                fallback_limit = min(p_idx + 3, N)
                chunk_text = "\n\n".join(paragraphs[p_idx:fallback_limit])
                section_chunks.append(chunk_text)
                p_idx = fallback_limit

        return section_chunks

    # Run rolling-window processing concurrently per section
    final_chunks = []
    global_idx = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_section, sec, i): i for i, sec in enumerate(sections)}

        # Preserve original section order despite concurrent completion
        ordered_results = [None] * len(sections)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            ordered_results[idx] = future.result()

    for chunk_strings in ordered_results:
        if chunk_strings:
            for text_block in chunk_strings:
                if text_block.strip():
                    final_chunks.append(Chunk(
                        text=text_block.strip(),
                        heading=None,
                        chunk_index=global_idx,
                        source_type=source_type
                    ))
                    global_idx += 1

    return final_chunks

# --- Infrastructure Interface Entry Point ---

def calculate_token_budget_safe_kwargs(strategy_str: str, chunk_size: int, overlap: int, enrichment: str) -> tuple[int, int]:
    """
    Adjusts token limits if post-chunking enrichment metadata tagging expands the token footprint.
    """
    allocated_chunk_size = chunk_size
    allocated_overlap = overlap
    
    if strategy_str == "sliding_window_tokens" and enrichment == "metadata_tagging":
        # Reserve buffer for the enrichment prefix so tokens don't overflow
        TOKEN_RESERVATION_BUFFER = 64
        allocated_chunk_size = max(128, chunk_size - TOKEN_RESERVATION_BUFFER)
        allocated_overlap = max(0, overlap - int(TOKEN_RESERVATION_BUFFER * 0.2))
        
    return allocated_chunk_size, allocated_overlap


def chunk_document(
    text: str,
    strategy: ChunkingStrategy | str,
    **kwargs: Any,
) -> list[Chunk]:
    """Routes text to the selected chunking strategy."""

    is_structured = kwargs.pop("is_structured", False)
    
    _STRATEGY_MAP = {
        ChunkingStrategy.HIERARCHICAL_MARKDOWN: chunk_hierarchical_markdown,
        ChunkingStrategy.SLIDING_WINDOW_TOKENS: chunk_sliding_window_tokens,
        ChunkingStrategy.PARENT_CHILD: chunk_parent_child,
        ChunkingStrategy.LUMBER_CHUNKER: chunk_lumber_chunker,
        ChunkingStrategy.TEXT_BASELINE: chunk_text_baseline,
    }

    if isinstance(strategy, str):
        try:
            strategy = ChunkingStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Supported choices: {[s.value for s in ChunkingStrategy]}"
            )

    strategy_str = strategy.value
    # pop, not get: enrichment isn't a real chunker param, and
    # chunk_sliding_window_tokens has no **kwargs to absorb it — get() left
    # it in kwargs and crashed with TypeError when passed directly.
    enrichment = kwargs.pop("enrichment", "none")

    if strategy == ChunkingStrategy.SLIDING_WINDOW_TOKENS:
        # Accept either generic (chunk_size) or specific (chunk_tokens) names
        raw_size = kwargs.get("chunk_tokens", kwargs.get("chunk_size", 512))
        raw_overlap = kwargs.get("overlap_tokens", kwargs.get("overlap", 50))

        safe_size, safe_overlap = calculate_token_budget_safe_kwargs(
            strategy_str, raw_size, raw_overlap, enrichment
        )

        kwargs["chunk_tokens"] = safe_size
        kwargs["overlap_tokens"] = safe_overlap

        # Remove now-duplicate generic keys to avoid conflicting kwargs
        if "chunk_size" in kwargs: del kwargs["chunk_size"]
        if "overlap" in kwargs: del kwargs["overlap"]

    chunker_function = _STRATEGY_MAP[strategy]
    return chunker_function(text, **kwargs)