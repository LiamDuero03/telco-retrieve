"""Text chunking strategies aligned with thesis benchmark requirements."""

from __future__ import annotations

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
    """Strategy 1: Identifies document layouts based on structural Markdown heading elements (# to #####)
    while maintaining a stateful parent breadcrumb trail stack across deep sub-sections.
    """
    if not text or not text.strip():
        return []

    # Regex split that captures heading lines while keeping them in the split stream
    heading_pattern = re.compile(r"(^#{1,5}\s+.*$)", re.MULTILINE)
    parts = heading_pattern.split(text)
    
    chunks: list[Chunk] = []
    # State stack to track active headers from # down to #####
    current_headers = {1: None, 2: None, 3: None, 4: None, 5: None}
    
    current_text_block = ""
    active_heading_attr = "None"
    global_idx = 0

    for part in parts:
        if not part.strip():
            continue
            
        # Check if the part is a heading line
        header_match = re.match(r"^(#{1,5})\s+(.*)$", part.strip())
        if header_match:
            level = len(header_match.group(1))
            heading_text = header_match.group(2).strip()
            
            # Update current level and clear deeper sub-levels to maintain stack hierarchy
            current_headers[level] = heading_text
            for l in range(level + 1, 6):
                current_headers[l] = None
                
            # Build stateful breadcrumb string
            active_breadcrumbs = [current_headers[i] for i in range(1, 6) if current_headers[i]]
            active_heading_attr = " > ".join(active_breadcrumbs) if active_breadcrumbs else "None"
            
            # Flush existing accumulated blocks before starting the new heading sector
            if current_text_block.strip():
                chunks.append(Chunk(
                    text=current_text_block.strip(),
                    heading=active_heading_attr,
                    chunk_index=global_idx,
                    source_type=source_type
                ))
                global_idx += 1
                current_text_block = ""
                
            # Keep the header title inline inside the chunk text payload
            current_text_block += part + "\n"
        else:
            # If appending this paragraph breaks structural size limits, flush it
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
    """Generic baseline paragraph-based chunker with window character overlap 
    and dynamic sentence-boundary backward snapping.
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
            
            # Sentence terminal backward snapping index check
            if len(current) > overlap:
                lookback_window = current[-overlap:]
                # Look for a natural punctuation split point inside the overlap tail
                split_match = list(re.finditer(r"[\.\!\?\n]\s*", lookback_window))
                if split_match:
                    # Snap overlap window index forward to start neatly right after punctuation terminal
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
    """Strategy 2: Fixed size segments tokenized precisely with explicit model sliding windows."""
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
) -> list[Chunk]:
    """Strategy 3: Two-Tier Retrieval. Identifies large parent segments 
    and breaks them down into small, highly targeted child vectors.
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
import re
import concurrent.futures
from typing import Any
import tiktoken
from open_telco_rag.types import Chunk

def chunk_lumber_chunker(
    text: str,
    llm_client: Any,
    model_name: str = "openai/gpt-4o-mini",
    source_type: str = "",
    theta: int = 550,  # Optimal token context window from the paper
) -> list[Chunk]:
    """Strategy 4: Lumber Chunker adapted for 3GPP Technical Specs.
    Splits by major Markdown headers to process sections concurrently,
    then executes the paper's rolling window shift-detection inside each section.
    """
    if not text or not text.strip():
        return []

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Step 1: Divide by top-level markdown headers to allow parallelization
    # This prevents the LLM from ever crossing a major Clause boundary by accident
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

    # Step 2: Define the paper's rolling-window algorithm for a single section
    def process_section(section_text: str, section_idx: int) -> list[str]:
        # Split section into raw paragraphs
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

            # If there's only 1 paragraph left or we reached the end, consume it completely
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
                
                # Extract the ID from "Answer: ID XXXX"
                llm_output = response.choices[0].message.content
                match = re.search(r"ID\s*(\d+)", llm_output, re.IGNORECASE)
                
                if match:
                    shift_id = int(match.group(1))
                    # Sanity check: Ensure the returned ID is actually inside our current window
                    if p_idx < shift_id < curr_slice_idx:
                        # Chunk spans from p_idx up to (but excluding) shift_id
                        chunk_text = "\n\n".join(paragraphs[p_idx:shift_id])
                        section_chunks.append(chunk_text)
                        p_idx = shift_id  # Next group starts exactly where the shift occurred
                        continue

                # Fallback if LLM hallucinations an out-of-bounds ID or fails regex
                raise ValueError("Invalid or out-of-bounds ID returned.")

            except Exception:
                # If the LLM call fails or returns garbage, advance by a baseline paragraph block
                fallback_limit = min(p_idx + 3, N)
                chunk_text = "\n\n".join(paragraphs[p_idx:fallback_limit])
                section_chunks.append(chunk_text)
                p_idx = fallback_limit

        return section_chunks

    # Step 3: Run the rolling windows concurrently across your structural sections
    final_chunks = []
    global_idx = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_section, sec, i): i for i, sec in enumerate(sections)}
        
        # Gather results keeping track of structural order
        ordered_results = [None] * len(sections)
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            ordered_results[idx] = future.result()

    # Flatten out to official Chunk objects
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
        # Reserve a safe token space margin (e.g., 64 tokens) strictly for the enrichment prefix string
        TOKEN_RESERVATION_BUFFER = 64
        allocated_chunk_size = max(128, chunk_size - TOKEN_RESERVATION_BUFFER)
        allocated_overlap = max(0, overlap - int(TOKEN_RESERVATION_BUFFER * 0.2))
        
    return allocated_chunk_size, allocated_overlap


def chunk_document(
    text: str,
    strategy: ChunkingStrategy | str,
    **kwargs: Any,
) -> list[Chunk]:
    """Master orchestration function to route text data to selected chunking strategy."""
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

    # --- TOKEN BUDGET ADJUSTMENT PASS ---
    strategy_str = strategy.value
    enrichment = kwargs.get("enrichment", "none")  # Extract enrichment string if passed from master sweep

    if strategy == ChunkingStrategy.SLIDING_WINDOW_TOKENS:
        # Map generic size parameters to specific token keyword arguments
        raw_size = kwargs.get("chunk_tokens", kwargs.get("chunk_size", 512))
        raw_overlap = kwargs.get("overlap_tokens", kwargs.get("overlap", 50))
        
        # Calculate safe parameters
        safe_size, safe_overlap = calculate_token_budget_safe_kwargs(
            strategy_str, raw_size, raw_overlap, enrichment
        )
        
        # Re-assign variables back to match signature specifications expected by chunk_sliding_window_tokens
        kwargs["chunk_tokens"] = safe_size
        kwargs["overlap_tokens"] = safe_overlap
        
        # Clean up baseline parameters to avoid unexpected argument conflicts
        if "chunk_size" in kwargs: del kwargs["chunk_size"]
        if "overlap" in kwargs: del kwargs["overlap"]

    chunker_function = _STRATEGY_MAP[strategy]
    return chunker_function(text, **kwargs)