"""Text chunking strategies aligned with thesis benchmark requirements."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from enum import Enum
from collections.abc import Callable
from typing import Any
import tiktoken

from open_telco_rag.types import Chunk

logger = logging.getLogger(__name__)


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


# ── Token-aware helpers (shared by the target_tokens/max_tokens code paths) ──
# All token counting uses cl100k_base, same encoding as the pre-existing
# sliding_window_tokens strategy, so counts are comparable across strategies.
# the encoding is created lazily (default char paths never load a tokenizer) and is
# shared with core/retrieval.py for budgeted context assembly
_ENCODING_CACHE: dict[str, Any] = {}


def _get_encoding() -> Any:
    enc = _ENCODING_CACHE.get("cl100k_base")
    if enc is None:
        enc = tiktoken.get_encoding("cl100k_base")
        _ENCODING_CACHE["cl100k_base"] = enc
    return enc


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def _check_token_overlap(overlap_tokens: int, size_tokens: int) -> None:
    """token-mode overlap must be in [0, size): a larger one would degrade to a one-token stride"""
    if overlap_tokens < 0 or overlap_tokens >= size_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be at least 0 and smaller than the chunk "
            f"size ({size_tokens} tokens)."
        )


def _token_chunk(
    text: str,
    heading: str | None,
    chunk_index: int,
    source_type: str,
    char_start: int,
    char_end: int,
    n_tokens: int | None = None,
) -> Chunk:
    """chunk with offsets + token count set as instance attrs (not dataclass fields)"""
    chunk = Chunk(text=text, heading=heading, chunk_index=chunk_index, source_type=source_type)
    chunk.char_start = char_start
    chunk.char_end = char_end
    chunk.n_tokens = _count_tokens(text) if n_tokens is None else n_tokens
    return chunk


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """offsets of blank-line-separated paragraphs, trimmed of whitespace; same split as the legacy
    paragraph split but keeps the offsets"""
    seps = [m.span() for m in re.finditer(r"\n[ \t]*\n[ \t\n]*", text)]
    bounds = []
    prev_end = 0
    for s_start, s_end in seps:
        bounds.append((prev_end, s_start))
        prev_end = s_end
    bounds.append((prev_end, len(text)))

    spans = []
    for start, end in bounds:
        seg = text[start:end]
        lstripped = len(seg) - len(seg.lstrip())
        rstripped = len(seg) - len(seg.rstrip())
        real_start = start + lstripped
        real_end = end - rstripped
        if real_start < real_end:
            spans.append((real_start, real_end))
    return spans


_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?]+(?:\s+|$)|[^.!?\n]+$")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """offsets of sentences (punctuation-based, best effort); only the fallback when a paragraph
    exceeds max_tokens"""
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        seg = m.group(0)
        if not seg.strip():
            continue
        lstripped = len(seg) - len(seg.lstrip())
        rstripped = len(seg) - len(seg.rstrip())
        s = m.start() + lstripped
        e = m.end() - rstripped
        if s < e:
            spans.append((s, e))
    return spans


def _hard_token_split(text: str, start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    """split text[start:end] into pieces of <= max_tokens tokens; last resort, never raises.
    offsets come from cumulative decode length: a lone token need not be valid UTF-8, but any
    prefix of the document's own token stream decodes exactly"""
    segment = text[start:end]
    tokens = _get_encoding().encode(segment)
    if not tokens:
        return [(start, end)] if start < end else []

    spans = []
    tok_start = 0
    while tok_start < len(tokens):
        tok_end = min(tok_start + max_tokens, len(tokens))
        char_a = len(_get_encoding().decode(tokens[:tok_start]))
        char_b = len(_get_encoding().decode(tokens[:tok_end]))
        spans.append((start + char_a, start + char_b))
        tok_start = tok_end
    return spans


def _accumulate_bounded(
    text: str,
    unit_spans: list[tuple[int, int]],
    max_tokens: int,
    deeper_split: Callable[[str, int, int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """greedily group consecutive unit spans up to max_tokens; a unit that alone exceeds it goes
    to `deeper_split` instead of being left oversized"""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(unit_spans)
    while i < n:
        u_start, u_end = unit_spans[i]
        u_tokens = _count_tokens(text[u_start:u_end])
        if u_tokens > max_tokens:
            out.extend(deeper_split(text, u_start, u_end, max_tokens))
            i += 1
            continue

        group_start, group_end = u_start, u_end
        j = i + 1
        while j < n:
            cand_start, cand_end = unit_spans[j]
            cand_tokens = _count_tokens(text[cand_start:cand_end])
            if cand_tokens > max_tokens:
                break
            merged_tokens = _count_tokens(text[group_start:cand_end])
            if merged_tokens > max_tokens:
                break
            group_end = cand_end
            j += 1
        out.append((group_start, group_end))
        i = j
    return out


def _bounded_split(text: str, start: int, end: int, max_tokens: int) -> list[tuple[int, int]]:
    """split into pieces of <= max_tokens: paragraph boundaries, then sentences, then a hard token cut"""
    if _count_tokens(text[start:end]) <= max_tokens:
        return [(start, end)]

    paras = [(s + start, e + start) for s, e in _paragraph_spans(text[start:end])]
    if not paras:
        return _hard_token_split(text, start, end, max_tokens)

    def _sentence_deeper(t: str, s: int, e: int, mt: int) -> list[tuple[int, int]]:
        sents = [(ss + s, se + s) for ss, se in _sentence_spans(t[s:e])]
        if not sents:
            return _hard_token_split(t, s, e, mt)
        return _accumulate_bounded(t, sents, mt, _hard_token_split)

    return _accumulate_bounded(text, paras, max_tokens, _sentence_deeper)


def _spans_to_chunks(
    text: str,
    spans: list[tuple[int, int]],
    heading: str | None,
    source_type: str,
    start_idx: int = 0,
) -> list[Chunk]:
    chunks = []
    for offset, (s, e) in enumerate(spans):
        chunk_text = text[s:e]
        chunks.append(_token_chunk(chunk_text, heading, start_idx + offset, source_type, s, e))
    return chunks


def _accumulate_paragraphs_with_overlap(
    text: str,
    spans: list[tuple[int, int]],
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> list[tuple[int, int]]:
    """group paragraph spans up to target_tokens; each next group re-includes trailing paragraphs
    worth up to overlap_tokens (whole paragraphs only, so offsets stay exact slices); a lone
    paragraph over max_tokens goes to _bounded_split"""
    n = len(spans)
    if n == 0:
        return []

    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        u_start, u_end = spans[i]
        u_tokens = _count_tokens(text[u_start:u_end])
        if u_tokens > max_tokens:
            groups.extend(_bounded_split(text, u_start, u_end, max_tokens))
            i += 1
            continue

        group_start_idx = i
        group_end = u_end
        j = i + 1
        while j < n:
            cand_tokens = _count_tokens(text[spans[j][0]:spans[j][1]])
            if cand_tokens > max_tokens:
                break
            merged_tokens = _count_tokens(text[spans[group_start_idx][0]:spans[j][1]])
            if merged_tokens > target_tokens:
                break
            group_end = spans[j][1]
            j += 1
        groups.append((spans[group_start_idx][0], group_end))

        if j >= n:
            break

        # overlap: walk back from the last included paragraph while the cumulative
        # tokens stay within overlap_tokens
        if overlap_tokens > 0:
            back = j - 1
            cum = 0
            while back > group_start_idx:
                seg_tokens = _count_tokens(text[spans[back][0]:spans[back][1]])
                if cum + seg_tokens > overlap_tokens:
                    break
                cum += seg_tokens
                back -= 1
            next_i = back + 1
            # guard against no forward progress
            i = next_i if next_i > group_start_idx else j
        else:
            i = j

    return groups


def chunk_hierarchical_markdown(
    text: str,
    chunk_size: int = 4000,
    source_type: str = "",
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> list[Chunk]:
    """Strategy 1: splits on Markdown headings (# to #####), tracking a breadcrumb trail.

    Legacy path (target_tokens and max_tokens both None): character-sized, offsets left at -1.
    Token path (either given): undersized sections merged up to target_tokens, oversized ones
    split so none exceeds max_tokens; chunk text is an exact slice of `text`.
    """
    if not text or not text.strip():
        return []

    if target_tokens is not None or max_tokens is not None:
        return _chunk_hierarchical_markdown_tokens(
            text, source_type=source_type,
            target_tokens=target_tokens, max_tokens=max_tokens,
        )

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
            # Flush under the previous heading before updating active_heading_attr.
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


def _hierarchical_sections(text: str) -> list[dict]:
    """(start, end, heading) for each heading-delimited section, same breadcrumb logic as the
    legacy character path"""
    heading_re = re.compile(r"^(#{1,5})\s+(.*)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    if not matches:
        return [{"start": 0, "end": len(text), "heading": "None"}]

    sections = []
    if matches[0].start() > 0 and text[:matches[0].start()].strip():
        sections.append({"start": 0, "end": matches[0].start(), "heading": "None"})

    current_headers: dict[int, str | None] = {1: None, 2: None, 3: None, 4: None, 5: None}
    for idx, m in enumerate(matches):
        level = len(m.group(1))
        heading_text = m.group(2).strip()
        current_headers[level] = heading_text
        for l in range(level + 1, 6):
            current_headers[l] = None
        breadcrumbs = [current_headers[i] for i in range(1, 6) if current_headers[i]]
        breadcrumb = " > ".join(breadcrumbs) if breadcrumbs else "None"

        sec_start = m.start()
        sec_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append({"start": sec_start, "end": sec_end, "heading": breadcrumb})

    return sections


def _chunk_hierarchical_markdown_tokens(
    text: str,
    source_type: str,
    target_tokens: int | None,
    max_tokens: int | None,
) -> list[Chunk]:
    sections = _hierarchical_sections(text)
    effective_target = target_tokens or max_tokens
    effective_max = max_tokens or target_tokens

    # Merge undersized adjacent sections up to effective_target, keeping the
    # first section's breadcrumb (most general context) for the merged block.
    merged: list[dict] = []
    cur = dict(sections[0])
    cur["n_tokens"] = _count_tokens(text[cur["start"]:cur["end"]])
    for sec in sections[1:]:
        seg_tokens = _count_tokens(text[sec["start"]:sec["end"]])
        if cur["n_tokens"] + seg_tokens <= effective_target:
            cur["end"] = sec["end"]
            cur["n_tokens"] += seg_tokens
        else:
            merged.append(cur)
            cur = dict(sec)
            cur["n_tokens"] = seg_tokens
    merged.append(cur)

    # Split anything still over effective_max at paragraph/sentence/hard-cut boundaries.
    final_chunks: list[Chunk] = []
    idx = 0
    for m in merged:
        if m["n_tokens"] > effective_max:
            spans = _bounded_split(text, m["start"], m["end"], effective_max)
        else:
            spans = [(m["start"], m["end"])]
        for s, e in spans:
            chunk_text = text[s:e]
            final_chunks.append(_token_chunk(chunk_text, m["heading"], idx, source_type, s, e))
            idx += 1

    return final_chunks


def chunk_text_baseline(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
    source_type: str = "",
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    **kwargs: Any,
) -> list[Chunk]:
    """Baseline paragraph-based chunker.

    Legacy path (target_tokens and max_tokens both None): character overlap snapped to a
    sentence boundary, offsets left at -1.
    Token path (either given): paragraphs accumulated up to target_tokens per chunk (max_tokens
    defaults to target_tokens), overlapping by up to overlap_tokens (default 0); exact slices.
    """
    if not text or not text.strip():
        return []

    if target_tokens is not None or max_tokens is not None:
        effective_target = target_tokens or max_tokens
        effective_max = max_tokens or target_tokens
        effective_overlap = overlap_tokens or 0
        _check_token_overlap(effective_overlap, effective_target)
        spans = _paragraph_spans(text)
        groups = _accumulate_paragraphs_with_overlap(
            text, spans, effective_target, effective_overlap, effective_max,
        )
        return _spans_to_chunks(text, groups, heading=None, source_type=source_type)

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

# tells "overlap_tokens not passed" from "passed 0": the legacy default is 50, the token path's is 0
_OVERLAP_TOKENS_UNSET = object()


def chunk_sliding_window_tokens(
    text: str,
    chunk_tokens: int = 512,
    overlap_tokens: Any = _OVERLAP_TOKENS_UNSET,
    encoding_name: str = "cl100k_base",
    source_type: str = "",
    target_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Strategy 2: fixed-size token windows with sliding overlap.

    overlap_tokens defaults to 50 on the legacy path and 0 on the token path (see
    _OVERLAP_TOKENS_UNSET); pass it explicitly to override.
    Legacy path (target_tokens and max_tokens both None): decoded text stripped, offsets -1.
    Token path (either given): target_tokens overrides chunk_tokens, max_tokens caps it; text is
    left unstripped so it stays an exact slice of `text`.
    """
    if not text or not text.strip():
        return []

    if target_tokens is None and max_tokens is None:
        legacy_overlap = 50 if overlap_tokens is _OVERLAP_TOKENS_UNSET else overlap_tokens
        tokenizer = tiktoken.get_encoding(encoding_name)
        tokens = tokenizer.encode(text)
        if chunk_tokens <= legacy_overlap and len(tokens) > chunk_tokens:
            # unchanged legacy behavior: the window cannot advance, so only the
            # first window is produced and the rest of the document is dropped
            logger.warning(
                "sliding_window_tokens: overlap_tokens (%d) >= chunk_tokens (%d), so only the first "
                "%d of %d tokens of this document are chunked; the rest is dropped. Use an overlap "
                "smaller than the chunk size.", legacy_overlap, chunk_tokens, chunk_tokens, len(tokens),
            )

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
            start_idx += (chunk_tokens - legacy_overlap)

            if start_idx >= len(tokens) or (chunk_tokens <= legacy_overlap):
                break

        return chunks

    effective_chunk_tokens = target_tokens or chunk_tokens
    if max_tokens is not None:
        effective_chunk_tokens = min(effective_chunk_tokens, max_tokens)
    effective_overlap = 0 if overlap_tokens is _OVERLAP_TOKENS_UNSET else overlap_tokens
    _check_token_overlap(effective_overlap, effective_chunk_tokens)

    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)

    chunks = []
    start_tok = 0
    chunk_index = 0
    while start_tok < len(tokens):
        end_tok = min(start_tok + effective_chunk_tokens, len(tokens))
        char_a = len(tokenizer.decode(tokens[:start_tok]))
        char_b = len(tokenizer.decode(tokens[:end_tok]))
        chunk_text = text[char_a:char_b]
        chunks.append(_token_chunk(chunk_text, None, chunk_index, source_type,
                                   char_a, char_b, n_tokens=end_tok - start_tok))
        chunk_index += 1
        start_tok += (effective_chunk_tokens - effective_overlap)

        if start_tok >= len(tokens) or (effective_chunk_tokens <= effective_overlap):
            break

    return chunks


def chunk_parent_child(
    text: str,
    parent_size: int = 2048,
    child_size: int = 512,
    source_type: str = "",
    target_tokens: int | None = None,
    parent_target_tokens: int | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> list[Chunk]:
    """Strategy 3: two-tier retrieval - large parent segments broken down into small child vectors.

    Legacy path (target_tokens and max_tokens both None): offsets left at -1.
    Token path: target_tokens is the CHILD size (max_tokens stands in when only that is given);
    parents come from the hierarchical-markdown token path at parent_target_tokens (default 4x
    the child target, the legacy parent/child ratio); children have no overlap and are capped
    at max_tokens. Parent and child text are exact slices of `text`.
    """
    if target_tokens is None and max_tokens is None:
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

    child_target = target_tokens or max_tokens
    effective_parent_target = parent_target_tokens or (child_target * 4)
    effective_max = max_tokens or child_target

    parent_chunks = chunk_hierarchical_markdown(
        text, source_type=source_type,
        target_tokens=effective_parent_target, max_tokens=effective_parent_target,
    )

    hierarchical_chunks = []
    global_child_idx = 0
    for p_chunk in parent_chunks:
        parent_chunk_id = f"{source_type}_P_{p_chunk.chunk_index}"
        child_spans = _paragraph_spans(p_chunk.text)
        child_groups = _accumulate_bounded(
            p_chunk.text, child_spans, child_target, _bounded_split,
        )
        # Cap at effective_max too, in case the child target > max_tokens.
        if effective_max < child_target:
            capped_groups = []
            for s, e in child_groups:
                capped_groups.extend(_bounded_split(p_chunk.text, s, e, effective_max))
            child_groups = capped_groups

        for s, e in child_groups:
            child_text = p_chunk.text[s:e]
            c_chunk = _token_chunk(child_text, p_chunk.heading, global_child_idx, source_type,
                                   p_chunk.char_start + s, p_chunk.char_start + e)
            c_chunk.metadata = {
                "parent_id": parent_chunk_id,
                "parent_context": p_chunk.text[:200] + "...",
                "full_parent_text": p_chunk.text,
                # the parent's own span, so resolving a child to its parent
                # keeps offsets that describe the text actually used
                "parent_char_start": p_chunk.char_start,
                "parent_char_end": p_chunk.char_end,
                "parent_n_tokens": p_chunk.n_tokens,
            }
            hierarchical_chunks.append(c_chunk)
            global_child_idx += 1

    return hierarchical_chunks


def _lumber_sections(text: str) -> list[tuple[int, int]]:
    """offsets of each level-1/2-heading section, same boundaries as the legacy section_pattern
    split"""
    section_re = re.compile(r"^##?\s+.*$", re.MULTILINE)
    matches = list(section_re.finditer(text))
    if not matches:
        return [(0, len(text))]

    spans = []
    if matches[0].start() > 0 and text[:matches[0].start()].strip():
        spans.append((0, matches[0].start()))
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spans.append((start, end))
    return spans


def _chunk_lumber_chunker_tokens(
    text: str,
    llm_client: Any,
    model_name: str,
    source_type: str,
    theta: int,
    max_tokens: int,
) -> list[Chunk]:
    """Token path: same rolling-window shift-detection as the legacy
    implementation, but operating on paragraph SPANS (via _paragraph_spans)
    instead of a stripped/rejoined string list, so every emitted chunk is an
    exact slice of `text` with a real char_start/char_end -- then capped at
    max_tokens via _bounded_split (theta bounds the LLM's view of a group,
    not the chunk it decides to emit)."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    section_spans = _lumber_sections(text)

    def process_section(sec_start: int, sec_end: int) -> list[tuple[int, int]]:
        para_spans = [(s + sec_start, e + sec_start) for s, e in _paragraph_spans(text[sec_start:sec_end])]
        n = len(para_spans)
        if n == 0:
            return []

        out: list[tuple[int, int]] = []
        p_idx = 0
        while p_idx < n:
            group_tokens = 0
            curr = p_idx
            while curr < n and group_tokens < theta:
                seg_text = f"ID {curr}: {text[para_spans[curr][0]:para_spans[curr][1]]}"
                group_tokens += len(tokenizer.encode(seg_text))
                curr += 1
            group_end_idx = curr

            if group_end_idx - p_idx <= 1 or p_idx == n - 1:
                out.append((para_spans[p_idx][0], para_spans[group_end_idx - 1][1]))
                break

            document_context = "\n".join(
                f"ID {i}: {text[para_spans[i][0]:para_spans[i][1]]}" for i in range(p_idx, group_end_idx)
            )
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
                    if p_idx < shift_id < group_end_idx:
                        out.append((para_spans[p_idx][0], para_spans[shift_id - 1][1]))
                        p_idx = shift_id
                        continue
                raise ValueError("Invalid or out-of-bounds ID returned.")
            except Exception:
                fallback_limit = min(p_idx + 3, n)
                out.append((para_spans[p_idx][0], para_spans[fallback_limit - 1][1]))
                p_idx = fallback_limit

        return out

    section_chunk_spans: list[list[tuple[int, int]] | None] = [None] * len(section_spans)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_section, s, e): i
            for i, (s, e) in enumerate(section_spans)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            section_chunk_spans[idx] = future.result()

    final_chunks: list[Chunk] = []
    idx = 0
    for spans in section_chunk_spans:
        if not spans:
            continue
        for s, e in spans:
            capped = _bounded_split(text, s, e, max_tokens) if _count_tokens(text[s:e]) > max_tokens else [(s, e)]
            for cs, ce in capped:
                chunk_text = text[cs:ce]
                final_chunks.append(_token_chunk(chunk_text, None, idx, source_type, cs, ce))
                idx += 1
    return final_chunks


def chunk_lumber_chunker(
    text: str,
    llm_client: Any,
    model_name: str = "openai/gpt-4o-mini",
    source_type: str = "",
    theta: int = 550,  # token context window (from the paper)
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,  # absorbs unused CLI chunk_size/overlap
) -> list[Chunk]:
    """Strategy 4: LumberChunker adapted for 3GPP specs. Splits by major Markdown headers, then
    runs the paper's rolling-window shift-detection per section, concurrently.

    Legacy path (target_tokens and max_tokens both None): offsets left at -1.
    Token path (either given): same algorithm, every chunk an exact slice of `text` and none over
    max_tokens (defaults to target_tokens); theta, the LLM's view window, is unchanged.
    """
    if not text or not text.strip():
        return []

    if target_tokens is not None or max_tokens is not None:
        return _chunk_lumber_chunker_tokens(
            text, llm_client, model_name, source_type, theta, max_tokens or target_tokens,
        )

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split on top-level headers so the LLM never crosses a major clause boundary.
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

            # Last paragraph or end reached: consume the group whole.
            if len(group_paras) <= 1 or p_idx == N - 1:
                section_chunks.append("\n\n".join([p.split(": ", 1)[1] for p in group_paras]))
                break

            # Prompt from the LumberChunker paper, Appendix B.
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
                    if p_idx < shift_id < curr_slice_idx:  # ignore out-of-window IDs
                        chunk_text = "\n\n".join(paragraphs[p_idx:shift_id])
                        section_chunks.append(chunk_text)
                        p_idx = shift_id
                        continue

                raise ValueError("Invalid or out-of-bounds ID returned.")

            except Exception:  # noqa: BLE001 - any LLM failure falls back to a fixed-size block
                # LLM failed or returned garbage: fall back to a fixed-size block.
                fallback_limit = min(p_idx + 3, N)
                chunk_text = "\n\n".join(paragraphs[p_idx:fallback_limit])
                section_chunks.append(chunk_text)
                p_idx = fallback_limit

        return section_chunks

    final_chunks = []
    global_idx = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_section, sec, i): i for i, sec in enumerate(sections)}

        # Preserve section order despite concurrent completion.
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

def calculate_token_budget_safe_kwargs(strategy_str: str, chunk_size: int, overlap: int, enrichment: str) -> tuple[int, int]:
    """Shrink token limits to leave room for a metadata-tagging prefix."""
    allocated_chunk_size = chunk_size
    allocated_overlap = overlap

    if strategy_str == "sliding_window_tokens" and enrichment == "metadata_tagging":
        TOKEN_RESERVATION_BUFFER = 64
        allocated_chunk_size = max(128, chunk_size - TOKEN_RESERVATION_BUFFER)
        allocated_overlap = max(0, overlap - int(TOKEN_RESERVATION_BUFFER * 0.2))

    return allocated_chunk_size, allocated_overlap


# combining these with target_tokens/max_tokens is rejected, not resolved silently
_CHAR_SIZE_KEYS = ("chunk_size", "overlap", "parent_size", "child_size")


def _validate_token_sizing(strategy: ChunkingStrategy, kwargs: dict[str, Any]) -> None:
    """ValueError on bad token values or a mix with explicit char sizes / chunk_tokens"""
    for name in ("target_tokens", "max_tokens"):
        value = kwargs.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    target, cap = kwargs.get("target_tokens"), kwargs.get("max_tokens")
    if target is not None and cap is not None and cap < target:
        raise ValueError(f"max_tokens ({cap}) must be >= target_tokens ({target})")

    conflicting = [k for k in _CHAR_SIZE_KEYS if kwargs.get(k) is not None]
    if strategy == ChunkingStrategy.SLIDING_WINDOW_TOKENS and kwargs.get("chunk_tokens") is not None:
        conflicting.append("chunk_tokens")
    if conflicting:
        raise ValueError(
            "target_tokens/max_tokens cannot be combined with an explicit size argument "
            f"({', '.join(conflicting)}). Use either token-based sizing "
            "(target_tokens/max_tokens) or the existing size arguments, not both."
        )


def chunk_document(
    text: str,
    strategy: ChunkingStrategy | str,
    **kwargs: Any,
) -> list[Chunk]:
    """Routes text to the selected chunking strategy.

    sizing is char-based by default; target_tokens/max_tokens opt in to token
    sizing for any strategy (mixing the two styles raises ValueError).
    """

    is_structured = kwargs.pop("is_structured", False)

    # None = not set
    for _opt in ("target_tokens", "max_tokens"):
        if kwargs.get(_opt) is None:
            kwargs.pop(_opt, None)

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

    if "target_tokens" in kwargs or "max_tokens" in kwargs:
        _validate_token_sizing(strategy, kwargs)

    strategy_str = strategy.value
    # pop: enrichment isn't a chunker param and sliding_window has no **kwargs.
    enrichment = kwargs.pop("enrichment", "none")

    # only the legacy chunk_tokens/overlap_tokens path gets the metadata_tagging safety margin
    # below; injecting its default overlap_tokens (50) into the token path would force near-total overlap
    if (strategy == ChunkingStrategy.SLIDING_WINDOW_TOKENS
            and "target_tokens" not in kwargs and "max_tokens" not in kwargs):
        # Accept generic (chunk_size) or specific (chunk_tokens) names.
        raw_size = kwargs.get("chunk_tokens", kwargs.get("chunk_size", 512))
        raw_overlap = kwargs.get("overlap_tokens", kwargs.get("overlap", 50))

        safe_size, safe_overlap = calculate_token_budget_safe_kwargs(
            strategy_str, raw_size, raw_overlap, enrichment
        )

        kwargs["chunk_tokens"] = safe_size
        kwargs["overlap_tokens"] = safe_overlap

        # Drop now-duplicate generic keys.
        if "chunk_size" in kwargs: del kwargs["chunk_size"]
        if "overlap" in kwargs: del kwargs["overlap"]

    chunker_function = _STRATEGY_MAP[strategy]
    return chunker_function(text, **kwargs)
