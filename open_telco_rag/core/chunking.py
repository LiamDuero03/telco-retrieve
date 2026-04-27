"""Text chunking strategies."""

from __future__ import annotations

import re

from open_telco_rag.types import Chunk


def _accumulate_chunks(
    paragraphs: list[str],
    chunk_size: int,
    heading:    str | None = None,
    source_type: str = "",
) -> list[Chunk]:
    """Group paragraphs into chunks up to chunk_size characters."""
    chunks: list[Chunk] = []
    current = ""
    idx     = 0

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
    text:        str,
    chunk_size:  int = 4000,
    source_type: str = "",
) -> list[Chunk]:
    """Split markdown on heading boundaries then by paragraph."""
    if not text or not text.strip():
        return []

    sections   = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    all_chunks: list[Chunk] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"^#{1,3}\s+(.+)$", section, re.MULTILINE)
        heading       = heading_match.group(1).strip() if heading_match else None

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
    text:        str,
    chunk_size:  int = 4000,
    overlap:     int = 200,
    source_type: str = "",
) -> list[Chunk]:
    """Generic paragraph-based chunker with overlap for PDFs and plain text."""
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    idx     = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(Chunk(
                text=current.strip(),
                heading=None,
                chunk_index=idx,
                source_type=source_type
            ))
            idx += 1
            # keep overlap from end of current chunk
            # note: overlap here counts words, not characters
            words   = current.split()
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