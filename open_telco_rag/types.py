"""Shared types — Protocols define pluggable contracts, dataclasses carry data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ── Protocols ─────────────────────────────────────────────────────────────────

class Embedder(Protocol):
    """Anything that converts text into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def model_key(self) -> str: ...

    @property
    def dimensions(self) -> int: ...


class VectorStore(Protocol):
    """Anything that stores and searches vectors."""

    def add(
        self,
        ids:        list[str],
        embeddings: list[list[float]],
        documents:  list[str],
        metadatas:  list[dict],
    ) -> None: ...

    def search(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> list[SearchResult]: ...

    def count(self) -> int: ...

    def exists(self, id: str) -> bool: ...


class Chunker(Protocol):
    """Anything that splits text into chunks."""

    def chunk(self, text: str) -> list[Chunk]: ...


class Source(Protocol):
    """Anything that loads and prepares documents from a directory."""

    def load(self, path) -> str: ...

    def clean(self, text: str) -> str: ...

    def metadata(self, path) -> dict: ...

    def supported_extensions(self) -> list[str]: ...


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A chunk of text with its context."""
    text:        str
    heading:     str | None
    chunk_index: int
    source_type: str = ""        # e.g. "oran", "3gpp", "camara"


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    text:     str
    distance: float
    metadata: dict
    doc_id:   str

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance

@dataclass
class IngestStats:
    """Statistics returned by ingest()."""
    files_processed: int = 0
    chunks_stored:   int = 0
    skipped_files:   int = 0
    error_files:     int = 0
    errors:          list[dict] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question:         str
    answer:           str
    sources:          list[str]
    chunks_used:      int
    keyword_score:    float | None
    keywords_hit:     list[str]
    keywords_missed:  list[str]
    ground_truth:     str
    retrieved_chunks: list[str]
    embedding_model:  str
    collection:       str

class Reranker(Protocol):
    """Anything that reranks a list of search results given a query."""

    def rerank(
        self,
        query:   str,
        results: list[SearchResult],
        top_k:   int = 5,
    ) -> list[SearchResult]: ...

    @property
    def model_name(self) -> str: ...

class Generator(Protocol):
    """Anything that generates an answer from a question and retrieved chunks.

    Implementations:
      - ClaudeGenerator         — Anthropic Claude
      - OpenRouterGenerator     — Any model via OpenRouter (GPT-4o, Llama, Mistral...)

    Returns a dict with keys:
      answer, sources, chunks_used, retrieved_chunks
    """

    def generate(
        self,
        question:    str,
        results:     list[SearchResult],
        source_type: str = "telecom",
    ) -> dict: ...