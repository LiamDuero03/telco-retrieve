"""Open Telco RAG — unified RAG pipeline for telecom specifications."""

from open_telco_rag.core.pipeline import ingest, search, generate
from open_telco_rag.types import Chunk, IngestStats, SearchResult

__version__ = "0.1.0"
__all__ = ["ingest", "search", "generate", "Chunk", "IngestStats", "SearchResult"]