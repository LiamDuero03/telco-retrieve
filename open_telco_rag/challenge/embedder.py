from __future__ import annotations

import json
import tempfile
from pathlib import Path

from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder


def build_custom_embedder(
    model: str,
    *,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    dimensions: int = 0,
) -> SentenceTransformerEmbedder:
    """A sentence-transformers model by local path or HF id, via the registry fields."""
    entry = {
        "model_name": model,
        "dimensions": dimensions,
        "description": "custom embedder (challenge flow)",
    }
    if query_prefix:
        entry["query_prefix"] = query_prefix
    if document_prefix:
        entry["document_prefix"] = document_prefix
    with tempfile.TemporaryDirectory() as folder:
        registry = Path(folder) / "embedding_models.json"
        registry.write_text(json.dumps({"custom": entry}), encoding="utf-8")
        return SentenceTransformerEmbedder(model_key="custom", models_file=registry)
