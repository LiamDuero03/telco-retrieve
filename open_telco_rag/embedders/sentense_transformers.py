"""Local embedding models via sentence-transformers."""

from __future__ import annotations

import json
from pathlib import Path


class SentenceTransformerEmbedder:
    """Embeds text using a local sentence-transformers model."""

    def __init__(
        self,
        model_key:    str = "minilm",
        models_file:  str | Path = "configs/embedding_models.json",
    ):
        models_file = Path(models_file)
        if not models_file.exists():
            raise FileNotFoundError(f"Embedding models file not found: {models_file}")

        with open(models_file) as f:
            available = json.load(f)

        if model_key not in available:
            raise ValueError(
                f"Model key '{model_key}' not in {models_file}. "
                f"Available: {list(available.keys())}"
            )

        self._model_key   = model_key
        self._model_name  = available[model_key]["model_name"]
        self._dimensions  = available[model_key]["dimensions"]
        self._description = available[model_key].get("description", "")
        self._model       = None   # lazy load

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install open-telco-rag[local-embed]"
                )
            print(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        return self._model.encode(texts, show_progress_bar=False).tolist()