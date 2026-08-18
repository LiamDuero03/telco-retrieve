"""Local embedding models via sentence-transformers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class SentenceTransformerEmbedder:
    """Embeds text using a local sentence-transformers model."""

    def __init__(
        self,
        model_key:    str = "minilm",
        models_file: str | Path = Path(__file__).parent.parent.parent / "configs" / "embedding_models.json",
    ):
        models_file = Path(models_file)
        if not models_file.exists():
            raise FileNotFoundError(f"Embedding models file not found: {models_file}")

        with open(models_file) as f:
            available = json.load(f)

        if model_key not in available:
            raise ValueError(f"Model key '{model_key}' not in {models_file}.")

        self._model_key   = model_key
        self._model_name  = available[model_key]["model_name"]
        self._dimensions  = available[model_key]["dimensions"]
        self._description = available[model_key].get("description", "")
        self._model       = None
        self._first_embed_done = False

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install open-telco-rag[local-embed]"
                )
            print(f"Loading embedding model: {self._model_name}")
            load_start = time.perf_counter()

            # cap threads: avoids PyTorch thread thrashing on high-core CPUs (e.g. 95-core RunPod hosts)
            if not torch.cuda.is_available():
                cpu_cores = max(1, min(16, (os.cpu_count() or 2) - 1))
                torch.set_num_threads(cpu_cores)

            if torch.cuda.is_available():
                self._model = SentenceTransformer(
                    self._model_name,
                    device="cuda",
                    model_kwargs={"torch_dtype": torch.float16}
                )
            else:
                self._model = SentenceTransformer(
                    self._model_name,
                    device="cpu"
                )

            # DIAGNOSTIC: prints once GPU move + first CUDA/HIP context init finish (can
            # stall on ROCm) -- lets a hang here be distinguished from one in first embed()
            print(f"Embedding model ready on {self._model.device} "
                  f"in {time.perf_counter() - load_start:.1f}s")

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch
        self._load()

        optimal_batch_size = 256 if torch.cuda.is_available() else 32

        # DIAGNOSTIC: first embed() is the first real GPU forward pass -- can stall on
        # VRAM pressure or a hung CUDA/HIP context, so log batch size explicitly here
        is_first_embed = not self._first_embed_done
        if is_first_embed:
            print(f"Embedding first batch of this process: {len(texts)} texts "
                  f"(sub-batched at {optimal_batch_size}) -- first GPU forward "
                  f"pass in this process, may be slower than subsequent batches...")
            embed_start = time.perf_counter()

        if torch.cuda.is_available():
            # fp16 weights alone don't guarantee ops dispatch to Tensor Cores; autocast
            # forces that while still letting sensitive ops (layernorm, softmax) run fp32
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = self._model.encode(
                    texts,
                    batch_size=optimal_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        else:
            embeddings = self._model.encode(
                texts,
                batch_size=optimal_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

        if is_first_embed:
            print(f"First batch embedded in {time.perf_counter() - embed_start:.1f}s "
                  f"({len(texts) / max(time.perf_counter() - embed_start, 1e-9):.1f} texts/s) "
                  f"-- subsequent batches should be faster now that CUDA/HIP context "
                  f"and kernels are warmed up.")
            self._first_embed_done = True

        return embeddings.tolist()