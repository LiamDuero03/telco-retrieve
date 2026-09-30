"""Local embedding models via sentence-transformers."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Embeds text using a local sentence-transformers model."""

    def __init__(
        self,
        model_key:    str = "minilm",
        models_file: str | Path = Path(__file__).parent.parent.parent / "configs" / "embedding_models.json",
        track_truncation: bool = False,
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
        # Plain string prefixes (e.g. e5's "query: "/"passage: ", Qwen3-
        # Embedding's instruction-style query prefix) applied only when set
        # -- every entry without these keys embeds exactly as before.
        self._query_prefix    = available[model_key].get("query_prefix")
        self._document_prefix = available[model_key].get("document_prefix")
        # SBERT prompt_name mechanism (e.g. EmbeddingGemma's named prompts,
        # resolved from the model's OWN config, not a literal string here)
        # -- mutually independent of the plain-prefix keys above; a model
        # entry uses one mechanism or the other, never both.
        self._query_prompt_name    = available[model_key].get("query_prompt_name")
        self._document_prompt_name = available[model_key].get("document_prompt_name")
        self._model       = None
        self._first_embed_done = False
        # opt-in: with track_truncation=True every embed() also tokenizes the
        # texts to count how many exceed max_seq_length (they get silently
        # truncated by the model otherwise); see truncation_rate below.
        # off by default: the extra tokenization pass costs time.
        self.track_truncation = track_truncation
        self.truncated_count = 0
        self.scored_count    = 0

    @property
    def max_seq_length(self) -> int | None:
        """The loaded model's max_seq_length (tokens), or None before the
        first embed() call (the model isn't loaded yet)."""
        return getattr(self._model, "max_seq_length", None) if self._model is not None else None

    @property
    def truncation_rate(self) -> float:
        """Fraction of texts embedded so far whose true token length
        exceeded max_seq_length. 0.0 if nothing has been embedded yet, or if
        the length check itself isn't available for this model."""
        return (self.truncated_count / self.scored_count) if self.scored_count else 0.0

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

            # Cap threads to avoid thrashing on high-core CPUs.
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

            # Logs once GPU move + first CUDA/HIP init finish.
            print(f"Embedding model ready on {self._model.device} "
                  f"in {time.perf_counter() - load_start:.1f}s")

    def _record_length_check(self, texts: list[str]) -> None:
        """Best-effort: counts how many of `texts` will be silently
        truncated by the model (true token length > max_seq_length), warns
        once per embed() call if any are, and accumulates into
        truncated_count/scored_count. Never raises -- not every
        SentenceTransformer backend exposes a HF-style .tokenizer.encode()."""
        if not self.track_truncation:
            return
        max_len = getattr(self._model, "max_seq_length", None)
        tokenizer = getattr(self._model, "tokenizer", None)
        if max_len is None or tokenizer is None:
            return
        try:
            lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        except Exception:  # noqa: BLE001 - best-effort length check, never raises
            return

        n_over = sum(1 for n in lengths if n > max_len)
        self.truncated_count += n_over
        self.scored_count += len(texts)
        if n_over:
            logger.warning(
                "%s: %d/%d text(s) in this batch exceed max_seq_length=%d and "
                "will be silently truncated by the model.",
                self._model_name, n_over, len(texts), max_len,
            )

    def _apply_prefix(self, texts: list[str], is_query: bool) -> list[str]:
        prefix = self._query_prefix if is_query else self._document_prefix
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def embed(self, texts: list[str], is_query: bool = False,
              apply_prefix: bool = True) -> list[list[float]]:
        import torch
        self._load()

        # Length-checked/embedded AFTER the prefix is applied -- the prefix's
        # own tokens count against max_seq_length too (matters most for
        # Qwen3-Embedding's fairly long instruction-style query prefix).
        # apply_prefix=False (text-vs-text metrics): no prefix / prompt at all
        if apply_prefix:
            texts = self._apply_prefix(texts, is_query)
        self._record_length_check(texts)
        prompt_name = (self._query_prompt_name if is_query else self._document_prompt_name) if apply_prefix else None
        encode_kwargs = {"prompt_name": prompt_name} if prompt_name else {}

        optimal_batch_size = 256 if torch.cuda.is_available() else 32

        # First embed() is the first real GPU forward pass; log it.
        is_first_embed = not self._first_embed_done
        if is_first_embed:
            print(f"Embedding first batch of this process: {len(texts)} texts "
                  f"(sub-batched at {optimal_batch_size}) -- first GPU forward "
                  f"pass in this process, may be slower than subsequent batches...")
            embed_start = time.perf_counter()

        if torch.cuda.is_available():
            # autocast forces Tensor Core dispatch while keeping sensitive ops fp32.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = self._model.encode(
                    texts,
                    batch_size=optimal_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    **encode_kwargs,
                )
        else:
            embeddings = self._model.encode(
                texts,
                batch_size=optimal_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                **encode_kwargs,
            )

        if is_first_embed:
            print(f"First batch embedded in {time.perf_counter() - embed_start:.1f}s "
                  f"({len(texts) / max(time.perf_counter() - embed_start, 1e-9):.1f} texts/s) "
                  f"-- subsequent batches should be faster now that CUDA/HIP context "
                  f"and kernels are warmed up.")
            self._first_embed_done = True

        return embeddings.tolist()