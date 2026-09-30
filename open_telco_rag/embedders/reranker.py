"""Cross-encoder and OTel rerankers for post-retrieval scoring."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from open_telco_rag.types import SearchResult

logger = logging.getLogger(__name__)

_RERANKER_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "reranker_models.json"


def load_reranker(model_key: str = "ms-marco-minilm"):
    """Load a reranker by key from configs/reranker_models.json (returns a
    CrossEncoderReranker or OTelReranker)."""
    if not _RERANKER_MODELS_FILE.exists():
        raise FileNotFoundError(f"Reranker models file not found: {_RERANKER_MODELS_FILE}")

    models = json.loads(_RERANKER_MODELS_FILE.read_text(encoding="utf-8"))

    if model_key not in models:
        raise ValueError(
            f"Unknown reranker key '{model_key}'. "
            f"Available: {list(models.keys())}"
        )

    cfg     = models[model_key]
    backend = cfg.get("backend", "cross-encoder")

    if backend == "otel":
        otel_kwargs = {}
        if "query_max_chars" in cfg:
            otel_kwargs["query_max_chars"] = cfg["query_max_chars"]
        if "doc_max_chars" in cfg:
            otel_kwargs["doc_max_chars"] = cfg["doc_max_chars"]
        return OTelReranker(model=cfg["model_name"], **otel_kwargs)
    else:
        return CrossEncoderReranker(model_name=cfg["model_name"])


# ── Standard cross-encoder reranker ──────────────────────────────────────────

class CrossEncoderReranker:
    """Reranks retrieved chunks using a standard cross-encoder model.

    Scores query+chunk jointly (unlike bi-encoders, which encode them
    separately) - more accurate but slower; use on a small candidate set.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model      = None  # lazy load
        self._load_lock  = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        # Double-checked locking: one reranker instance is shared across
        # worker threads, so the lazy load must be synchronised.
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install open-telco-rag[local-embed]"
                )
            print(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query:   str,
        results: list[SearchResult],
        top_k:   int = 5,
    ) -> list[SearchResult]:
        """Rerank results by cross-encoder score and return top_k."""
        if not results:
            return results

        self._load()

        pairs  = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result.metadata["rerank_score"] = round(float(score), 4)

        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r for r, _ in reranked[:top_k]]


# ── OTel reranker (Qwen3-based sequence-classification cross-encoder) ────────

class OTelReranker:
    """Telecom-domain reranker: Qwen3 sequence-classification cross-encoder.

    Models: farbodtavakkoli/OTel-Reranker-{0.6B,4B,8B} (0.6B recommended).
    device: 'cpu', 'cuda', or 'auto' (default).

    Loaded with AutoModelForSequenceClassification and read as one logit per
    pair (per the model's README) - not a causal LM prompted for yes/no.
    """

    def __init__(
        self,
        model:  str = "farbodtavakkoli/OTel-Reranker-0.6B",
        device: str = "auto",
        query_max_chars: int = 512,
        doc_max_chars:   int = 1024,
    ):
        self.model_name  = model
        self.device      = device
        self.query_max_chars = query_max_chars
        self.doc_max_chars   = doc_max_chars
        self._model      = None
        self._tokenizer  = None
        self._load_lock  = threading.Lock()
        self.truncated_count = 0
        self.scored_count    = 0

    @property
    def truncation_rate(self) -> float:
        """Fraction of documents scored so far that were cut down to
        doc_max_chars (or whose query was cut to query_max_chars). 0.0 if
        nothing has been scored yet."""
        return (self.truncated_count / self.scored_count) if self.scored_count else 0.0

    def _load(self):
        # Double-checked locking; see CrossEncoderReranker._load().
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            print(f"Loading OTel reranker: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # No device_map: this model fits on one device, and "auto" can
            # offload layers to CPU in mixed float32/float16 ("mat1 and mat2
            # must have the same dtype").
            resolved_device = (
                ("cuda" if torch.cuda.is_available() else "cpu")
                if self.device == "auto" else self.device
            )
            dtype = torch.float16 if resolved_device != "cpu" else torch.float32
            # device_map={"": device} (not .to()): accelerate materialises meta tensors properly.
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map={"": resolved_device},
            )
            model.eval()
            self._model = model  # assign last, after it's fully built

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        """P(relevant) (sigmoid of the single logit) for every (query,
        document) pair, in one batched forward pass."""
        import torch

        query_truncated = len(query) > self.query_max_chars
        if query_truncated:
            logger.debug(
                "OTelReranker: query truncated to query_max_chars=%d (was %d chars).",
                self.query_max_chars, len(query),
            )

        pairs = []
        for doc in documents:
            doc_truncated = len(doc) > self.doc_max_chars
            if doc_truncated:
                logger.debug(
                    "OTelReranker: document truncated to doc_max_chars=%d (was %d chars).",
                    self.doc_max_chars, len(doc),
                )
            if query_truncated or doc_truncated:
                self.truncated_count += 1
            self.scored_count += 1
            pairs.append([query[:self.query_max_chars], doc[:self.doc_max_chars]])

        inputs = self._tokenizer(
            pairs,
            padding        = True,
            truncation     = True,
            max_length     = 2048,
            return_tensors = "pt",
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)

        return torch.sigmoid(logits).tolist()

    def rerank(
        self,
        query:   str,
        results: list[SearchResult],
        top_k:   int = 5,
    ) -> list[SearchResult]:
        """Rerank results by telecom-domain relevance and return top_k."""
        if not results:
            return results

        self._load()

        scores = self._score_batch(query, [r.text for r in results])
        for r, score in zip(results, scores):
            r.metadata["rerank_score"] = round(float(score), 4)

        ranked = sorted(results, key=lambda r: r.metadata["rerank_score"], reverse=True)
        return ranked[:top_k]