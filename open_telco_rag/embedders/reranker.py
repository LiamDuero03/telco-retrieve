"""Cross-encoder and OTel rerankers for post-retrieval scoring."""

from __future__ import annotations

import json
from pathlib import Path

from open_telco_rag.types import SearchResult

_RERANKER_MODELS_FILE = Path(__file__).parent.parent.parent / "configs" / "reranker_models.json"


def load_reranker(model_key: str = "ms-marco-minilm"):
    """
    Factory — load the right reranker from reranker_models.json.

    Args:
        model_key: key from configs/reranker_models.json,
                   e.g. 'ms-marco-minilm', 'otel-reranker-0.6b'

    Returns:
        CrossEncoderReranker or OTelReranker instance
    """
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
        return OTelReranker(model=cfg["model_name"])
    else:
        return CrossEncoderReranker(model_name=cfg["model_name"])


# ── Standard cross-encoder reranker ──────────────────────────────────────────

class CrossEncoderReranker:
    """Reranks retrieved chunks using a standard cross-encoder model.

    Scores query+chunk jointly (unlike bi-encoders, which encode them
    separately) — more accurate but slower; use on a small candidate set.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model      = None  # lazy load

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
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


# ── OTel reranker (Qwen3-based causal LM) ────────────────────────────────────

class OTelReranker:
    """Telecom-domain reranker: Qwen3-based causal LM, scores P(yes) from yes/no logits.

    Available models (via load_reranker() or direct):
        farbodtavakkoli/OTel-Reranker-0.6B  (recommended)
        farbodtavakkoli/OTel-Reranker-4B
        farbodtavakkoli/OTel-Reranker-8B

    Args:
        model:  HuggingFace model ID
        device: 'cpu', 'cuda', or 'auto' (default)
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "Judge whether the following document is relevant to the query. "
        "Answer with 'yes' or 'no'.<|im_end|>\n"
        "<|im_start|>user\n"
        "Query: {query}\n\n"
        "Document: {document}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(
        self,
        model:  str = "farbodtavakkoli/OTel-Reranker-0.6B",
        device: str = "auto",
    ):
        self.model_name = model
        self.device     = device
        self._model     = None
        self._tokenizer = None
        self._yes_id    = None
        self._no_id     = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading OTel reranker: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model     = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype      = torch.float16 if self.device != "cpu" else torch.float32,
                device_map = self.device,
            )
            self._model.eval()
            self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
            self._no_id  = self._tokenizer.encode("no",  add_special_tokens=False)[0]

    def _score(self, query: str, document: str) -> float:
        """Return P(yes) for a single query-document pair."""
        import torch
        import math

        prompt = self.PROMPT_TEMPLATE.format(
            query    = query[:512],
            document = document[:1024],
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 2048,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        yes_score   = last_logits[self._yes_id].item()
        no_score    = last_logits[self._no_id].item()

        exp_yes = math.exp(yes_score)
        exp_no  = math.exp(no_score)
        return exp_yes / (exp_yes + exp_no)

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

        for r in results:
            score = self._score(query, r.text)
            r.metadata["rerank_score"] = round(score, 4)

        ranked = sorted(results, key=lambda r: r.metadata["rerank_score"], reverse=True)
        return ranked[:top_k]