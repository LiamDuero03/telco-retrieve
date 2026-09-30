"""The tokenizer used by token chunking and by budgeted retrieval is created lazily
and patchable through tiktoken.get_encoding, whichever module counts the tokens."""
from __future__ import annotations

import subprocess
import sys

import tiktoken

from open_telco_rag.core import chunking, retrieval
from open_telco_rag.core.chunking import chunk_document
from open_telco_rag.core.retrieval import fill_context_budget
from open_telco_rag.types import SearchResult

DOC = "First paragraph about the E2 interface.\n\nSecond paragraph about the O1 interface.\n\n" * 6


def _reset_caches(monkeypatch):
    for module in (chunking, retrieval):
        if hasattr(module, "_ENCODING_CACHE"):
            monkeypatch.setattr(module, "_ENCODING_CACHE", {})


def _candidates(n=5):
    return [SearchResult(text=f"chunk number {i} " * 6, distance=0.1 * i, metadata={}, doc_id=f"c{i}")
            for i in range(n)]


def test_importing_both_modules_does_not_initialize_the_tokenizer():
    code = ("import tiktoken;tiktoken.get_encoding = lambda *a, **k: 1 / 0;"
            "import open_telco_rag.core.chunking, open_telco_rag.core.retrieval")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_character_chunking_and_default_selection_never_ask_for_a_tokenizer(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tiktoken.get_encoding called on a default path")

    _reset_caches(monkeypatch)
    monkeypatch.setattr(tiktoken, "get_encoding", boom)
    assert chunk_document(DOC, "text_baseline", chunk_size=300, overlap=0)


def test_token_chunking_and_budget_fill_use_the_patched_encoding(monkeypatch):
    real = tiktoken.get_encoding
    calls = []

    def counting(name):
        calls.append(name)
        return real(name)

    _reset_caches(monkeypatch)
    monkeypatch.setattr(tiktoken, "get_encoding", counting)
    chunks = chunk_document(DOC, "text_baseline", target_tokens=30)
    kept = fill_context_budget(_candidates(), budget=40)
    chunk_document(DOC, "text_baseline", target_tokens=30)
    fill_context_budget(_candidates(), budget=40)

    assert chunks and kept
    assert set(calls) == {"cl100k_base"}
    assert calls == ["cl100k_base"], "chunking and retrieval share one lazily created encoding"


def test_retrieval_counts_tokens_with_the_chunking_helper():
    assert retrieval._count_tokens is chunking._count_tokens
    assert retrieval._get_encoding is chunking._get_encoding


def test_a_fake_tokenizer_changes_the_token_counts(monkeypatch):
    class OneTokenPerWord:
        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens)

    _reset_caches(monkeypatch)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: OneTokenPerWord())
    kept = fill_context_budget(_candidates(), budget=10)     # 3 words x 6 = 18 words per candidate
    assert kept == [], "the first candidate alone (18 fake tokens) exceeds a 10-token budget"
