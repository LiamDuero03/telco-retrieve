"""Tests for the budgeted context assembly added to
open_telco_rag.core.retrieval (context_token_budget/candidate_pool/
budget_overflow) -- see fill_context_budget() and execute_retrieval().
"""
from __future__ import annotations

import tiktoken

from open_telco_rag.core.retrieval import (
    execute_retrieval,
    fill_context_budget,
    _resolve_parent_chunks,
)
from open_telco_rag.types import SearchResult

_ENC = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _result(text: str, doc_id: str, distance: float = 0.1, metadata: dict | None = None) -> SearchResult:
    return SearchResult(text=text, distance=distance, metadata=metadata or {}, doc_id=doc_id)


# ── fill_context_budget: overflow policies ──────────────────────────────────

def _candidates_of_known_sizes() -> list[SearchResult]:
    # Tokens are deterministic for repeated simple words under cl100k_base.
    return [
        _result("alpha " * 10, "c1"),   # ~10-11 tokens
        _result("bravo " * 10, "c2"),   # ~10-11 tokens
        _result("charlie " * 10, "c3"),  # ~10-11 tokens
    ]


def test_fill_context_budget_stop_halts_at_first_non_fitting_candidate():
    cands = _candidates_of_known_sizes()
    sizes = [_n_tokens(c.text) for c in cands]
    budget = sizes[0] + sizes[1] - 1  # fits candidate 0, not 0+1 together

    selected = fill_context_budget(cands, budget, overflow="stop")

    assert [c.doc_id for c in selected] == ["c1"]
    assert sum(_n_tokens(c.text) for c in selected) <= budget


def test_fill_context_budget_skip_continues_past_an_oversized_candidate():
    big = _result("delta " * 50, "big")     # too big to fit alone
    small = _result("echo " * 3, "small")   # fits after skipping `big`
    cands = [big, small]
    budget = _n_tokens(small.text) + 2

    selected = fill_context_budget(cands, budget, overflow="skip")

    assert [c.doc_id for c in selected] == ["small"]
    assert sum(_n_tokens(c.text) for c in selected) <= budget


def test_fill_context_budget_truncate_cuts_the_last_chunk_to_fill_exactly():
    cands = [_result("foxtrot " * 20, "c1")]
    budget = 5

    selected = fill_context_budget(cands, budget, overflow="truncate")

    assert len(selected) == 1
    assert _n_tokens(selected[0].text) <= budget
    assert _n_tokens(selected[0].text) > 0


def test_fill_context_budget_never_exceeds_budget_for_every_policy():
    cands = _candidates_of_known_sizes()
    budget = 15
    for policy in ("stop", "skip", "truncate"):
        selected = fill_context_budget(cands, budget, overflow=policy)
        assert sum(_n_tokens(c.text) for c in selected) <= budget, policy


def test_fill_context_budget_unknown_policy_raises():
    import pytest
    with pytest.raises(ValueError):
        fill_context_budget(_candidates_of_known_sizes(), 100, overflow="bogus")


def test_fill_context_budget_warns_when_candidate_pool_is_exhausted(caplog):
    import logging
    cands = _candidates_of_known_sizes()  # ~30 tokens total, well under budget
    with caplog.at_level(logging.WARNING, logger="open_telco_rag.core.retrieval"):
        selected = fill_context_budget(cands, budget=1000, overflow="stop")
    assert len(selected) == len(cands)
    assert any("exhausted" in rec.message for rec in caplog.records)


# ── parent_child dedup composed with budget filling ─────────────────────────

def test_resolve_parents_then_budget_keeps_best_ranked_parent_once():
    parent_text = "Parent context. " * 5
    children = [
        _result("child A text", "child_a", metadata={"parent_id": "P1", "full_parent_text": parent_text}),
        _result("child B text", "child_b", metadata={"parent_id": "P1", "full_parent_text": parent_text}),
        _result("other parent text " * 3, "child_c", metadata={"parent_id": "P2", "full_parent_text": "Other parent. " * 5}),
    ]

    resolved = _resolve_parent_chunks(children)
    # Only the FIRST (best-ranked) occurrence of each parent_id survives.
    assert [r.doc_id for r in resolved] == ["child_a", "child_c"]
    assert resolved[0].text == parent_text

    budget = _n_tokens(parent_text) + 2
    selected = fill_context_budget(resolved, budget, overflow="stop")
    assert sum(_n_tokens(c.text) for c in selected) <= budget


# ── execute_retrieval: budget unset reproduces the old fixed-top_k path ─────

class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    """Minimal VectorStore stub: dense search returns a fixed, pre-ranked list."""

    def __init__(self, results: list[SearchResult]):
        self._results = results

    def search(self, embedding, n_results=5):
        return self._results[:n_results]


def test_execute_retrieval_budget_unset_matches_old_fixed_top_k_behaviour():
    results = [_result(f"chunk {i} text", f"c{i}") for i in range(10)]
    store = _FakeStore(results)

    out = execute_retrieval("query", store, _FakeEmbedder(), top_k=3, search_method="dense")

    assert [r.doc_id for r in out] == ["c0", "c1", "c2"]


def test_execute_retrieval_with_budget_caps_total_tokens():
    results = [_result("token word " * 20, f"c{i}") for i in range(10)]
    store = _FakeStore(results)
    budget = 50

    out = execute_retrieval(
        "query", store, _FakeEmbedder(), search_method="dense",
        context_token_budget=budget, candidate_pool=10,
    )

    assert sum(_n_tokens(r.text) for r in out) <= budget
