"""Resolving a child chunk to its parent (--resolve-parents) must keep the span
offsets consistent with the text that is actually handed to the generator."""
from __future__ import annotations

from open_telco_rag.core.chunking import chunk_document
from open_telco_rag.core.retrieval import _resolve_parent_chunks
from open_telco_rag.evaluation.metrics.retrieval import span_hit_at_k
from open_telco_rag.types import SearchResult

DOC = "\n\n".join(
    f"## Section {i}\n\n" + "\n\n".join(
        f"Clause {i}.{j}: the gNB performs procedure {i}{j} and reports it over the Xn interface." for j in range(1, 6))
    for i in range(1, 4))


def _children(**kw):
    return chunk_document(DOC, "parent_child", **kw)


def test_token_mode_children_carry_their_parents_span():
    children = _children(target_tokens=30, max_tokens=40)
    assert children
    for c in children:
        m = c.metadata
        assert DOC[m["parent_char_start"]:m["parent_char_end"]] == m["full_parent_text"]
        assert m["parent_char_start"] <= c.char_start and c.char_end <= m["parent_char_end"]
        assert m["parent_n_tokens"] >= c.n_tokens


def test_default_mode_children_have_no_parent_span_keys():
    children = _children()
    assert children and all("parent_char_start" not in c.metadata for c in children)


def _result(text, meta, doc_id="a.md::m::0"):
    return SearchResult(text=text, distance=0.1, metadata=meta, doc_id=doc_id)


def test_resolution_adopts_the_parents_offsets():
    child = _result("child", {"parent_id": "P1", "full_parent_text": "parent text", "char_start": 10,
                              "char_end": 20, "n_tokens": 3, "parent_char_start": 0, "parent_char_end": 100,
                              "parent_n_tokens": 30})
    (resolved,) = _resolve_parent_chunks([child])
    assert resolved.text == "parent text"
    assert (resolved.metadata["char_start"], resolved.metadata["char_end"], resolved.metadata["n_tokens"]) == (0, 100, 30)
    assert child.metadata["char_start"] == 10          # the original result is not mutated


def test_resolution_without_parent_offsets_is_unchanged():
    child = _result("child", {"parent_id": "P1", "full_parent_text": "parent text", "char_start": 10, "char_end": 20})
    (resolved,) = _resolve_parent_chunks([child])
    assert (resolved.metadata["char_start"], resolved.metadata["char_end"]) == (10, 20)
    plain = _result("no parent", {"char_start": 1, "char_end": 2})
    assert _resolve_parent_chunks([plain]) == [plain]


def test_sibling_child_whose_parent_contains_the_evidence_is_a_span_hit_after_resolution():
    children = _children(target_tokens=30, max_tokens=40)
    parent = children[0].metadata
    evidence = (parent["parent_char_start"] + 5, parent["parent_char_start"] + 15)
    sibling = next(c for c in children
                   if c.metadata["parent_id"] == children[0].metadata["parent_id"]
                   and not (c.char_start < evidence[1] and evidence[0] < c.char_end))
    as_result = _result(sibling.text, {**sibling.metadata, "char_start": sibling.char_start,
                                       "char_end": sibling.char_end, "n_tokens": sibling.n_tokens},
                        doc_id="doc.md::m::9")
    assert span_hit_at_k([as_result], *evidence, "doc.md") == 0            # child alone: miss
    assert span_hit_at_k(_resolve_parent_chunks([as_result]), *evidence, "doc.md") == 1
