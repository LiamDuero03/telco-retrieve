"""Tests for open_telco_rag.evaluation._claude.parse_json().

Regression: text before JSON (e.g. reasoning-model preamble, or the old
lstrip("```json") only stripping a character set, not a literal prefix) broke
json.loads(). Confirmed in production: gpt-oss-120b's preamble made
answer_relevancy/context_recall silently return 0.0 (the parse-failure
fallback), not a genuine score.
"""
from __future__ import annotations

import pytest

from open_telco_rag.evaluation._claude import parse_json


def test_reasoning_model_preamble_before_bare_json_array():
    raw = ('Let me think about this step by step. The answer discusses AMF registration.\n\n'
           '["What does AMF handle?", "What is registration?"]')
    assert parse_json(raw) == ["What does AMF handle?", "What is registration?"]


def test_reasoning_model_preamble_before_bare_json_object():
    raw = 'I will now count the facts.\n\n{"total": 3, "supported": 2}'
    assert parse_json(raw) == {"total": 3, "supported": 2}


def test_fenced_json_with_language_tag_still_works():
    raw = '```json\n["a", "b"]\n```'
    assert parse_json(raw) == ["a", "b"]


def test_bare_json_with_no_fence_or_preamble_still_works():
    raw = '["a", "b"]'
    assert parse_json(raw) == ["a", "b"]


def test_preamble_plus_fence_plus_trailing_commentary():
    raw = ('Sure, here is my analysis:\n\n'
           '```json\n{"correct": 2, "incorrect": 0, "missing": 1}\n```\n\n'
           'Let me know if you need anything else.')
    assert parse_json(raw) == {"correct": 2, "incorrect": 0, "missing": 1}


def test_fence_without_json_language_tag():
    raw = '```\n["x", "y", "z"]\n```'
    assert parse_json(raw) == ["x", "y", "z"]


def test_truly_unparseable_response_raises_a_clear_error():
    with pytest.raises((ValueError, Exception)):
        parse_json("This response contains no JSON at all, just prose.")


# ── <think> reasoning blocks (Qwen3-style) ──
# Qwen3-32B emits reasoning inline as <think>...</think> text, not a separate field.
# Confirmed in production: caused answer_relevancy/context_recall to silently return 0.0.
# (mc/tf accuracy=0/120 in the same run is an unrelated evaluate.py bug, not this.)

def test_think_block_with_no_incidental_brackets_still_worked_before_this_fix():
    """Confirms the bug is incidental brackets in reasoning, not <think> tags themselves."""
    raw = ('<think>\nLet me analyze this answer and generate some questions '
           'it could be responding to. The answer discusses AMF registration.\n</think>\n\n'
           '["What does AMF handle?", "What is registration?"]')
    assert parse_json(raw) == ["What does AMF handle?", "What is registration?"]


def test_think_block_containing_its_own_json_example_no_longer_corrupts_the_match():
    """Bug: greedy bracket-search spanned from a bracket inside <think> to the
    real JSON's closing bracket, corrupting the match."""
    raw = ('<think>\nI need to return a JSON array like ["q1", "q2", "q3"]. '
           'Let me think about what questions this answer addresses. The answer '
           'talks about {AMF} registration and the {5GC} network functions involved.\n'
           '</think>\n\n'
           '["What does AMF handle?", "What is registration?"]')
    assert parse_json(raw) == ["What does AMF handle?", "What is registration?"]


def test_think_block_with_json_object_and_incidental_braces_also_recovers():
    raw = ('<think>\nThe expected output shape is {"total": N, "supported": M}. '
           'Looking at the reference answer, I count 3 statements total.\n</think>\n\n'
           '{"total": 3, "supported": 2}')
    assert parse_json(raw) == {"total": 3, "supported": 2}


def test_unclosed_think_block_raises_a_specific_truncation_error_not_a_generic_one():
    """max_tokens exhausted mid-reasoning; error must name the actual fix
    (bigger budget), not a generic "could not find JSON" message."""
    raw = ('<think>\nLet me carefully consider this answer step by step. First, I need '
           'to identify the main topic. The answer discusses AMF registration procedures '
           'in the context of 5G core network functions and I should think about what')
    with pytest.raises(ValueError, match="truncated mid-reasoning"):
        parse_json(raw)


def test_fenced_json_after_a_closed_think_block_still_works():
    """<think> stripping must happen BEFORE fence-stripping, not instead
    of it -- a model can do both (reason, then fence its final JSON)."""
    raw = ('<think>\nSome reasoning here.\n</think>\n\n'
           '```json\n["a", "b"]\n```')
    assert parse_json(raw) == ["a", "b"]
