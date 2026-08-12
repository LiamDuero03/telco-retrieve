"""Tests for open_telco_rag.evaluation._claude.parse_json() — a real,
confirmed-in-production bug: the previous implementation only handled a
response that literally STARTED with a markdown code fence
(str.lstrip("```json"), which is also its own separate bug — lstrip's
argument is a character SET, not a literal prefix). Any text BEFORE the
JSON — e.g. a reasoning-style model's chain-of-thought preamble — made
json.loads() fail immediately, since none of faithfulness()/
answer_relevancy()/context_recall()/answer_correctness()'s prompts
request response_format=json_object.

Confirmed against a real pilot run: openai/gpt-oss-120b (used as both the
RAG generator and eval-metrics judge) produced exactly this failure
pattern — faithfulness/answer_correctness scored normally (those
particular LLM calls happened to return cleanly parseable responses),
while answer_relevancy/context_recall came back EXACTLY 0.0 for every
open question — the literal fallback both metric functions return on any
parse failure, not a genuine near-zero score.
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


# ── <think> reasoning blocks (Qwen3-style) — real, confirmed-in-production ──
#
# A SEPARATE failure mode from gpt-oss-120b's plain-prose preamble above:
# Qwen/Qwen3-32B (used as the eval-metrics judge) commonly emits its
# reasoning INLINE as literal <think>...</think> text in the response
# content (rather than in a separate field a server-side reasoning parser
# would split out). Confirmed against real aggregated grid results:
# answer_relevancy scored exactly 0.0 for all 120/120 open questions across
# every embedding model, with zero variance, while context_recall's judge
# call and answer_correctness's F1 judge call were ALSO silently failing
# (masked in answer_correctness's case by its 50%-weighted raw-embedding-
# similarity fallback, which doesn't depend on the judge at all — see
# metrics/ground_truth.py's answer_correctness()). mc/tf accuracy=0/120
# across the same run is a SEPARATE, unrelated bug (evaluate.py's exact-
# match scoring against the RAG generator's raw prose answer) — not
# something parse_json() is involved in at all.

def test_think_block_with_no_incidental_brackets_still_worked_before_this_fix():
    """A <think> block that happens not to mention any bracket/brace
    syntax already parsed fine even before this fix -- confirming the
    bug is specifically about incidental JSON-like syntax inside the
    reasoning, not <think> tags on their own."""
    raw = ('<think>\nLet me analyze this answer and generate some questions '
           'it could be responding to. The answer discusses AMF registration.\n</think>\n\n'
           '["What does AMF handle?", "What is registration?"]')
    assert parse_json(raw) == ["What does AMF handle?", "What is registration?"]


def test_think_block_containing_its_own_json_example_no_longer_corrupts_the_match():
    """Real, reproduced bug: every one of these prompts explicitly asks
    the model to "return a JSON array/object", so the model's own
    <think> reasoning very often talks about that literal format -- e.g.
    mentioning an example array or referring to a term in curly braces.
    Before this fix, the greedy bracket-search fallback matched from the
    FIRST bracket INSIDE the <think> block through the LAST bracket at
    the very end (the real JSON's own closing bracket), producing a
    mashed-together blob that raised JSONDecodeError("Extra data: ...")
    instead of ever reaching the actual JSON that follows cleanly."""
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
    """Real, plausible failure mode: max_tokens exhausted before the model
    ever finished reasoning, let alone emitted any JSON. Must raise a
    message that points at the actual fix (a bigger max_tokens budget for
    this call), not the generic "could not find JSON" message -- so this
    is diagnosable from the exception text alone with DEBUG off."""
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
