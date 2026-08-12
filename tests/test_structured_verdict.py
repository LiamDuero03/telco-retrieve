"""Tests for the structured, schema-constrained mc/tf verdict machinery —
the root-cause fix replacing free-text prose parsing (three separate regex
bugs found and patched in one session; see evaluate.py's
_match_deterministic_answer_prose() docstring for the full history).

_match_deterministic_answer() is the actual entry point: it tries the
structured path (_parse_structured_verdict() + _classify_structured_verdict())
first, and ONLY falls back to prose parsing when the answer isn't (or
doesn't fully match) the expected structured shape. Since prose input is
never valid JSON, every existing prose-based test in
test_evaluate_question_mc_tf.py continues to exercise the fallback path
unchanged — these tests cover the NEW structured path specifically.
"""
from __future__ import annotations

import json

from open_telco_rag.evaluation.evaluate import (
    _classify_structured_verdict,
    _match_deterministic_answer,
    _mc_response_schema,
    _parse_structured_verdict,
    _response_schema_for,
    _tf_response_schema,
)
from open_telco_rag.evaluation.questions import EvalQuestion


# ── Schema builders ──────────────────────────────────────────────────────

def test_mc_response_schema_enum_is_built_from_the_actual_options_not_a_fixed_list():
    """A 3-option question must not have "D" as a valid choice, and a
    5-option question must not be artificially capped at 4."""
    schema = _mc_response_schema({"A": "N1", "B": "E2", "C": "N2"})
    assert schema["properties"]["option"]["enum"] == ["A", "B", "C", None]

    schema5 = _mc_response_schema({"A": "x", "B": "x", "C": "x", "D": "x", "E": "x"})
    assert schema5["properties"]["option"]["enum"] == ["A", "B", "C", "D", "E", None]


def test_mc_response_schema_requires_all_three_fields():
    schema = _mc_response_schema({"A": "x", "B": "y"})
    assert set(schema["required"]) == {"verdict", "option", "justification"}
    assert schema["additionalProperties"] is False


def test_tf_response_schema_shape():
    schema = _tf_response_schema()
    assert schema["properties"]["boolean"]["type"] == ["boolean", "null"]
    assert set(schema["required"]) == {"verdict", "boolean", "justification"}


def test_response_schema_for_returns_none_for_open_questions():
    q = EvalQuestion(question="q", keywords=[], corpus="3gpp", answer_type="open")
    assert _response_schema_for(q) is None


def test_response_schema_for_mc_and_tf():
    mc_q = EvalQuestion(question="q", keywords=[], corpus="3gpp", answer_type="mc",
                         options={"A": "x", "B": "y"})
    assert _response_schema_for(mc_q)["properties"]["option"]["enum"] == ["A", "B", None]

    tf_q = EvalQuestion(question="q", keywords=[], corpus="3gpp", answer_type="tf")
    assert "boolean" in _response_schema_for(tf_q)["properties"]


def test_response_schema_for_mc_without_options_falls_back_to_none():
    """A malformed mc question with no options dict must not crash --
    just no structural constraint, same as open questions."""
    q = EvalQuestion(question="q", keywords=[], corpus="3gpp", answer_type="mc", options=None)
    assert _response_schema_for(q) is None


# ── _parse_structured_verdict ────────────────────────────────────────────

def test_parse_structured_verdict_accepts_well_formed_mc_json():
    raw = json.dumps({"verdict": "answered", "option": "B", "justification": "because X"})
    parsed = _parse_structured_verdict(raw, "mc")
    assert parsed == {"verdict": "answered", "option": "B", "justification": "because X"}


def test_parse_structured_verdict_accepts_well_formed_tf_json():
    raw = json.dumps({"verdict": "answered", "boolean": False, "justification": "because X"})
    parsed = _parse_structured_verdict(raw, "tf")
    assert parsed == {"verdict": "answered", "boolean": False, "justification": "because X"}


def test_parse_structured_verdict_returns_none_for_plain_prose():
    """The exact property that makes the prose fallback safe: free text
    is never valid JSON, so it's unambiguously routed to the fallback."""
    assert _parse_structured_verdict("The answer is B.", "mc") is None


def test_parse_structured_verdict_returns_none_for_invalid_verdict_value():
    raw = json.dumps({"verdict": "maybe", "option": "B", "justification": "x"})
    assert _parse_structured_verdict(raw, "mc") is None


def test_parse_structured_verdict_returns_none_when_required_key_missing_for_answer_type():
    """A tf-shaped object (has "boolean") presented for an mc question
    (needs "option") must fall through to prose parsing, not silently
    misread the wrong field."""
    raw = json.dumps({"verdict": "answered", "boolean": True, "justification": "x"})
    assert _parse_structured_verdict(raw, "mc") is None


def test_parse_structured_verdict_returns_none_for_json_array_not_object():
    assert _parse_structured_verdict('["answered", "B"]', "mc") is None


def test_parse_structured_verdict_returns_none_for_malformed_json():
    assert _parse_structured_verdict('{"verdict": "answered", "option": ', "mc") is None


# ── _classify_structured_verdict ─────────────────────────────────────────

def test_classify_structured_verdict_mc_correct():
    obj = {"verdict": "answered", "option": "B", "justification": "x"}
    assert _classify_structured_verdict(obj, "B", "mc") == "correct"


def test_classify_structured_verdict_mc_incorrect():
    obj = {"verdict": "answered", "option": "A", "justification": "x"}
    assert _classify_structured_verdict(obj, "B", "mc") == "incorrect"


def test_classify_structured_verdict_mc_abstained():
    obj = {"verdict": "abstained", "option": None, "justification": "insufficient context"}
    assert _classify_structured_verdict(obj, "B", "mc") == "abstained"


def test_classify_structured_verdict_mc_answered_but_option_null_is_defensively_abstained():
    """Internally inconsistent response (verdict=answered but no option)
    -- shouldn't happen under a properly enforced schema, but must not
    silently score "incorrect" against a genuinely absent answer."""
    obj = {"verdict": "answered", "option": None, "justification": "x"}
    assert _classify_structured_verdict(obj, "B", "mc") == "abstained"


def test_classify_structured_verdict_tf_correct_true():
    obj = {"verdict": "answered", "boolean": True, "justification": "x"}
    assert _classify_structured_verdict(obj, "True", "tf") == "correct"


def test_classify_structured_verdict_tf_correct_false():
    obj = {"verdict": "answered", "boolean": False, "justification": "x"}
    assert _classify_structured_verdict(obj, "False", "tf") == "correct"


def test_classify_structured_verdict_tf_incorrect():
    obj = {"verdict": "answered", "boolean": True, "justification": "x"}
    assert _classify_structured_verdict(obj, "False", "tf") == "incorrect"


def test_classify_structured_verdict_tf_abstained():
    obj = {"verdict": "abstained", "boolean": None, "justification": "insufficient context"}
    assert _classify_structured_verdict(obj, "True", "tf") == "abstained"


def test_classify_structured_verdict_tf_answered_but_boolean_null_is_defensively_abstained():
    obj = {"verdict": "answered", "boolean": None, "justification": "x"}
    assert _classify_structured_verdict(obj, "True", "tf") == "abstained"


# ── _match_deterministic_answer — structured path takes priority ────────

def test_match_deterministic_answer_uses_structured_path_when_available():
    """No regex, no phrase-list matching involved -- confirms structured
    JSON input is read directly rather than falling through to prose
    parsing (which would also happen to work for this particular input,
    but for the wrong reason)."""
    raw = json.dumps({"verdict": "answered", "option": "A", "justification": "the specification states X"})
    assert _match_deterministic_answer(raw, "A", "mc") == "correct"
    assert _match_deterministic_answer(raw, "B", "mc") == "incorrect"


def test_match_deterministic_answer_structured_path_immune_to_the_justification_text_bug():
    """The exact real bug this whole redesign was motivated by: a
    justification that explains away the OTHER options using a phrase
    like "not mentioned" must NOT cause a misclassification when the
    verdict/option fields are read directly instead of pattern-matched
    out of the prose."""
    raw = json.dumps({
        "verdict": "answered", "option": "A",
        "justification": ("The other options (allocating additional IP addresses or a "
                           "UE-initiated handover) are not mentioned as conditions that "
                           "cause an Update Bearer Request."),
    })
    assert _match_deterministic_answer(raw, "A", "mc") == "correct"
    assert _match_deterministic_answer(raw, "B", "mc") == "incorrect"


def test_match_deterministic_answer_falls_back_to_prose_when_not_structured():
    """Confirms the fallback still works end to end through the public
    entry point, not just via _match_deterministic_answer_prose()
    directly -- a server/model that doesn't honor response_format:
    json_schema must not simply break."""
    assert _match_deterministic_answer("The answer is B.", "B", "mc") == "correct"
