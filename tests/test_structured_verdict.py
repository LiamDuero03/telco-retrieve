"""Tests for the structured, schema-constrained mc/tf verdict machinery,
which replaces free-text prose parsing (see evaluate.py's
_match_deterministic_answer_prose() docstring for the regex-bug history).

_match_deterministic_answer() tries the structured path first
(_parse_structured_verdict() + _classify_structured_verdict()), falling back
to prose parsing only when the answer doesn't match the structured shape.
These tests cover the new structured path; existing prose tests in
test_evaluate_question_mc_tf.py still exercise the fallback unchanged.
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
    """Enum options come from the actual dict, not a fixed A-D cap."""
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
    """Malformed mc question (no options dict) must not crash; falls back
    like open questions."""
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
    """Free text is never valid JSON, so it's routed unambiguously to the
    prose fallback."""
    assert _parse_structured_verdict("The answer is B.", "mc") is None


def test_parse_structured_verdict_returns_none_for_invalid_verdict_value():
    raw = json.dumps({"verdict": "maybe", "option": "B", "justification": "x"})
    assert _parse_structured_verdict(raw, "mc") is None


def test_parse_structured_verdict_returns_none_when_required_key_missing_for_answer_type():
    """A tf-shaped object given for an mc question must fall through to
    prose parsing, not misread the wrong field."""
    raw = json.dumps({"verdict": "answered", "boolean": True, "justification": "x"})
    assert _parse_structured_verdict(raw, "mc") is None


def test_parse_structured_verdict_returns_none_for_json_array_not_object():
    assert _parse_structured_verdict('["answered", "B"]', "mc") is None


def test_parse_structured_verdict_returns_none_for_malformed_json():
    assert _parse_structured_verdict('{"verdict": "answered", "option": ', "mc") is None


# ── _parse_structured_verdict - tf "boolean" type guard ──────────────────
#
# Regression: the schema declares boolean as JSON boolean-or-null, but a
# model/server can still emit the JSON STRING "false" instead of the JSON
# literal false. bool("false") is True (any non-empty string is truthy),
# which used to silently flip a correct "False" answer to "incorrect".

def test_parse_structured_verdict_normalizes_string_false_to_real_bool():
    raw = json.dumps({"verdict": "answered", "boolean": "false", "justification": "x"})
    parsed = _parse_structured_verdict(raw, "tf")
    assert parsed["boolean"] is False


def test_parse_structured_verdict_normalizes_string_true_to_real_bool():
    raw = json.dumps({"verdict": "answered", "boolean": "TRUE", "justification": "x"})  # case-insensitive
    parsed = _parse_structured_verdict(raw, "tf")
    assert parsed["boolean"] is True


def test_parse_structured_verdict_rejects_an_unrecognized_boolean_string():
    """Not "true"/"false" -- can't be trusted enough to normalize; falls
    through to prose parsing on the raw text instead of guessing."""
    raw = json.dumps({"verdict": "answered", "boolean": "maybe", "justification": "x"})
    assert _parse_structured_verdict(raw, "tf") is None


def test_parse_structured_verdict_rejects_a_non_string_non_bool_boolean():
    raw = json.dumps({"verdict": "answered", "boolean": 1, "justification": "x"})
    assert _parse_structured_verdict(raw, "tf") is None


def test_parse_structured_verdict_still_accepts_real_json_null_boolean():
    """Null is a legitimate schema value (abstain signal), not something
    the new type guard should reject."""
    raw = json.dumps({"verdict": "answered", "boolean": None, "justification": "x"})
    parsed = _parse_structured_verdict(raw, "tf")
    assert parsed["boolean"] is None


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
    """Inconsistent verdict=answered+no option must not silently score
    as "incorrect"."""
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


# ── _match_deterministic_answer - structured path takes priority ────────

def test_match_deterministic_answer_uses_structured_path_when_available():
    """Confirms structured JSON is read directly, not via the prose
    fallback (which would coincidentally also work here)."""
    raw = json.dumps({"verdict": "answered", "option": "A", "justification": "the specification states X"})
    assert _match_deterministic_answer(raw, "A", "mc") == "correct"
    assert _match_deterministic_answer(raw, "B", "mc") == "incorrect"


def test_match_deterministic_answer_structured_path_immune_to_the_justification_text_bug():
    """Regression: a justification phrase like "not mentioned" describing
    other options must not cause misclassification when option/verdict
    are read directly instead of pattern-matched."""
    raw = json.dumps({
        "verdict": "answered", "option": "A",
        "justification": ("The other options (allocating additional IP addresses or a "
                           "UE-initiated handover) are not mentioned as conditions that "
                           "cause an Update Bearer Request."),
    })
    assert _match_deterministic_answer(raw, "A", "mc") == "correct"
    assert _match_deterministic_answer(raw, "B", "mc") == "incorrect"


def test_match_deterministic_answer_falls_back_to_prose_when_not_structured():
    """Confirms the fallback works via the public entry point when
    response_format: json_schema isn't honored."""
    assert _match_deterministic_answer("The answer is B.", "B", "mc") == "correct"


def test_match_deterministic_answer_string_false_is_not_treated_as_true():
    """The regression this whole fix targets, exercised end to end: a
    genuinely correct "False" answer, delivered as the JSON string "false"
    instead of the JSON literal false, must score "correct" -- not
    "incorrect" via bool("false") == True."""
    raw = json.dumps({"verdict": "answered", "boolean": "false", "justification": "the spec says otherwise"})
    assert _match_deterministic_answer(raw, "False", "tf") == "correct"
    assert _match_deterministic_answer(raw, "True", "tf") == "incorrect"
