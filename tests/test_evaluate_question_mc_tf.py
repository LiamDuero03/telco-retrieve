"""Tests for evaluate_question()'s mc/tf scoring branch.

Fix: mc/tf correctness used to be written into the OE metrics (faithfulness/
answer_relevancy/etc, with answer_relevancy hardcoded to 1.0). Now accuracy is
set directly and the four OE metrics stay at their -1.0 "not applicable" default.

search()/generate() are patched on pipeline_module since evaluate_question()
imports them locally — patching the module attribute affects that import at
call time.
"""
from __future__ import annotations

import open_telco_rag.core.pipeline as pipeline_module
from open_telco_rag.evaluation.evaluate import (
    _build_generation_question, _match_deterministic_answer, evaluate_question,
)
from open_telco_rag.evaluation.questions import EvalQuestion
from open_telco_rag.types import SearchResult


def _fake_chunks():
    return [SearchResult(text="The AMF handles registration.", distance=0.2,
                          metadata={"source": "23501-j00.md"}, doc_id="c1")]


def test_mc_correct_answer_sets_accuracy_and_leaves_oe_metrics_not_applicable(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.error == ""
    assert result.accuracy == 1.0
    assert result.faithfulness == -1.0
    assert result.answer_relevancy == -1.0
    assert result.context_recall == -1.0
    assert result.answer_correctness == -1.0


def test_mc_incorrect_answer_sets_accuracy_zero(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "A", "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.accuracy == 0.0


def test_tf_correctly_false_answer_sets_accuracy_one(monkeypatch):
    """Confirms exact-match scoring has no "always expects True" bias."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "False", "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.accuracy == 1.0
    assert result.faithfulness == -1.0


def test_open_question_still_computes_oe_metrics_not_accuracy(monkeypatch):
    """Confirms the fix didn't break the OE path — metrics still run with a
    real llm_call/embedder rather than crashing."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    def _fake_llm_call(prompt, max_tokens=1024):
        # Empty JSON array short-circuits faithfulness()/answer_relevancy() to 0.0.
        return "[]"

    class _FakeEmbedder:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    q = EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
    )

    result = evaluate_question(
        q, store=object(), embedder=_FakeEmbedder(), llm_call=_fake_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.error == ""
    assert result.accuracy == -1.0
    # Ran and returned 0.0 (empty-claims short circuit), not skipped/left at default.
    assert result.faithfulness == 0.0
    assert result.answer_relevancy == 0.0


# ── _match_deterministic_answer ──
# Regression: reasoning models explain before concluding ("No, this is incorrect
# because...") which pure exact-match/startswith logic never matched, causing
# mc/tf accuracy=0/120 across a real grid. Separate from parse_json()'s <think>-tag
# bug (test_claude_parse_json.py) — mc/tf scoring never calls the judge.

def test_tf_reasoning_style_no_answer_matches_false_reference():
    answer = ("No, this is not correct. According to the specification, the "
              "secMethods attribute is mandatory (M) and can contain 1 to N "
              "security methods, not up to 1 as stated.")
    assert _match_deterministic_answer(answer, "False", "tf") == "correct"


def test_tf_reasoning_style_yes_answer_matches_true_reference():
    answer = "Yes, this is correct. The specification confirms this requirement in clause 5.2."
    assert _match_deterministic_answer(answer, "True", "tf") == "correct"


def test_tf_reasoning_style_no_answer_does_not_match_true_reference():
    """Lenient matching must still catch real mismatches, not score every explanation 1.0."""
    answer = "No, this is not correct according to the specification."
    assert _match_deterministic_answer(answer, "True", "tf") == "incorrect"


def test_tf_does_not_spuriously_match_boolean_word_buried_mid_explanation():
    """Must not match on the word "false" buried in reasoning — avoids false positives."""
    answer = ("The specification does not explicitly state this is false; however, "
              "after careful analysis of clause 5.2, the correct determination is "
              "that the statement holds and is therefore accurate.")
    assert _match_deterministic_answer(answer, "False", "tf") == "incorrect"


def test_mc_reasoning_style_answer_concluding_with_the_letter_matches():
    answer = ("Looking at the retrieved context, the interface responsible for "
              "this function is the N2 interface, which corresponds to option B.")
    assert _match_deterministic_answer(answer, "B", "mc") == "correct"


def test_mc_reasoning_style_answer_concluding_with_a_different_letter_does_not_match():
    answer = "Based on the specification, this corresponds to option C."
    assert _match_deterministic_answer(answer, "B", "mc") == "incorrect"


def test_mc_bare_letter_prefix_still_matches_original_behavior():
    assert _match_deterministic_answer("B", "B", "mc") == "correct"
    assert _match_deterministic_answer("B) The E2 interface.", "B", "mc") == "correct"


def test_mc_the_answer_is_b_phrasing_matches():
    """Regression: [^A-Z0-9]{0,10} gap excluded "IS" (uppercase letters), so
    "The answer is B." never matched."""
    assert _match_deterministic_answer("The answer is B.", "B", "mc") == "correct"
    assert _match_deterministic_answer("I believe the correct option is C.", "C", "mc") == "correct"


def test_mc_the_answer_is_not_b_phrasing_does_not_spuriously_match():
    """The "IS" widening must not reopen the negation false-positive door
    ("answer is NOT B" must still fail)."""
    assert _match_deterministic_answer("The answer is not B, it is actually C.", "B", "mc") == "incorrect"


def test_tf_end_to_end_via_evaluate_question_with_reasoning_style_generator(monkeypatch):
    """Same shape as the confirmed incident, exercised through the full
    evaluate_question() path."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": ("No, the secMethods attribute is not optional -- the specification "
                   "marks it as mandatory (M) with 1 to N allowed values."),
        "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question='The "secMethods" attribute is optional and can contain up to 1 security method.',
        keywords=[], corpus="3gpp", answer_type="tf", reference_answer="False",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.accuracy == 1.0


# ── _build_generation_question ──
# Regression: mc/tf questions used the same generation prompt as open questions
# — q.options was never referenced, so answers never mentioned an option letter.

def test_build_generation_question_includes_options_and_letter_instruction_for_mc():
    q = EvalQuestion(
        question="Which interface handles this function?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2", "C": "N2"}, reference_answer="B",
    )
    prompt = _build_generation_question(q, q.question)

    assert "Which interface handles this function?" in prompt
    assert "A. N1" in prompt
    assert "B. E2" in prompt
    assert "C. N2" in prompt
    assert "letter" in prompt.lower()


def test_build_generation_question_includes_true_false_instruction_for_tf():
    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )
    prompt = _build_generation_question(q, q.question)

    assert "The AMF requires exactly 3 PLMN IDs." in prompt
    assert "boolean" in prompt
    assert "true" in prompt.lower() and "false" in prompt.lower()


def test_build_generation_question_leaves_open_questions_unchanged():
    q = EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration and mobility management.",
    )
    assert _build_generation_question(q, q.question) == q.question


def test_build_generation_question_falls_back_to_bare_question_when_mc_has_no_options():
    """Malformed mc question with no options must fall back to the bare question,
    not crash."""
    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options=None, reference_answer="B",
    )
    assert _build_generation_question(q, q.question) == q.question


def test_mc_end_to_end_generator_receives_options_not_the_bare_question(monkeypatch):
    """Confirms the generator receives options, and retrieval search() is not
    polluted with them."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())

    captured = {}

    def _fake_generate(question, results, source_type=None, generator=None, **kw):
        captured["question"] = question
        return {"answer": "The answer is B.", "sources": [], "chunks_used": 1}

    monkeypatch.setattr(pipeline_module, "generate", _fake_generate)

    q = EvalQuestion(
        question="Which interface handles this function?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert "A. N1" in captured["question"]
    assert "B. E2" in captured["question"]
    assert result.accuracy == 1.0


def test_tf_end_to_end_generator_receives_true_false_instruction(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())

    captured = {}

    def _fake_generate(question, results, source_type=None, generator=None, **kw):
        captured["question"] = question
        return {"answer": "False, this is incorrect.", "sources": [], "chunks_used": 1}

    monkeypatch.setattr(pipeline_module, "generate", _fake_generate)

    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert "boolean" in captured["question"]
    assert result.accuracy == 1.0


# ── Abstention ──
# Design: honest abstention is a different failure than confidently guessing wrong
# — collapsing both into "incorrect" would reward forced-guessing over honesty.

def test_build_generation_question_mc_explicitly_permits_abstention():
    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )
    prompt = _build_generation_question(q, q.question)
    assert "insufficient" in prompt.lower() or "does not contain enough information" in prompt.lower() \
        or "say so" in prompt.lower()
    assert "guess" in prompt.lower()  # explicitly discourages guessing


def test_build_generation_question_tf_explicitly_permits_abstention():
    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )
    prompt = _build_generation_question(q, q.question)
    assert "does not contain enough information" in prompt.lower()
    assert "guess" in prompt.lower()  # explicitly discourages guessing
    assert "abstained" in prompt.lower()


def test_match_deterministic_answer_classifies_mc_abstention():
    answer = "The provided context does not contain enough information to determine the correct option."
    assert _match_deterministic_answer(answer, "B", "mc") == "abstained"


def test_match_deterministic_answer_classifies_tf_abstention():
    answer = "I cannot determine whether this is true or false based on the given context."
    assert _match_deterministic_answer(answer, "False", "tf") == "abstained"


def test_match_deterministic_answer_commitment_is_checked_before_abstention_phrases():
    """SUPERSEDES an earlier version of this test that asserted the
    opposite priority (abstention-first). Real, confirmed production bug
    found by spot-checking 3gpp-r19_bge_parent_child_none_c512_o50's
    question_results directly: a real mc answer read "The answer is
    **A**.\\n\\nJustification: ... The other options (allocating
    additional IP addresses or a UE-initiated handover) are not mentioned
    as conditions that cause an Update Bearer Request." -- a clean,
    unambiguous commitment to A, whose justification explains why the
    OTHER options are wrong (completely normal mc reasoning) using the
    phrase "not mentioned", which is also on _is_not_found_answer_mc_tf()'s
    phrase list. With abstention checked first (the original ordering),
    this was misclassified as "abstained" despite clearly committing to a
    letter -- and because BOTH a correct commitment and a wrong one could
    trigger the same phrase in their own justification, this silently
    capped wrong_rate at 0.0 (suspiciously clean) across several real grid
    combinations rather than reflecting genuine wrong answers. Checking
    commitment FIRST fixes this: an abstention phrase appearing anywhere
    in an answer that has already committed to a letter is just normal
    reasoning prose about the other options, not a signal the generator
    didn't know the answer."""
    answer = ("The answer is **A**.\n\nJustification:\n"
              "The context states that the Presence Reporting Area Information IE is included in an "
              "Update Bearer Response only when the PGW has requested, in the corresponding Update Bearer "
              "Request, that the MME/SGSN start or modify reporting of UE presence in a Presence Reporting "
              "Area. Thus, a request to report a change of UE presence in a Presence Reporting Area triggers "
              "the sending of an Update Bearer Request (from PGW to S-GW, then to SGSN). The other options "
              "(allocating additional IP addresses or a UE-initiated handover) are not mentioned as conditions "
              "that cause an Update Bearer Request.")
    # Committed to A: correct when reference is A...
    assert _match_deterministic_answer(answer, "A", "mc") == "correct"
    # ...incorrect (not abstained) when reference differs — proves commitment is checked.
    assert _match_deterministic_answer(answer, "B", "mc") == "incorrect"


def test_match_deterministic_answer_still_abstains_when_no_commitment_is_present_at_all():
    """Genuine abstention (no commitment pattern) must still classify as
    abstained after the reordering."""
    answer = "The context does not contain enough information to determine which option is correct."
    assert _match_deterministic_answer(answer, "B", "mc") == "abstained"


def test_mc_end_to_end_abstained_answer_sets_outcome_and_zero_accuracy(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The context does not contain enough information to determine the answer.",
        "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.mc_tf_outcome == "abstained"
    assert result.accuracy == 0.0  # unchanged aggregate semantics -- see QuestionResult's own comment


def test_tf_end_to_end_abstained_answer_sets_outcome_and_zero_accuracy(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "I cannot determine this from the provided context.",
        "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.mc_tf_outcome == "abstained"
    assert result.accuracy == 0.0


def test_mc_end_to_end_correct_answer_sets_outcome_field_too(monkeypatch):
    """mc_tf_outcome must be populated on the success path too, not just abstained/incorrect."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "B", "sources": [], "chunks_used": 1,
    })

    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.mc_tf_outcome == "correct"


def test_open_question_mc_tf_outcome_stays_none(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration.", "sources": [], "chunks_used": 1,
    })

    def _fake_llm_call(prompt, max_tokens=1024):
        return "[]"

    class _FakeEmbedder:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    q = EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration.",
    )

    result = evaluate_question(
        q, store=object(), embedder=_FakeEmbedder(), llm_call=_fake_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert result.mc_tf_outcome is None


# ── End-to-end structured-output wiring ─────────────────────────────────

def test_mc_end_to_end_response_schema_reaches_generate_and_drives_the_outcome(monkeypatch):
    """Root-cause fix: response_schema must reach generate(), and the JSON answer
    is read directly — no regex, no phrase-list matching."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())

    captured = {}

    def _fake_generate(question, results, source_type=None, generator=None, response_schema=None):
        captured["question"] = question
        captured["response_schema"] = response_schema
        return {
            "answer": '{"verdict": "answered", "option": "B", "justification": "the context says so"}',
            "sources": [], "chunks_used": 1,
        }

    monkeypatch.setattr(pipeline_module, "generate", _fake_generate)

    q = EvalQuestion(
        question="Which interface handles this function?", keywords=[], corpus="3gpp",
        answer_type="mc", options={"A": "N1", "B": "E2"}, reference_answer="B",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert captured["response_schema"] is not None
    assert captured["response_schema"]["properties"]["option"]["enum"] == ["A", "B", None]
    assert result.mc_tf_outcome == "correct"
    assert result.accuracy == 1.0


def test_tf_end_to_end_structured_abstained_answer_sets_outcome(monkeypatch):
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())

    def _fake_generate(question, results, source_type=None, generator=None, response_schema=None):
        return {
            "answer": '{"verdict": "abstained", "boolean": null, "justification": "not enough info"}',
            "sources": [], "chunks_used": 1,
        }

    monkeypatch.setattr(pipeline_module, "generate", _fake_generate)

    q = EvalQuestion(
        question="The AMF requires exactly 3 PLMN IDs.", keywords=[], corpus="3gpp",
        answer_type="tf", reference_answer="False",
    )

    result = evaluate_question(q, store=object(), embedder=object(), rag_generator_name="local",
                                rag_model_name="local-model")

    assert result.mc_tf_outcome == "abstained"
    assert result.accuracy == 0.0


def test_open_question_response_schema_is_none(monkeypatch):
    """Confirms open questions get no structural constraint — response_schema=None."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())

    captured = {}

    def _fake_generate(question, results, source_type=None, generator=None, response_schema=None):
        captured["response_schema"] = response_schema
        return {"answer": "The AMF handles registration.", "sources": [], "chunks_used": 1}

    monkeypatch.setattr(pipeline_module, "generate", _fake_generate)

    def _fake_llm_call(prompt, max_tokens=1024):
        return "[]"

    class _FakeEmbedder:
        def embed(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    q = EvalQuestion(
        question="What does the AMF handle?", keywords=[], corpus="3gpp",
        answer_type="open", reference_answer="Registration.",
    )

    evaluate_question(
        q, store=object(), embedder=_FakeEmbedder(), llm_call=_fake_llm_call,
        rag_generator_name="local", rag_model_name="local-model",
    )

    assert captured["response_schema"] is None
