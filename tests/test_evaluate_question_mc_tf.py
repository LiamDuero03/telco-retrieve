"""Tests for evaluate_question()'s mc/tf scoring branch — the real design
fix: mc/tf exact-match correctness used to be written into faithfulness/
answer_relevancy/context_recall/answer_correctness (OE-specific metrics,
with answer_relevancy even hardcoded to 1.0 regardless of anything). These
confirm the fix directly: accuracy is now set, and the four OE metrics are
left at their -1.0 "not applicable" default for mc/tf questions.

search()/generate() are patched at their source module
(open_telco_rag.core.pipeline) since evaluate_question() imports them
locally inside the function body — patching the module attribute affects
that local import at call time.
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
    """A correctly-labeled FALSE answer is a CORRECT tf item, not a
    mismatch -- confirms exact-match scoring (not some "always expects
    True" bias)."""
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
    """Confirms the fix didn't accidentally break the OE path — accuracy
    stays "not applicable" and faithfulness/answer_relevancy still
    actually run (using the real metric functions, so this also proves
    they still receive a genuine llm_call/embedder rather than crashing)."""
    monkeypatch.setattr(pipeline_module, "search", lambda *a, **kw: _fake_chunks())
    monkeypatch.setattr(pipeline_module, "generate", lambda *a, **kw: {
        "answer": "The AMF handles registration and mobility management.",
        "sources": [], "chunks_used": 1,
    })

    def _fake_llm_call(prompt, max_tokens=1024):
        # decompose_claims / verify_claims / reverse_questions all expect
        # a JSON array back -- return something trivially valid+empty so
        # faithfulness()/answer_relevancy() short-circuit to 0.0 rather
        # than raising, without needing a real LLM.
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
    # faithfulness/answer_relevancy ran (returned 0.0 from the empty-claims
    # short circuit above) rather than being skipped/left at their default.
    assert result.faithfulness == 0.0
    assert result.answer_relevancy == 0.0


# ── _match_deterministic_answer — real, confirmed-in-production bug ────────
#
# Aggregating a real 24-combination grid's all_results.json showed mc/tf
# accuracy at exactly 0/120 in EVERY combination, zero variance across 6
# very different embedding models, with several manually-inspected answers
# confirmed substantively correct on reading. Root cause: the RAG generator
# (a reasoning-style model, e.g. openai/gpt-oss-120b) routinely explains
# before concluding -- "No, this is incorrect because..." for a tf question
# whose reference_answer is "False" -- which the original pure exact-match/
# startswith/"ANSWER IS X"-substring logic could never match, regardless of
# retrieval quality. This is a COMPLETELY SEPARATE bug from the judge-side
# parse_json() <think>-tag issue (see test_claude_parse_json.py) -- mc/tf
# scoring never calls the judge or parse_json() at all, only the RAG
# generator's own raw answer text.

def test_tf_reasoning_style_no_answer_matches_false_reference():
    answer = ("No, this is not correct. According to the specification, the "
              "secMethods attribute is mandatory (M) and can contain 1 to N "
              "security methods, not up to 1 as stated.")
    assert _match_deterministic_answer(answer, "False", "tf") == "correct"


def test_tf_reasoning_style_yes_answer_matches_true_reference():
    answer = "Yes, this is correct. The specification confirms this requirement in clause 5.2."
    assert _match_deterministic_answer(answer, "True", "tf") == "correct"


def test_tf_reasoning_style_no_answer_does_not_match_true_reference():
    """The lenient matching must still be a REAL mismatch check, not
    "any explanation scores 1.0" -- a "No" answer against a "True"
    reference is genuinely wrong."""
    answer = "No, this is not correct according to the specification."
    assert _match_deterministic_answer(answer, "True", "tf") == "incorrect"


def test_tf_does_not_spuriously_match_boolean_word_buried_mid_explanation():
    """"False" reference must not match an answer that merely CONTAINS the
    word "false" somewhere deep in its reasoning without that being the
    actual leading conclusion -- avoids the opposite failure mode (false
    positives) while fixing the false negatives."""
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
    """Regression test for a real bug in the matcher fix itself: the
    original explicit-phrasing regex's [^A-Z0-9]{0,10} gap excluded the
    word "IS" (I/S are uppercase LETTERS, not outside A-Z0-9), so "The
    answer is B." -- arguably the single most natural way to phrase this
    -- silently never matched at all. Caught by this fix's own end-to-end
    test (test_mc_end_to_end_generator_receives_options_not_the_bare_
    question) once the generation-prompt fix started actually eliciting
    this exact phrasing from a fake generator."""
    assert _match_deterministic_answer("The answer is B.", "B", "mc") == "correct"
    assert _match_deterministic_answer("I believe the correct option is C.", "C", "mc") == "correct"


def test_mc_the_answer_is_not_b_phrasing_does_not_spuriously_match():
    """The "(?:\\s+IS)?" widening must not reopen the negation false-
    positive door -- "answer is NOT B" must still correctly fail, the
    same way the plain punctuation-only gap already protected against it
    (see test_tf_does_not_spuriously_match_boolean_word_buried_mid_
    explanation for the tf analogue)."""
    assert _match_deterministic_answer("The answer is not B, it is actually C.", "B", "mc") == "incorrect"


def test_tf_end_to_end_via_evaluate_question_with_reasoning_style_generator(monkeypatch):
    """Same real-world shape as the confirmed incident, exercised through
    the full evaluate_question() path, not just the helper directly."""
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


# ── _build_generation_question — real, confirmed-in-production bug ─────────
#
# Real DEBUG output on a live run confirmed: even after the matcher fix
# above, mc accuracy stayed at 0/N. Raw answer previews for mc questions
# were pure open-ended prose that never mentioned an option letter at all.
# Root cause: q.options was never referenced anywhere in evaluate.py, and
# no generator template has any mc/tf-specific instruction either -- mc/tf
# questions were sent through the EXACT SAME generation prompt as open
# questions. The matcher was correctly implemented but had nothing to
# match against.

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
    """Defensive: a malformed mc question with no options dict must not
    crash -- just falls back to the bare question rather than injecting
    an empty/broken "Options:" block."""
    q = EvalQuestion(
        question="Which interface?", keywords=[], corpus="3gpp",
        answer_type="mc", options=None, reference_answer="B",
    )
    assert _build_generation_question(q, q.question) == q.question


def test_mc_end_to_end_generator_receives_options_not_the_bare_question(monkeypatch):
    """The actual fix, exercised through the full evaluate_question() path
    -- confirms the generator is called with options included, and that
    search() (retrieval) is NOT polluted with the options text."""
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


# ── Abstention — distinguishing "retrieval failed" from "reasoning wrong" ──
#
# Design requirement: a generator honestly saying the context doesn't
# contain enough information is a DIFFERENT failure than confidently
# picking the wrong answer -- collapsing both into "incorrect" makes a
# retrieval/chunking problem indistinguishable from a generation/reasoning
# one in the aggregate numbers, and would reward forced-guessing (tf's
# ~50% random-guess floor looks "better" than honest abstention despite
# being strictly less informative about retrieval quality).

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
    phrase "not mentioned", which is also on _is_not_found_answer()'s
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
    # Committed to A -- correct when the reference IS A...
    assert _match_deterministic_answer(answer, "A", "mc") == "correct"
    # ...and INCORRECT (not "abstained") when the reference is something
    # else -- proving this isn't just "always correct now", the
    # commitment is genuinely being checked against the reference.
    assert _match_deterministic_answer(answer, "B", "mc") == "incorrect"


def test_match_deterministic_answer_still_abstains_when_no_commitment_is_present_at_all():
    """A genuine abstention -- no letter/boolean commitment pattern
    anywhere, only an honest "insufficient information" -- must still
    classify as abstained. The commitment-first reordering only changes
    what happens when a commitment IS found; it must not make abstention
    unreachable."""
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
    """mc_tf_outcome must be populated on the SUCCESS path too, not just
    for abstained/incorrect answers."""
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
    """The actual root-cause fix, exercised through the full
    evaluate_question() path: response_schema must reach generate() (and
    therefore the generator), and a structured JSON answer must be read
    directly -- no regex, no phrase-list matching -- to produce the
    outcome."""
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
    """Confirms open questions get NO structural constraint -- generate()
    still gets called, but with response_schema=None."""
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
