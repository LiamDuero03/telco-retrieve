"""
Evaluation orchestrator for Open Telco RAG.
"""

from __future__ import annotations

import json
import re
import time
import tiktoken
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from open_telco_rag.evaluation._claude import claude_call
from open_telco_rag.evaluation.questions import EvalQuestion, load_questions
from open_telco_rag.evaluation.results import (
    QuestionResult, CorpusResult, aggregate, save_results, print_summary,
)
from open_telco_rag.evaluation.metrics.generation import (
    faithfulness, answer_relevancy, _is_not_found_answer,
)
from open_telco_rag.evaluation.metrics.retrieval import (
    cosine_sim_stats, source_coverage, canonical_source_hit, coverage_ratio,
)
from open_telco_rag.evaluation.metrics.ground_truth import (
    context_recall, answer_correctness,
)

def _build_generator(
    generator_name: str,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Construct the RAG-answer Generator matching generator_name
    ("claude", "openrouter", or "local") for pipeline.generate() — the
    same three backends make_llm_call() supports for the eval-metrics LLM
    judge, so --rag-generator and --generator each independently select
    their own backend.
    """
    if generator_name == "claude":
        from open_telco_rag.generators.claude import ClaudeGenerator
        return ClaudeGenerator(model=model_name) if model_name else ClaudeGenerator()

    if generator_name == "local":
        from open_telco_rag.generators.local import LocalGenerator
        return LocalGenerator(model=model_name or "local-model", base_url=base_url, api_key=api_key)

    # openrouter (default)
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    return OpenRouterGenerator(model=model_name or "meta-llama/llama-3.1-8b-instruct")

def _local_token_count(text: str) -> int:
    """Calculates tokens locally using the cl100k_base encoding."""
    if not text: return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback approximation if tiktoken fails
        return int(len(str(text).split()) * 1.3)

def _snapshot_llm_usage(llm_call: Callable[[str, int], str]) -> tuple[int, int]:
    """(cumulative prompt_tokens, completion_tokens) recorded so far on
    llm_call — a plain-attribute read, matching how _claude.py's
    claude_call/_make_local_call/_make_openrouter_call/_custom_claude_call
    each accumulate total_prompt_tokens/total_completion_tokens as
    attributes on the callable itself (see that module's own comments for
    why: keeping the Callable[[str, int], str] signature every metrics
    function already calls, rather than threading a richer return type
    through faithfulness()/answer_relevancy()/context_recall()/
    answer_correctness()). (0, 0) for any llm_call that doesn't track
    usage at all (e.g. a plain test-fixture lambda) — evaluate_question()
    takes the DELTA between two snapshots, so an untracked callable just
    contributes a delta of (0, 0), never an error.
    """
    return (
        getattr(llm_call, "total_prompt_tokens", 0),
        getattr(llm_call, "total_completion_tokens", 0),
    )


def _build_baseline_generator(llm_call: Callable[[str, int], str]):
    """Generator that answers from LLM knowledge only — no context chunks."""
    class _BaselineGenerator:
        def generate(
            self, question: str, results: list, source_type: str = "telecom",
            response_schema: dict | None = None,
        ) -> dict:
            # response_schema accepted for interface parity with generate()
            # but not wired up here -- llm_call is a bare Callable[[str,
            # int], str], no structured-output mechanism to plug a JSON
            # schema into. A caller requesting it here still gets free
            # text; evaluate.py's _parse_structured_verdict() correctly
            # treats that as "not structured" and falls back to prose
            # parsing (see ClaudeGenerator.generate()'s own comment for
            # the same reasoning).
            prompt = (
                f"You are a {source_type} technical expert. "
                "Answer the following question based on your knowledge. "
                "Be precise and technical.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            answer = llm_call(prompt, max_tokens=1000)
            return {
                "answer":           answer,
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
            }
    return _BaselineGenerator()


# reference_answer's canonical value -> the words a free-text tf answer
# commonly uses to mean the same thing, when the RAG generator is a
# reasoning-style model that explains before concluding (e.g. "No,
# because the specification states X" instead of a bare "False").
# Deliberately just YES/NO, not CORRECT/INCORRECT -- those are unsafe
# near negation ("this is NOT correct" means FALSE, but naively finding
# the word "correct" among the leading words would score it as TRUE; see
# test_tf_does_not_spuriously_match_boolean_word_buried_mid_explanation).
_TF_WORD_TO_BOOL: dict[str, str] = {
    "TRUE": "TRUE", "YES": "TRUE",
    "FALSE": "FALSE", "NO": "FALSE",
}

# The option letter as its own token right at the very start of the
# answer (e.g. "B) The E2 interface." or "B, because..."). Deliberately
# excludes "I" from the letter class — "I" is virtually always the
# pronoun ("I believe the correct option is C...") at the start of a
# sentence, never genuinely an option letter in practice; without this
# exclusion a generic [A-Z] class would wrongly treat "I" itself as a
# committed answer and never even reach the "option C" explicit mention
# later in the same sentence. A genuinely lettered "I" option (vanishingly
# rare) would still be caught by _MC_EXPLICIT_RE below.
_MC_NEAR_START_RE = re.compile(r"^[\s*`\"']*([A-HJ-Z])\b")
# Or an explicit "answer is B" / "option B" / "(B)" phrasing anywhere in
# the answer, covering a reasoning model that concludes with the letter
# instead of leading with it. The optional "(?:\s+IS)?" is required for
# the single most common phrasing, "the answer is B" -- found via this
# fix's own test suite: a plain [^A-Z0-9]{0,10} gap excludes "IS" itself
# (I/S are uppercase LETTERS, not in that character class), so "ANSWER IS
# B" would silently never match at all. Deliberately only special-cases
# the literal word "IS", not an arbitrary gap -- keeps the same negation-
# safety a punctuation-only gap already has ("ANSWER IS NOT B" still
# correctly fails to capture a letter: "NOT" isn't "IS" and isn't
# punctuation either, so neither branch reaches the letter).
_MC_EXPLICIT_RE = re.compile(r"(?:ANSWER|OPTION)(?:\s+IS)?[^A-Z0-9]{0,5}\(?([A-Z])\)?\b")
_TF_EXPLICIT_RE = re.compile(r"ANSWER IS (TRUE|FALSE)\b")


def _mc_committed_letter(clean_ans: str) -> str | None:
    """Returns the option letter clean_ans commits to (ANY letter, not
    just a check against one specific reference), or None if no
    commitment is found at all. See _match_deterministic_answer()'s
    docstring for why detecting ANY commitment (not "does it match the
    reference letter specifically") is what makes commitment-detection
    safe to run BEFORE the abstention check."""
    near_start = _MC_NEAR_START_RE.match(clean_ans)
    if near_start:
        return near_start.group(1)
    explicit = _MC_EXPLICIT_RE.search(clean_ans)
    if explicit:
        return explicit.group(1)
    return None


def _tf_committed_bool(clean_ans: str) -> str | None:
    """Returns "TRUE"/"FALSE" clean_ans commits to, or None. Mirrors
    _mc_committed_letter()'s "detect ANY commitment first" reasoning."""
    if clean_ans in ("TRUE", "FALSE"):
        return clean_ans
    # The claimed boolean as its own WORD among the first couple of
    # words -- not a bare substring check, so e.g. an answer that happens
    # to contain "not false" or "falsely" deep in its explanation doesn't
    # get picked up out of context (see test_tf_does_not_spuriously_
    # match_boolean_word_buried_mid_explanation).
    first_words = re.findall(r"[A-Z']+", clean_ans)[:3]
    for w in first_words:
        if w in _TF_WORD_TO_BOOL:
            return _TF_WORD_TO_BOOL[w]
    explicit = _TF_EXPLICIT_RE.search(clean_ans)
    if explicit:
        return explicit.group(1)
    return None


def _match_deterministic_answer_prose(answer: str, reference_answer: Any, answer_type: str) -> str:
    """FALLBACK ONLY — see _match_deterministic_answer() below, the actual
    entry point every caller should use. This prose-parsing path exists
    for legacy/ungraded data (results computed before structured
    generation existed) and as a graceful-degradation path for any
    generator/server combination that doesn't honor response_format:
    json_schema (silently returns free text instead — see
    _parse_structured_verdict()'s own docstring). Classifies an mc/tf
    answer against reference_answer as one of three outcomes: "correct",
    "incorrect", or "abstained".

    Kept for a real reason, not just backward compatibility: three
    separate regex-matching bugs were found and patched in this exact
    function across one session (a naive [^A-Z0-9]{0,10} gap silently
    excluding the word "IS" so "the answer is B" never matched; a generic
    near-start letter class colliding with the English pronoun "I"; and
    abstention-phrase-matching running before commitment-detection,
    misclassifying a clearly-committed answer as "abstained" whenever its
    OWN justification happened to explain away a wrong option using a
    phrase like "not mentioned"). Each was a genuine, confirmed
    production bug — but all three came from the same root cause: trying
    to infer a structured verdict from free-form prose after the fact.
    _match_deterministic_answer() now avoids this class of bug entirely
    for any generation call that honors response_format: json_schema
    (see _mc_response_schema()/_tf_response_schema() and
    _build_generation_question()) by reading verdict/option/boolean
    directly out of the model's own structured JSON output — no regex,
    no phrase-list matching, nothing left to patch a fourth time. This
    function remains exactly as it was (same three fixes, same
    reasoning) as the fallback for whatever it can't structurally
    guarantee: an older result, or a server/model that doesn't actually
    honor the schema constraint.

    Real, confirmed bug this fixes (the matching logic): the original
    logic was pure exact-match / prefix-match against the RAG generator's
    raw text (clean_ref == clean_ans, or clean_ans.startswith(clean_ref),
    or the literal substring "ANSWER IS {clean_ref}") — which silently
    scored 0.0 for any answer that doesn't literally BEGIN with the
    reference token. A reasoning-style RAG generator (e.g. openai/gpt-
    oss-120b) routinely explains before concluding — "No, this is
    incorrect because..." for a tf question whose reference_answer is
    "False", or a full explanatory sentence for an mc question that never
    leads with the bare option letter — which the original logic could
    never match, regardless of embedding model or retrieval quality
    (confirmed via a real aggregated grid: mc/tf accuracy was exactly
    0/120 across EVERY one of 24 combinations, with several manually-
    inspected answers verified as substantively correct on reading).

    Still intentionally conservative — this recognizes a handful of
    common yes/no-style phrasings and an mc option letter appearing near
    the start (as its own WORD, not a substring) or in an explicit
    "answer is X"/"option X" pattern, not literally any way to imply an
    answer; a still-unmatched (and non-abstained) answer means the RAG
    generator's phrasing genuinely doesn't correspond to the reference
    answer's canonical form, which is worth knowing (a prompt-template
    fix, not a scorer fix) rather than papering over.

    "abstained" (a real design distinction, not just a bug fix): a
    generator honestly saying the provided context doesn't contain
    enough information to answer is a DIFFERENT failure than confidently
    picking the wrong letter/boolean — one points at retrieval/chunking
    (nothing relevant was ever surfaced), the other at generation/judge
    reasoning (something relevant WAS surfaced, but the model still got
    it wrong). Collapsing both into "incorrect" (accuracy=0.0, same as a
    genuinely wrong guess) makes those two completely different problems
    indistinguishable in the aggregate numbers — and worse, would reward
    forced-guessing: a generator that never abstains lands on tf's ~50%
    random-guess floor by chance alone, which looks "better" than honest
    abstention despite being strictly less informative about retrieval
    quality.

    Commitment (letter/boolean) is checked FIRST, abstention SECOND —
    NOT the reverse. Real, confirmed bug this fixes: a real mc answer
    read "The answer is A.\n\nJustification: ... The other options
    (allocating additional IP addresses or a UE-initiated handover) are
    not mentioned as conditions that cause an Update Bearer Request." —
    a clean, unambiguous commitment to A, whose justification explains
    why the OTHER options are wrong (completely normal mc reasoning) using
    the phrase "not mentioned" — which is also on _is_not_found_answer()'s
    phrase list. With abstention checked FIRST (the original ordering),
    this answer was misclassified as "abstained" despite committing to a
    letter, which silently capped wrong_rate at 0.0 across several real
    grid combinations (a truly-wrong-but-confident answer that happened to
    explain away a wrong option this same way would ALSO have been
    swallowed into "abstained" instead of "incorrect", making wrong_rate
    look suspiciously clean rather than genuinely zero). Checking
    commitment first and only falling through to the abstention phrase
    check when NO letter/boolean was found at all fixes this: an
    abstention phrase appearing anywhere in an answer that has already
    committed to a letter/boolean is just normal reasoning prose, not a
    signal the generator didn't know the answer.

    Critically, commitment-detection here checks for ANY letter/boolean,
    not just whether reference_answer specifically appears — necessary
    for this ordering to be safe: an answer committing to the WRONG
    letter must still be classified "incorrect", not fall through to the
    abstention check just because ITS specific reference letter wasn't
    found (see _mc_committed_letter()/_tf_committed_bool()).
    """
    clean_ans = answer.strip().strip(".*`\"'").upper()
    clean_ref = str(reference_answer).strip().upper()
    if not clean_ref:
        return "incorrect"

    committed = _mc_committed_letter(clean_ans) if answer_type == "mc" else _tf_committed_bool(clean_ans)
    if committed is not None:
        return "correct" if committed == clean_ref else "incorrect"

    if _is_not_found_answer(answer):
        return "abstained"

    return "incorrect"


# ── Structured, schema-constrained generation (mc/tf) ───────────────────────
#
# Root-cause fix, not another prose-parsing patch: instead of inferring a
# verdict from free text after the fact, ask the model to emit exactly
# {"verdict": "answered"|"abstained", "option"/"boolean": ..., "justification":
# str} via response_format: json_schema (see generators/local.py's own
# comment for the exact request shape — same "response_format" field
# core/enrichment.py's enrich_llm_metadata() already uses successfully
# against both OpenRouter and this project's vLLM server, just with a real
# schema instead of the generic {"type": "json_object"} mode). When the
# server/model actually honors this, _match_deterministic_answer() reads
# verdict/option/boolean directly — no regex, no phrase-list matching, no
# fourth prose-parsing bug to find. _match_deterministic_answer_prose()
# above remains the fallback for anything that doesn't.

def _mc_response_schema(options: dict[str, str]) -> dict:
    """JSON schema for one mc answer. The "option" enum is built from the
    ACTUAL option letters for THIS question (options.keys()), not a fixed
    A-E guess — a 3-option question can't have the model "choose" a D
    that was never offered, and a question with 5 options isn't
    artificially capped at 4."""
    letters = sorted(options.keys())
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["answered", "abstained"]},
            "option": {"type": ["string", "null"], "enum": [*letters, None]},
            "justification": {"type": "string"},
        },
        "required": ["verdict", "option", "justification"],
        "additionalProperties": False,
    }


def _tf_response_schema() -> dict:
    """JSON schema for one tf answer."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["answered", "abstained"]},
            "boolean": {"type": ["boolean", "null"]},
            "justification": {"type": "string"},
        },
        "required": ["verdict", "boolean", "justification"],
        "additionalProperties": False,
    }


def _response_schema_for(q: EvalQuestion) -> dict | None:
    """The response_format: json_schema payload for THIS question, or
    None for open questions (no structural change there — free text
    stays free text)."""
    if q.answer_type == "mc" and q.options:
        return _mc_response_schema(q.options)
    if q.answer_type == "tf":
        return _tf_response_schema()
    return None


def _parse_structured_verdict(answer: str, answer_type: str) -> dict | None:
    """Parses answer as the mc/tf structured-verdict JSON object, or
    returns None if it isn't one — the signal _match_deterministic_answer()
    uses to fall back to prose parsing. None (not an exception) covers
    every way a generation call can fail to produce schema-conformant
    JSON: the server/model doesn't honor response_format: json_schema at
    all (older vLLM build, no guided-decoding backend configured — see
    generators/local.py's own comment); it honors response_format:
    json_object generically but not the schema/enum constraint, producing
    valid-but-wrong-shaped JSON; or this is legacy data from before
    structured generation existed, when `answer` was always free prose.
    Deliberately checks for the exact required key for THIS answer_type
    ("option" for mc, "boolean" for tf) rather than just "is this valid
    JSON with a verdict key" — a same-shaped-but-wrong-field object (e.g.
    an mc question whose generator somehow returned the tf schema) must
    also fall through to the prose fallback, not silently misread the
    wrong field.
    """
    try:
        obj = json.loads(answer.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("verdict") not in ("answered", "abstained"):
        return None
    required_key = "option" if answer_type == "mc" else "boolean"
    if required_key not in obj:
        return None
    return obj


def _classify_structured_verdict(obj: dict, reference_answer: Any, answer_type: str) -> str:
    """Classifies an already-parsed structured verdict dict (see
    _parse_structured_verdict()) against reference_answer. No regex, no
    phrase-list matching — the model already told us directly whether it
    answered or abstained."""
    if obj["verdict"] == "abstained":
        return "abstained"

    clean_ref = str(reference_answer).strip().upper()
    if answer_type == "mc":
        option = obj.get("option")
        if option is None:
            # verdict said "answered" but option is null -- an internally
            # inconsistent response (shouldn't happen under a properly
            # enforced schema, but a model can still emit this if the
            # server only validates SHAPE, not this kind of cross-field
            # consistency). Treat defensively as abstained rather than
            # silently scoring "incorrect" against a genuinely absent
            # answer.
            return "abstained"
        return "correct" if str(option).strip().upper() == clean_ref else "incorrect"

    # tf
    boolean = obj.get("boolean")
    if boolean is None:
        return "abstained"
    ref_is_true = clean_ref in ("TRUE", "YES")
    return "correct" if bool(boolean) == ref_is_true else "incorrect"


def _match_deterministic_answer(answer: str, reference_answer: Any, answer_type: str) -> str:
    """THE entry point for mc/tf scoring — every caller should use this,
    not _match_deterministic_answer_prose()/_parse_structured_verdict()/
    _classify_structured_verdict() directly. Tries the structured verdict
    path first (answer is the model's own {"verdict":..., "option"/
    "boolean":..., "justification":...} JSON, produced under
    response_format: json_schema — see _response_schema_for()); falls
    back to prose parsing only when answer isn't (or doesn't fully match)
    that shape. See _match_deterministic_answer_prose()'s own docstring
    for why the fallback still exists and is deliberately kept exactly as
    it was, not removed.
    """
    structured = _parse_structured_verdict(answer, answer_type)
    if structured is not None:
        return _classify_structured_verdict(structured, reference_answer, answer_type)
    return _match_deterministic_answer_prose(answer, reference_answer, answer_type)


def _build_generation_question(q: EvalQuestion, active_question: str) -> str:
    """Builds the question text sent to the RAG generator for answer
    generation — kept separate from active_question/search_query (used
    for retrieval), so option text/format instructions here never distort
    which chunks get retrieved.

    Which fields exist and their types/enum come from response_format:
    json_schema (_response_schema_for()), threaded through generate()
    alongside this text. A schema can only constrain shape, not meaning,
    so this text's job is explaining what each field means and when to
    use "abstained" vs "answered" — naming the field names explicitly so
    the model isn't guessing what to fill in.

    Abstention is deliberately still permitted rather than forcing a
    guess every time: a bad retrieval/chunking config that surfaces
    nothing relevant should show up as low accuracy, not be masked by
    landing near tf's ~50% random-guess floor by chance. See
    _mc_response_schema()/_tf_response_schema() for why "option"/
    "boolean" are nullable.
    """
    if q.answer_type == "mc" and q.options:
        options_text = "\n".join(f"{letter}. {text}" for letter, text in sorted(q.options.items()))
        return (
            f"{active_question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Respond with your verdict on this multiple-choice question. If the provided "
            f"context contains enough information to determine the answer, set \"verdict\" to "
            f"\"answered\" and \"option\" to the correct option letter. If the context does NOT "
            f"contain enough information to determine the answer, set \"verdict\" to \"abstained\" "
            f"and \"option\" to null — do not guess. Use \"justification\" to briefly explain your "
            f"reasoning either way, grounded in the context."
        )
    if q.answer_type == "tf":
        return (
            f"{active_question}\n\n"
            f"Respond with your verdict on this true/false statement. If the provided context "
            f"contains enough information to determine whether it is true or false, set "
            f"\"verdict\" to \"answered\" and \"boolean\" to true or false accordingly. If the "
            f"context does NOT contain enough information to determine this, set \"verdict\" to "
            f"\"abstained\" and \"boolean\" to null — do not guess. Use \"justification\" to "
            f"briefly explain your reasoning either way, grounded in the context."
        )
    return active_question


def evaluate_question(
    q:             EvalQuestion,
    store:         Any,
    embedder:      Any,
    top_k:         int   = 5,
    sim_threshold: float = 0.5,
    llm_call:      Callable[[str, int], str] = claude_call,
    rag_llm_call:  Callable[[str, int], str] | None = None,
    baseline:      bool  = False,
    reranker:      Any   = None,
    rerank_top_k:  int   = 5,
    debug:         bool  = False,
    rag_model_name: str | None = None,
    rag_generator_name: str = "openrouter",
    rag_base_url:  str | None = None,
    rag_api_key:   str | None = None,
    inject_metadata: bool = False,
    include_keywords: bool = False,
    search_method: str = "dense",
    resolve_parents:  bool = False,
) -> QuestionResult:
    """Run all applicable metrics for one question."""
    from open_telco_rag.core.pipeline import search, generate
    
    # Rewrite the question if metadata injection is enabled and a source exists
    active_question = q.question
    if inject_metadata and q.canonical_source:
        active_question = f"According to {q.corpus.upper()} document {q.canonical_source}, {q.question}"

    # Prepare search query (appends keywords if flag is enabled) -- built
    # from active_question, NOT generation_question below: options text/
    # format instructions belong in the GENERATION prompt only, appending
    # them here would dilute/distort which chunks retrieval finds.
    search_query = active_question
    if include_keywords and getattr(q, "keywords", None):
        kw_str = " ".join(q.keywords) if isinstance(q.keywords, list) else str(q.keywords)
        if kw_str.strip():
            search_query = f"{active_question} Keywords: {kw_str}"

    # See _build_generation_question()'s own docstring -- this is what
    # actually gets sent to the RAG/baseline generator, distinct from
    # active_question (kept plain for search_query and result.question's
    # own display purposes).
    generation_question = _build_generation_question(q, active_question)
    # None for open questions -- see _response_schema_for()'s own
    # docstring. Passed to BOTH the baseline and RAG generate() calls
    # below so mc/tf get schema-constrained structured verdicts
    # regardless of which path produced the answer.
    response_schema = _response_schema_for(q)

    result = QuestionResult(
        question=active_question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
        answer_type=q.answer_type,
    )

    try:
        start_time = time.perf_counter()
        if baseline:
            retrieved = []
            generator = _build_baseline_generator(rag_llm_call or llm_call)

            gen_start = time.perf_counter()
            gen       = generate(generation_question, retrieved, source_type=q.corpus, generator=generator,
                                  response_schema=response_schema)
            result.gen_latency = time.perf_counter() - gen_start
            result.search_latency = 0.0
        else:
            # 1. Time the base search retrieval pass (WITHOUT reranker)
            search_start = time.perf_counter()
            retrieved_candidates = search(
                search_query, 
                store=store, 
                embedder=embedder,
                top_k=top_k, 
                reranker=None,  
                rerank_top_k=top_k,
                search_method=search_method,
                resolve_parents=resolve_parents
            )
            result.search_latency = time.perf_counter() - search_start
            
            # 2. Time the Reranking pass & calculate its tokens
            result.rerank_latency = 0.0
            result.rerank_tokens = 0
            
            if reranker:
                rerank_start = time.perf_counter()
                
                # Reranker processes the query + all retrieved candidate chunks
                rerank_payload = search_query + " ".join([c.text for c in retrieved_candidates])
                result.rerank_tokens = _local_token_count(rerank_payload)
                
                # Execute reranking explicitly
                try:
                    retrieved = reranker.rerank(active_question, retrieved_candidates, top_k=rerank_top_k)
                except AttributeError:
                    # Fallback if your reranker wrapper uses a different method name
                    retrieved = retrieved_candidates[:rerank_top_k]
                    
                result.rerank_latency = time.perf_counter() - rerank_start
            else:
                retrieved = retrieved_candidates[:rerank_top_k]
            
            # 3. Time the text generation pass
            gen_start = time.perf_counter()
            generator = _build_generator(
                rag_generator_name, model_name=rag_model_name,
                base_url=rag_base_url, api_key=rag_api_key,
            )
            gen       = generate(generation_question, retrieved, source_type=q.corpus, generator=generator,
                                  response_schema=response_schema)
            result.gen_latency = time.perf_counter() - gen_start

        result.latency = time.perf_counter() - start_time

        # 4. Token accounting — prefer the REAL usage block the generator
        # itself returned (local.py/openrouter.py/claude.py all now
        # include one, normalized to prompt_tokens/completion_tokens/
        # total_tokens regardless of backend — see each module's own
        # comment) over a local tiktoken approximation. Only estimate when
        # the generator genuinely didn't return usage data at all (e.g. a
        # baseline/custom generator, or a server whose response omits
        # "usage") — token_source records which happened, so downstream
        # analysis (grid reports, leaderboard) can tell a measured number
        # from an approximation instead of the two looking identical.
        answer = gen.get("answer", "")
        usage = gen.get("usage") or {}
        real_prompt_tokens = usage.get("prompt_tokens")
        real_completion_tokens = usage.get("completion_tokens")

        if real_prompt_tokens is not None and real_completion_tokens is not None:
            result.prompt_tokens = real_prompt_tokens
            result.completion_tokens = real_completion_tokens
            result.total_tokens = usage.get(
                "total_tokens", real_prompt_tokens + real_completion_tokens
            )
            result.token_source = "api"
            prompt_details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            result.cached_tokens = prompt_details.get("cached_tokens")
            result.reasoning_tokens = completion_details.get("reasoning_tokens")
        else:
            # Generator processes generation_question (options/format
            # instructions included for mc/tf) + the final filtered chunks
            # -- matches what generate() was actually called with above.
            prompt_payload = generation_question + " ".join([c.text for c in retrieved])
            result.prompt_tokens = _local_token_count(prompt_payload)
            result.completion_tokens = _local_token_count(answer)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.token_source = "estimated"
            result.cached_tokens = None
            result.reasoning_tokens = None

        # The generator's own real computed cost when it returned one
        # (openrouter.py's per-token pricing math, or local.py's genuine
        # $0.00 self-hosted value) — None ("not computed") only when the
        # generator didn't return a cost at all (currently claude.py, no
        # pricing table wired up there). Deliberately NOT unconditionally
        # zeroed the way this used to be: a bare 0.0 looked identical
        # whether the answer was genuinely free or the cost was simply
        # never computed, which made local vLLM runs indistinguishable
        # from "we forgot to compute this."
        result.cost = gen.get("cost")

        result.answer  = answer
        result.sources = gen.get("sources", [])
        result.chunks_used = gen.get("chunks_used", len(retrieved))

        if debug:
            mode = "BASELINE" if baseline else "RAG"
            print(f"\n  [DEBUG] [{mode}] question: {active_question[:80]}")
            print(f"  [DEBUG] [{mode}] search query: {search_query[:80]}")
            if generation_question != active_question:
                print(f"  [DEBUG] [{mode}] generation prompt: {generation_question}")
            print(f"  [DEBUG] [{mode}] answer preview: {answer[:200]}")

        if q.answer_type in ("mc", "tf"):
            outcome = _match_deterministic_answer(answer, q.reference_answer, q.answer_type)

            # mc/tf gets its own dedicated accuracy field rather than being
            # blended into the OE metrics (faithfulness/answer_relevancy/
            # context_recall/answer_correctness), which stay at their
            # -1.0 "not applicable" default here. accuracy is a plain
            # 1.0/0.0 over ALL mc/tf questions (abstained counts as 0.0,
            # same as genuinely wrong); mc_tf_outcome carries the
            # three-way correct/incorrect/abstained classification
            # aggregate()/CorpusResult use to additionally compute
            # mean_accuracy_attempted, abstention_rate, and wrong_rate —
            # see results.py.
            result.mc_tf_outcome = outcome
            result.accuracy = 1.0 if outcome == "correct" else 0.0
            if debug:
                # Shows whether this model/server honored response_format:
                # json_schema (structured verdict parsed) or silently
                # fell back to prose parsing.
                structured = _parse_structured_verdict(answer, q.answer_type) is not None
                print(f"  [DEBUG] [{q.answer_type}] reference={q.reference_answer!r} "
                      f"outcome={outcome} accuracy={result.accuracy} "
                      f"structured={'yes' if structured else 'NO (prose fallback)'} "
                      f"(answer: {answer[:120]!r})")
        else:
            # Free-text evaluation. judge_before/after snapshot llm_call's
            # cumulative usage (see _snapshot_llm_usage()'s own docstring)
            # immediately around JUST these four judge calls -- taking the
            # "before" snapshot here rather than at the top of the function
            # is what excludes an earlier RAG/baseline generation call that
            # happens to share the same llm_call (baseline mode passes
            # rag_llm_call or llm_call into the answer generator) from this
            # question's judge-token delta.
            judge_before = _snapshot_llm_usage(llm_call)
            result.faithfulness     = faithfulness(answer, retrieved, llm_call=llm_call, debug=debug)
            result.answer_relevancy = answer_relevancy(active_question, answer, embedder, llm_call=llm_call, debug=debug)
            if q.reference_answer:
                result.context_recall     = context_recall(q.reference_answer, retrieved, llm_call=llm_call, debug=debug)
                result.answer_correctness = answer_correctness(answer, q.reference_answer, embedder, llm_call=llm_call, debug=debug)
            judge_after = _snapshot_llm_usage(llm_call)
            result.judge_prompt_tokens = judge_after[0] - judge_before[0]
            result.judge_completion_tokens = judge_after[1] - judge_before[1]

        if not baseline:
            mean_s, min_s, max_s  = cosine_sim_stats(retrieved)
            result.mean_similarity = mean_s
            result.min_similarity  = min_s
            result.max_similarity  = max_s
            result.source_coverage = source_coverage(retrieved)
            result.canonical_hit   = canonical_source_hit(retrieved, q.canonical_source)
            result.coverage_ratio  = coverage_ratio(retrieved, sim_threshold)

    except Exception as e:
        result.error = str(e)
        result.latency = 0.0
        result.prompt_tokens = 0
        result.completion_tokens = 0
        result.search_latency = 0.0 
        result.gen_latency = 0.0

    return result


def evaluate_corpus(
    corpus:        str,
    store:         Any,
    embedder:      Any,
    questions:     list[EvalQuestion] | None = None,
    model:         str   = "minilm",
    top_k:         int   = 5,
    sim_threshold: float = 0.5,
    delay:         float = 1.0,
    output_dir:    Path | None = None,
    llm_call:      Callable[[str, int], str] = claude_call,
    rag_llm_call:  Callable[[str, int], str] | None = None,
    include_baseline: bool = False,
    reranker:      Any   = None,
    rerank_top_k:  int   = 5,
    debug:         bool  = False,
    rag_generator_model: str | None = None,
    rag_generator_name: str = "openrouter",
    rag_base_url:  str | None = None,
    rag_api_key:   str | None = None,
    inject_metadata: bool = False,
    include_keywords: bool = False,
    search_method: str = "dense",
    resolve_parents:  bool = False,
) -> tuple[CorpusResult, CorpusResult | None]:
    """Run the full evaluation suite for one corpus.

    Returns:
        (rag_result, baseline_result) — baseline_result is None if
        include_baseline=False.
    """
    # Label for the leaderboard/results row.
    questions_source_name = "default"
    if questions:
        questions_source_name = "custom_list"
    else:
        questions_source_name = f"{corpus}.json"
        
    qs = questions or load_questions(corpus)

    print(f"\nEvaluating {corpus.upper()} — {len(qs)} questions | "
          f"model={model} | top-k={top_k}"
          + (" | +BASELINE" if include_baseline else "")
          + (" | DEBUG ON" if debug else ""))
    
    # Default fallback string if none is provided down the pipe
    resolved_rag_model = rag_generator_model or "meta-llama/llama-3.1-8b-instruct"

    # ── RAG pass ──────────────────────────────────────────────────────────────
    rag_results = []
    for i, q in enumerate(qs, 1):
        display_q = q.question
        if inject_metadata and q.canonical_source:
            display_q = f"According to {corpus.upper()} document {q.canonical_source}, {q.question}"
        print(f"  [RAG {i}/{len(qs)}] {display_q[:65]}...")
        qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                               llm_call=llm_call, rag_llm_call=rag_llm_call,
                               baseline=False, reranker=reranker,
                               rerank_top_k=rerank_top_k, debug=debug,
                               rag_model_name=resolved_rag_model,
                               rag_generator_name=rag_generator_name,
                               rag_base_url=rag_base_url, rag_api_key=rag_api_key,
                               inject_metadata=inject_metadata,
                               include_keywords=include_keywords,
                               search_method=search_method,
                               resolve_parents=resolve_parents,)
        rag_results.append(qr)
        if qr.error:
            print(f"    [ERROR] {qr.error}")
        elif qr.answer_type in ("mc", "tf"):
            print(f"    accuracy={qr.accuracy:.2f}  sim={qr.mean_similarity:.3f}")
        else:
            print(f"    faithful={qr.faithfulness:.2f}  "
                  f"relevancy={qr.answer_relevancy:.2f}  "
                  f"sim={qr.mean_similarity:.3f}  "
                  f"recall={qr.context_recall:.2f}  "
                  f"correctness={qr.answer_correctness:.2f}")
        if delay > 0 and i < len(qs):
            time.sleep(delay)

    rag_result = CorpusResult(
        corpus=corpus, model=model, top_k=top_k,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        questions_file=questions_source_name,
        question_results=rag_results,
        **aggregate(rag_results),
    )

    print_summary(rag_result)

    if output_dir:
        save_results(rag_result, Path(output_dir))

    # ── Baseline pass (optional) ──────────────────────────────────────────────
    baseline_result = None
    if include_baseline:
        print(f"\n  Running BASELINE pass ({len(qs)} questions, no retrieval)…")
        bl_results = []
        for i, q in enumerate(qs, 1):
            print(f"  [BL  {i}/{len(qs)}] {q.question[:65]}...")
            qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                                   llm_call=llm_call, rag_llm_call=rag_llm_call,
                                   baseline=True, debug=debug)
            bl_results.append(qr)
            if qr.error:
                print(f"    [ERROR] {qr.error}")
            elif qr.answer_type in ("mc", "tf"):
                print(f"    accuracy={qr.accuracy:.2f}")
            else:
                print(f"    faithful={qr.faithfulness:.2f}  "
                      f"relevancy={qr.answer_relevancy:.2f}  "
                      f"recall={qr.context_recall:.2f}  "
                      f"correctness={qr.answer_correctness:.2f}")
            if delay > 0 and i < len(qs):
                time.sleep(delay)

        baseline_result = CorpusResult(
            corpus=corpus, model=model, top_k=0,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            question_results=bl_results,
            **aggregate(bl_results),
        )

        print(f"\n  BASELINE summary:")
        print_summary(baseline_result)

        if output_dir:
            save_results(baseline_result, Path(output_dir) / "baseline")

    return rag_result, baseline_result