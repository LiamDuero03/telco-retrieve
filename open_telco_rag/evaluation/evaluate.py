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
    faithfulness, answer_relevancy, _is_not_found_answer_mc_tf,
)
from open_telco_rag.evaluation.metrics.retrieval import (
    cosine_sim_stats, distinct_sources_retrieved, canonical_source_hit,
    exact_chunk_hit, coverage_ratio, is_discriminating_source,
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
    """Construct the RAG-answer Generator ("claude", "openrouter", or "local")
    for pipeline.generate(). Same three backends make_llm_call() supports for
    the judge, so --rag-generator and --generator select independently.
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
    llm_call — reads attributes _claude.py's call wrappers accumulate on
    the callable itself, keeping the plain Callable[[str, int], str]
    signature metrics functions expect. (0, 0) for any llm_call that
    doesn't track usage; evaluate_question() takes the delta between two
    snapshots, so an untracked callable just contributes (0, 0).
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
            # response_schema accepted for interface parity but unused — llm_call
            # is a bare Callable with no structured-output mechanism. Callers still
            # get free text; _parse_structured_verdict() falls back to prose parsing.
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


# Words a free-text tf answer uses for TRUE/FALSE when a reasoning model
# explains before concluding (e.g. "No, because..."). Deliberately just
# YES/NO, not CORRECT/INCORRECT — those are unsafe near negation ("this is
# NOT correct" means FALSE, but matching "correct" alone would say TRUE).
_TF_WORD_TO_BOOL: dict[str, str] = {
    "TRUE": "TRUE", "YES": "TRUE",
    "FALSE": "FALSE", "NO": "FALSE",
}

# Option letter as its own token at the very start of the answer (e.g. "B)
# The E2 interface."). Excludes "I" — virtually always the pronoun ("I
# believe..."), never a real option letter; a genuine lettered "I" option
# is still caught by _MC_EXPLICIT_RE below.
_MC_NEAR_START_RE = re.compile(r"^[\s*`\"']*([A-HJ-Z])\b")
# Explicit "answer is B" / "option B" / "(B)" phrasing anywhere, for a
# reasoning model that concludes with the letter. "(?:\s+IS)?" specifically
# handles "the answer is B" — a plain punctuation-only gap excludes "IS"
# (uppercase letters, not punctuation), so it would never match otherwise.
# Only special-cases "IS" so negation safety is preserved ("ANSWER IS NOT
# B" still fails to capture a letter).
_MC_EXPLICIT_RE = re.compile(r"(?:ANSWER|OPTION)(?:\s+IS)?[^A-Z0-9]{0,5}\(?([A-Z])\)?\b")
_TF_EXPLICIT_RE = re.compile(r"ANSWER IS (TRUE|FALSE)\b")


def _mc_committed_letter(clean_ans: str) -> str | None:
    """Returns the option letter clean_ans commits to (any letter, not just
    a check against one specific reference), or None. Detecting ANY
    commitment (not "does it match the reference") is what makes running
    this before the abstention check safe — see _match_deterministic_answer_
    prose()'s docstring."""
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
    # Checked as its own WORD among the first few, not a substring — avoids
    # picking up "not false"/"falsely" buried in the explanation.
    first_words = re.findall(r"[A-Z']+", clean_ans)[:3]
    for w in first_words:
        if w in _TF_WORD_TO_BOOL:
            return _TF_WORD_TO_BOOL[w]
    explicit = _TF_EXPLICIT_RE.search(clean_ans)
    if explicit:
        return explicit.group(1)
    return None


def _match_deterministic_answer_prose(answer: str, reference_answer: Any, answer_type: str) -> str:
    """FALLBACK ONLY — see _match_deterministic_answer(), the real entry
    point. Used for legacy results (pre-structured-generation) and when a
    generator/server doesn't honor response_format: json_schema (returns
    free text instead — see _parse_structured_verdict()). Classifies an
    mc/tf answer against reference_answer as "correct"/"incorrect"/
    "abstained".

    Three confirmed production bugs were fixed here, all from the same
    root cause (inferring a verdict from free prose after the fact) —
    kept only as the fallback now that structured generation avoids the
    whole class of bug:
    1. Original matching was exact/prefix-match against raw text, so any
       reasoning model that explains before concluding ("No, because...")
       never matched — confirmed via a grid where mc/tf accuracy was
       exactly 0/120 across all 24 combos despite many answers being
       substantively correct.
    2. A punctuation-only gap regex excluded the word "IS", so "the
       answer is B" never matched (fixed by _MC_EXPLICIT_RE's optional
       "(?:\\s+IS)?").
    3. Abstention-phrase matching ran BEFORE commitment-detection, so an
       answer that clearly committed to a letter but explained away a
       wrong option using "not mentioned" (also on _is_not_found_answer's
       phrase list) was misclassified "abstained" — silently capping
       wrong_rate at 0.0. Fixed by checking commitment first; an
       abstention phrase after a committed answer is just reasoning
       prose, not real abstention.

    Still intentionally conservative: only a handful of yes/no phrasings
    and near-start/explicit letter patterns are recognized. An unmatched,
    non-abstained answer means the generator's phrasing doesn't map to
    the reference's canonical form — worth surfacing as a prompt-template
    issue, not silently guessed at.

    "abstained" is tracked separately from "incorrect" because an honest
    "not enough context" points at retrieval/chunking, while a confident
    wrong answer points at generation/reasoning — collapsing them would
    also reward forced-guessing (tf's ~50% random floor beats honest
    abstention in raw accuracy despite being less informative).

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
    the phrase "not mentioned" — which is also on _is_not_found_answer_mc_tf()'s
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

    if _is_not_found_answer_mc_tf(answer):
        return "abstained"

    return "incorrect"


# ── Structured, schema-constrained generation (mc/tf) ───────────────────────
#
# Root-cause fix instead of another prose-parsing patch: ask the model to
# emit {"verdict": "answered"|"abstained", "option"/"boolean": ...,
# "justification": str} via response_format: json_schema (see
# generators/local.py — same mechanism enrich_llm_metadata() already uses,
# with a real schema instead of generic json_object mode). When honored,
# _match_deterministic_answer() reads the fields directly — no regex, no
# phrase-list matching. _match_deterministic_answer_prose() is the fallback
# for anything that doesn't honor it.

def _mc_response_schema(options: dict[str, str]) -> dict:
    """JSON schema for one mc answer. "option" enum is built from this
    question's actual option letters, not a fixed A-E guess."""
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
    """Parses answer as the mc/tf structured-verdict JSON object, or None if
    it isn't one — the signal _match_deterministic_answer() uses to fall
    back to prose parsing. None (not an exception) covers: server/model
    doesn't honor json_schema at all; honors json_object generically but not
    the schema/enum constraint; or legacy free-prose data. Checks for the
    exact required key for this answer_type ("option"/"boolean") so a
    wrong-shaped object (e.g. tf schema for an mc question) also falls
    through to prose rather than being misread.
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
    if answer_type == "tf" and not _normalize_tf_boolean_in_place(obj):
        # obj["boolean"] is neither a real bool/null nor an unambiguous
        # "true"/"false" string -- can't trust it enough to classify, and
        # coercing it with bool() would silently mis-score (bool("false")
        # is True). Falls through to prose parsing on the raw text instead.
        return None
    return obj


def _normalize_tf_boolean_in_place(obj: dict) -> bool:
    """Normalizes obj["boolean"] to an actual Python bool/None, mutating
    obj, and returns whether that succeeded.

    The schema declares boolean as JSON boolean-or-null
    (_tf_response_schema()), but guided decoding doesn't always enforce
    that strictly -- a model/server can still emit the JSON STRING "false"
    instead of the JSON literal false. Recognizes unambiguous string
    spellings (case/whitespace-insensitive) so a real verdict isn't thrown
    away; any other type (number, list, an unrecognized string, ...) is
    rejected rather than guessed at -- see _classify_structured_verdict()'s
    tf branch, which trusts this field's type unconditionally afterwards.
    """
    boolean = obj["boolean"]
    if boolean is None or isinstance(boolean, bool):
        return True
    if isinstance(boolean, str):
        lowered = boolean.strip().lower()
        if lowered in ("true", "false"):
            obj["boolean"] = lowered == "true"
            return True
    return False


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
            # "answered" but option is null — inconsistent response a server
            # that only validates shape can still emit. Treat as abstained
            # rather than scoring "incorrect" against a missing answer.
            return "abstained"
        return "correct" if str(option).strip().upper() == clean_ref else "incorrect"

    # tf
    boolean = obj.get("boolean")
    if boolean is None:
        return "abstained"
    # boolean is guaranteed a real Python bool here, never a stray type
    # like the string "false" -- see _parse_structured_verdict()'s
    # _normalize_tf_boolean_in_place() call, which already rejected/
    # normalized anything else before this object could reach us.
    ref_is_true = clean_ref in ("TRUE", "YES")
    return "correct" if boolean == ref_is_true else "incorrect"


def _match_deterministic_answer(answer: str, reference_answer: Any, answer_type: str) -> str:
    """Entry point for mc/tf scoring — use this, not the helper functions
    directly. Tries the structured verdict path first (JSON produced under
    response_format: json_schema — see _response_schema_for()); falls back
    to prose parsing only when the answer doesn't match that shape.
    """
    structured = _parse_structured_verdict(answer, answer_type)
    if structured is not None:
        return _classify_structured_verdict(structured, reference_answer, answer_type)
    return _match_deterministic_answer_prose(answer, reference_answer, answer_type)


def _build_generation_question(q: EvalQuestion, active_question: str) -> str:
    """Builds the question text sent to the RAG generator — kept separate
    from active_question/search_query (used for retrieval) so option
    text/format instructions never distort which chunks get retrieved.

    The schema (_response_schema_for()) only constrains shape, not meaning,
    so this text explains what each field means and when to use "abstained"
    vs "answered", naming fields explicitly.

    Abstention is permitted rather than forcing a guess: bad retrieval
    should show up as low accuracy, not be masked by landing near tf's ~50%
    guess floor by chance.
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
    
    active_question = q.question
    if inject_metadata and q.canonical_source:
        active_question = f"According to {q.corpus.upper()} document {q.canonical_source}, {q.question}"

    # Built from active_question, not generation_question below — options/
    # format instructions belong in the generation prompt only, and would
    # distort retrieval if included in the search query.
    search_query = active_question
    if include_keywords and getattr(q, "keywords", None):
        kw_str = " ".join(q.keywords) if isinstance(q.keywords, list) else str(q.keywords)
        if kw_str.strip():
            search_query = f"{active_question} Keywords: {kw_str}"

    generation_question = _build_generation_question(q, active_question)
    # None for open questions. Passed to both baseline and RAG generate()
    # calls so mc/tf get schema-constrained verdicts either way.
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
            # Search pass, without reranker.
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

            result.rerank_latency = 0.0
            result.rerank_tokens = 0

            if reranker:
                rerank_start = time.perf_counter()

                rerank_payload = search_query + " ".join([c.text for c in retrieved_candidates])
                result.rerank_tokens = _local_token_count(rerank_payload)

                try:
                    retrieved = reranker.rerank(active_question, retrieved_candidates, top_k=rerank_top_k)
                except AttributeError:
                    # Reranker wrapper may use a different method name.
                    retrieved = retrieved_candidates[:rerank_top_k]

                result.rerank_latency = time.perf_counter() - rerank_start
            else:
                retrieved = retrieved_candidates[:rerank_top_k]

            gen_start = time.perf_counter()
            generator = _build_generator(
                rag_generator_name, model_name=rag_model_name,
                base_url=rag_base_url, api_key=rag_api_key,
            )
            gen       = generate(generation_question, retrieved, source_type=q.corpus, generator=generator,
                                  response_schema=response_schema)
            result.gen_latency = time.perf_counter() - gen_start

        result.latency = time.perf_counter() - start_time

        # Prefer the generator's real usage block over a tiktoken estimate;
        # only estimate when usage data is genuinely absent. token_source
        # records which happened so downstream analysis can tell them apart.
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
            # Matches what generate() was actually called with above.
            prompt_payload = generation_question + " ".join([c.text for c in retrieved])
            result.prompt_tokens = _local_token_count(prompt_payload)
            result.completion_tokens = _local_token_count(answer)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.token_source = "estimated"
            result.cached_tokens = None
            result.reasoning_tokens = None

        # None ("not computed") only when the generator returned no cost at
        # all (e.g. claude.py, no pricing table) — not unconditionally
        # zeroed, so genuinely-free (local vLLM) isn't indistinguishable
        # from never-computed.
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

            # accuracy is a plain 1.0/0.0 over all mc/tf (abstained = 0.0);
            # mc_tf_outcome carries the three-way classification aggregate()
            # uses for mean_accuracy_attempted/abstention_rate/wrong_rate.
            result.mc_tf_outcome = outcome
            result.accuracy = 1.0 if outcome == "correct" else 0.0
            if debug:
                # Whether the model/server honored json_schema, or fell
                # back to prose parsing.
                structured = _parse_structured_verdict(answer, q.answer_type) is not None
                print(f"  [DEBUG] [{q.answer_type}] reference={q.reference_answer!r} "
                      f"outcome={outcome} accuracy={result.accuracy} "
                      f"structured={'yes' if structured else 'NO (prose fallback)'} "
                      f"(answer: {answer[:120]!r})")
        else:
            # judge_before/after bracket just these four judge calls — taken
            # here (not at function top) so an earlier RAG/baseline
            # generation call sharing the same llm_call isn't counted in the
            # judge-token delta.
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
            result.distinct_sources_retrieved = distinct_sources_retrieved(retrieved)
            result.canonical_hit   = canonical_source_hit(
                retrieved, q.canonical_source, store=store, expected_release=q.release,
            )
            result.exact_chunk_hit = exact_chunk_hit(retrieved, q.chunk_id, q.source_text)
            result.coverage_ratio  = coverage_ratio(retrieved, sim_threshold)

            # Separate from canonical_hit so a reader can tell "no ground
            # truth available" (None) from "rejected as too generic" (False)
            # — see is_discriminating_source() for the bug this guards against.
            if q.canonical_source is None:
                result.canonical_source_discriminating = None
            else:
                result.canonical_source_discriminating = is_discriminating_source(
                    q.canonical_source, store
                )

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