"""
Evaluation orchestrator for Open Telco RAG.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
import tiktoken
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm


class _TqdmLoggingHandler(logging.Handler):
    """Routes logging.warning() through tqdm.write() so warnings don't break
    an active progress bar. Installed for one evaluate_corpus() call."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

from open_telco_rag.evaluation._claude import claude_call
from open_telco_rag.generators._prompts import DEFAULT_PROMPT_MODE
from open_telco_rag.evaluation.questions import EvalQuestion, load_questions
from open_telco_rag.evaluation.results import (
    QuestionResult, CorpusResult, aggregate, save_results, print_summary,
)
from open_telco_rag.evaluation.metrics.generation import (
    faithfulness, answer_relevancy, _is_not_found_answer_mc_tf,
)
from open_telco_rag.evaluation.metrics.retrieval import (
    cosine_sim_stats, distinct_sources_retrieved, canonical_source_hit,
    exact_chunk_hit, chunk_rank, coverage_ratio, is_discriminating_source,
    span_hit_at_k, span_hit_in_context,
)
from open_telco_rag.core.retrieval import fill_context_budget, resolve_budget_args
from open_telco_rag.evaluation.metrics.ground_truth import (
    context_recall, answer_correctness,
)

def _build_generator(
    generator_name: str,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = 60,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_tokens: int = 1000,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
):
    """Construct the RAG-answer Generator ("claude", "openrouter", or
    "local") for pipeline.generate().

    temperature only matters for self-consistency (n_generations>1).
    top_p=None omits it from the API call (provider default applies).
    prompt_mode: see generators/_prompts.py. Raise max_tokens for
    reasoning-capable OpenRouter models, whose reasoning tokens share the
    completion budget and can leave the visible answer empty. ClaudeGenerator
    ignores the temperature/prompt_mode/max_tokens overrides here.
    """
    if generator_name == "claude":
        from open_telco_rag.generators.claude import ClaudeGenerator
        return ClaudeGenerator(model=model_name, top_p=top_p) if model_name else ClaudeGenerator(top_p=top_p)

    if generator_name == "local":
        from open_telco_rag.generators.local import LocalGenerator
        return LocalGenerator(model=model_name or "local-model", base_url=base_url,
                               api_key=api_key, timeout=timeout, temperature=temperature,
                               top_p=top_p, max_tokens=max_tokens, prompt_mode=prompt_mode)

    # openrouter (default)
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    return OpenRouterGenerator(model=model_name or "meta-llama/llama-3.1-8b-instruct",
                                timeout=timeout, temperature=temperature, top_p=top_p,
                                max_tokens=max_tokens, prompt_mode=prompt_mode)

def _local_token_count(text: str) -> int:
    """Calculates tokens locally using the cl100k_base encoding."""
    if not text: return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback approximation if tiktoken fails
        return int(len(str(text).split()) * 1.3)

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _reranker_stats(reranker: Any) -> dict | None:
    """truncation counters of the run's reranker (OTel), or None if it has none"""
    scored = getattr(reranker, "scored_count", None)
    if not _is_number(scored) or not scored:
        return None
    return {
        "truncation_rate": reranker.truncation_rate,
        "truncated": reranker.truncated_count,
        "scored": scored,
        "query_max_chars": reranker.query_max_chars,
        "doc_max_chars": reranker.doc_max_chars,
    }


def _embedder_stats(embedder: Any) -> dict | None:
    """truncation counters (when tracking is on) and OpenRouter usage, if present"""
    stats: dict = {}
    scored = getattr(embedder, "scored_count", None)
    if _is_number(scored) and scored:
        stats.update(truncation_rate=embedder.truncation_rate,
                     truncated=embedder.truncated_count, scored=scored)
    tokens = getattr(embedder, "total_prompt_tokens", None)
    cost = getattr(embedder, "total_cost_usd", None)
    if _is_number(tokens) and tokens:
        stats["prompt_tokens"] = tokens
    if _is_number(cost) and cost:
        stats["cost_usd"] = cost
    return stats or None


def _snapshot_llm_usage(llm_call: Callable[[str, int], str]) -> tuple[int, int]:
    """(cumulative prompt_tokens, completion_tokens) recorded on llm_call by
    _claude.py's wrappers. (0, 0) for a callable that doesn't track usage.
    """
    return (
        getattr(llm_call, "total_prompt_tokens", 0),
        getattr(llm_call, "total_completion_tokens", 0),
    )


def _recording_llm_call(
    llm_call: Callable[[str, int], str], metric: str, sink: list[dict],
) -> Callable[[str, int], str]:
    """wraps llm_call so each (metric, prompt, raw response) is appended to sink;
    the real call (and its usage tracking) is delegated unchanged"""
    def _wrapped(prompt: str, max_tokens: int) -> str:
        raw = llm_call(prompt, max_tokens)
        sink.append({"metric": metric, "prompt": prompt, "raw_response": raw})
        return raw
    return _wrapped


def _build_baseline_generator(llm_call: Callable[[str, int], str], max_tokens: int = 1000):
    """Generator that answers from LLM knowledge only - no context chunks.
    Raise max_tokens for reasoning-capable OpenRouter models.
    """
    class _BaselineGenerator:
        def generate(
            self, question: str, results: list, source_type: str = "telecom",
            response_schema: dict | None = None,
        ) -> dict:
            # response_schema accepted for parity but unused; llm_call has no
            # structured-output mechanism (falls back to prose parsing).
            prompt = (
                f"You are a {source_type} technical expert. "
                "Answer the following question based on your knowledge. "
                "Be precise and technical.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            # Before/after delta on llm_call's thread-local usage/cost
            # tracking, to surface real usage/cost into this dict.
            prompt_before, completion_before = (
                getattr(llm_call, "total_prompt_tokens", 0), getattr(llm_call, "total_completion_tokens", 0),
            )
            cost_before = getattr(llm_call, "total_cost", None)
            answer = llm_call(prompt, max_tokens=max_tokens)
            prompt_after, completion_after = (
                getattr(llm_call, "total_prompt_tokens", 0), getattr(llm_call, "total_completion_tokens", 0),
            )
            cost_after = getattr(llm_call, "total_cost", None)

            usage = {}
            if hasattr(llm_call, "total_prompt_tokens"):
                usage = {
                    "prompt_tokens":     prompt_after - prompt_before,
                    "completion_tokens": completion_after - completion_before,
                    "total_tokens":      (prompt_after - prompt_before) + (completion_after - completion_before),
                }
            # cost_before is None only because nothing accumulated on this
            # thread yet -> treat as 0.0. cost_after None means the backend
            # never computes a cost (local vLLM, Claude).
            cost = (cost_after - (cost_before or 0.0)) if cost_after is not None else None

            return {
                "answer":           answer,
                "sources":          [],
                "chunks_used":      0,
                "retrieved_chunks": [],
                "usage":            usage,
                "cost":             cost,
            }
    return _BaselineGenerator()


# TRUE/FALSE words a free-text tf answer might use. Just YES/NO -
# CORRECT/INCORRECT is unsafe near negation.
_TF_WORD_TO_BOOL: dict[str, str] = {
    "TRUE": "TRUE", "YES": "TRUE",
    "FALSE": "FALSE", "NO": "FALSE",
}

# Option letter as its own token at the start ("B) ..."). Excludes "I" (the pronoun).
_MC_NEAR_START_RE = re.compile(r"^[\s*`\"']*([A-HJ-Z])\b")
# "answer is B" / "option B" / "(B)" anywhere. "(?:\s+IS)?" handles "the
# answer is B" while keeping negation safety ("ANSWER IS NOT B" captures nothing).
_MC_EXPLICIT_RE = re.compile(r"(?:ANSWER|OPTION)(?:\s+IS)?[^A-Z0-9]{0,5}\(?([A-Z])\)?\b")
# BOOLEAN as an equivalent label to ANSWER: baseline prose answers
# "**boolean**: false", not "ANSWER IS FALSE".
_TF_EXPLICIT_RE = re.compile(r"(?:ANSWER|BOOLEAN)(?:\s+IS)?[^A-Z0-9]{0,5}(TRUE|FALSE)\b")


def _mc_committed_letter(clean_ans: str) -> str | None:
    """The option letter clean_ans commits to (any letter), or None.
    Detecting any commitment first is what makes running this before the
    abstention check safe."""
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
    # Checked as a whole word, not a substring (avoids "not false"/"falsely").
    first_words = re.findall(r"[A-Z']+", clean_ans)[:3]
    for w in first_words:
        if w in _TF_WORD_TO_BOOL:
            return _TF_WORD_TO_BOOL[w]
    explicit = _TF_EXPLICIT_RE.search(clean_ans)
    if explicit:
        return explicit.group(1)
    return None


def _match_deterministic_answer_prose(answer: str, reference_answer: Any, answer_type: str) -> str:
    """FALLBACK ONLY - see _match_deterministic_answer(). Classifies an
    mc/tf free-text answer against reference_answer as
    "correct"/"incorrect"/"abstained".

    Commitment (any letter / TRUE-FALSE word) is checked FIRST, abstention
    phrases SECOND - so an answer that commits to a letter and then explains
    a wrong option away with "not mentioned" isn't misread as an abstention.
    Detecting ANY commitment (not just the reference letter) keeps a
    wrong-but-confident answer scored "incorrect", not "abstained".

    "abstained" is tracked separately from "incorrect": an honest "not
    enough context" points at retrieval, a confident wrong answer at
    generation, and collapsing them would reward forced-guessing.

    Conservative: an unmatched, non-abstained answer stays "incorrect" -
    a prompt-template issue worth surfacing, not guessed at.
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
# The model emits {"verdict": "answered"|"abstained", "option"/"boolean",
# "justification"} via response_format: json_schema. When honoured,
# _match_deterministic_answer() reads the fields directly; the prose parser
# is the fallback.

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
    None for open questions (no structural change there - free text
    stays free text)."""
    if q.answer_type == "mc" and q.options:
        return _mc_response_schema(q.options)
    if q.answer_type == "tf":
        return _tf_response_schema()
    return None


def _parse_structured_verdict(answer: str, answer_type: str) -> dict | None:
    """Parse answer as the mc/tf structured-verdict JSON object, or None if
    it isn't one (the signal to fall back to prose parsing). Checks the
    required key for this answer_type ("option"/"boolean") so a wrong-shaped
    object also falls through.
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
        # obj["boolean"] isn't a bool/null or an unambiguous "true"/"false"
        # string; fall through to prose parsing (bool("false") is True).
        return None
    return obj


def _normalize_tf_boolean_in_place(obj: dict) -> bool:
    """Normalise obj["boolean"] to a real Python bool/None (mutating obj);
    returns whether that succeeded. Accepts the JSON string "true"/"false"
    (some servers emit it); rejects any other type rather than guessing.
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
    phrase-list matching - the model already told us directly whether it
    answered or abstained."""
    if obj["verdict"] == "abstained":
        return "abstained"

    clean_ref = str(reference_answer).strip().upper()
    if answer_type == "mc":
        option = obj.get("option")
        if option is None:
            # "answered" but option null: treat as abstained, not incorrect.
            return "abstained"
        return "correct" if str(option).strip().upper() == clean_ref else "incorrect"

    # tf
    boolean = obj.get("boolean")
    if boolean is None:
        return "abstained"
    # boolean is a real Python bool here (normalised in _parse_structured_verdict).
    ref_is_true = clean_ref in ("TRUE", "YES")
    return "correct" if boolean == ref_is_true else "incorrect"


def _match_deterministic_answer(answer: str, reference_answer: Any, answer_type: str) -> str:
    """Entry point for mc/tf scoring - use this, not the helper functions
    directly. Tries the structured verdict path first (JSON produced under
    response_format: json_schema - see _response_schema_for()); falls back
    to prose parsing only when the answer doesn't match that shape.
    """
    structured = _parse_structured_verdict(answer, answer_type)
    if structured is not None:
        return _classify_structured_verdict(structured, reference_answer, answer_type)
    return _match_deterministic_answer_prose(answer, reference_answer, answer_type)


def _run_retrieval_pass(
    q: EvalQuestion,
    store: Any,
    embedder: Any,
    search_query: str,
    active_question: str,
    top_k: int,
    reranker: Any,
    rerank_top_k: int,
    search_method: str,
    resolve_parents: bool,
    context_token_budget: int | None,
    candidate_pool: int | None,
    budget_overflow: str | None,
    result: "QuestionResult",
) -> tuple[list, list]:
    """search -> optional rerank -> optional budget fill, filling in result's
    latency / candidate_* / used_* / budget fields. shared by the RAG and
    retrieval_only paths so they can't diverge. returns (candidates, retrieved).
    """
    from open_telco_rag.core.pipeline import search

    # in budget mode, over-fetch candidate_pool instead of top_k; the budget
    # fill decides how many are actually used
    budgeted = context_token_budget is not None
    fetch_k = candidate_pool if budgeted else top_k
    search_start = time.perf_counter()
    retrieved_candidates = search(
        search_query,
        store=store,
        embedder=embedder,
        top_k=fetch_k,
        reranker=None,
        rerank_top_k=fetch_k,
        search_method=search_method,
        resolve_parents=resolve_parents
    )
    result.search_latency = time.perf_counter() - search_start

    result.candidate_chunk_ids = [getattr(r, "doc_id", None) for r in retrieved_candidates]
    result.candidate_chunk_texts = [getattr(r, "text", "") for r in retrieved_candidates]

    result.rerank_latency = 0.0
    result.rerank_tokens = 0

    rerank_n = candidate_pool if budgeted else rerank_top_k
    if reranker:
        rerank_start = time.perf_counter()

        rerank_payload = search_query + " ".join([c.text for c in retrieved_candidates])
        result.rerank_tokens = _local_token_count(rerank_payload)

        try:
            retrieved = reranker.rerank(active_question, retrieved_candidates, top_k=rerank_n)
        except AttributeError:
            # Reranker wrapper may use a different method name.
            retrieved = retrieved_candidates[:rerank_n]

        result.rerank_latency = time.perf_counter() - rerank_start
    else:
        retrieved = retrieved_candidates[:rerank_n]

    if budgeted:
        retrieved = fill_context_budget(retrieved, context_token_budget, budget_overflow)
        result.context_tokens = sum(_local_token_count(r.text) for r in retrieved)
        result.n_chunks_used = len(retrieved)
        result.budget = context_token_budget
        result.fill_ratio = result.context_tokens / context_token_budget

    # chunks handed to the generator (post budget fill when a budget is set)
    result.used_chunk_ids = [getattr(r, "doc_id", None) for r in retrieved]
    result.used_chunk_texts = [getattr(r, "text", "") for r in retrieved]

    return retrieved_candidates, retrieved


def _compute_retrieval_metrics(
    result: "QuestionResult",
    retrieved_candidates: list,
    retrieved: list,
    q: EvalQuestion,
    store: Any,
    sim_threshold: float,
) -> None:
    """fill in every retrieval-tier metric on result; shared by the RAG path
    (after scoring) and the retrieval_only path. never called for baseline."""
    mean_s, min_s, max_s  = cosine_sim_stats(retrieved)
    result.mean_similarity = mean_s
    result.min_similarity  = min_s
    result.max_similarity  = max_s
    result.distinct_sources_retrieved = distinct_sources_retrieved(retrieved)
    result.canonical_hit   = canonical_source_hit(
        retrieved, q.canonical_source, store=store, expected_release=q.release,
    )
    result.exact_chunk_hit = exact_chunk_hit(retrieved, q.chunk_id, q.source_text)
    result.candidate_chunk_rank = chunk_rank(retrieved_candidates, q.chunk_id, q.source_text)
    result.used_chunk_rank = chunk_rank(retrieved, q.chunk_id, q.source_text)
    # overlap-based companions to the exact-chunk metrics; only when the
    # question carries an evidence span (otherwise the fields stay None)
    if q.evidence_doc_id and q.evidence_char_start is not None and q.evidence_char_end is not None:
        result.span_hit_at_k = span_hit_at_k(
            retrieved_candidates, q.evidence_char_start, q.evidence_char_end, q.evidence_doc_id,
        )
        result.span_hit_in_context = span_hit_in_context(
            retrieved, q.evidence_char_start, q.evidence_char_end, q.evidence_doc_id,
        )
    result.coverage_ratio  = coverage_ratio(retrieved, sim_threshold)

    # Separate from canonical_hit: None = no ground truth, False =
    # rejected as too generic.
    if q.canonical_source is None:
        result.canonical_source_discriminating = None
    else:
        result.canonical_source_discriminating = is_discriminating_source(
            q.canonical_source, store
        )


# ── Self-consistency (n_generations>1 in evaluate_question()) ───────────────
# Retrieval runs once; generation repeats N times at a non-zero temperature.
# For mc/tf: extract what each generation committed to, majority-vote on
# that, then score the winning answer once against ground truth.

def _extract_predicted_value(answer: str, answer_type: str) -> str | None:
    """The raw predicted option letter (mc) or "TRUE"/"FALSE" (tf) an answer
    commits to, or None if it abstains / can't be parsed (a votable category
    of its own). Same structured-first, prose-fallback detection as
    _match_deterministic_answer(), minus the reference comparison.
    """
    structured = _parse_structured_verdict(answer, answer_type)
    if structured is not None:
        if structured["verdict"] == "abstained":
            return None
        if answer_type == "mc":
            option = structured.get("option")
            return str(option).strip().upper() if option is not None else None
        boolean = structured.get("boolean")
        return None if boolean is None else ("TRUE" if boolean else "FALSE")

    clean_ans = answer.strip().strip(".*`\"'").upper()
    committed = _mc_committed_letter(clean_ans) if answer_type == "mc" else _tf_committed_bool(clean_ans)
    return committed  # None covers abstention and unparseable alike


def _majority_vote(values: list) -> tuple[int, dict]:
    """values[i] is generation i's predicted value (None = abstained, a
    valid votable category). Returns (winning_index, counts): winning_index
    is the first value matching the plurality winner (ties broken by first
    occurrence); counts maps each distinct value to its generation count.
    """
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    winning_index = values.index(winner)
    return winning_index, counts


def _build_generation_question(q: EvalQuestion, active_question: str) -> str:
    """The question text sent to the RAG generator - kept separate from the
    retrieval query so option text / format instructions don't distort
    retrieval. Explains each schema field and permits "abstained" rather
    than forcing a guess.
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
    retrieval_only: bool = False,
    reranker:      Any   = None,
    rerank_top_k:  int   = 5,
    debug:         bool  = False,
    rag_model_name: str | None = None,
    rag_generator_name: str = "openrouter",
    rag_base_url:  str | None = None,
    rag_api_key:   str | None = None,
    rag_generator_timeout: int = 60,
    inject_metadata: bool = False,
    include_keywords: bool = False,
    search_method: str = "dense",
    resolve_parents:  bool = False,
    n_generations: int = 1,
    generation_temperature: float | None = None,
    generation_top_p: float | None = None,
    rag_max_tokens: int = 1000,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    context_token_budget: int | None = None,
    candidate_pool: int | None = None,
    budget_overflow: str | None = None,
    record_judge_calls: bool = False,
) -> QuestionResult:
    """Run all applicable metrics for one question.

    record_judge_calls=True: also store every judge prompt/raw response for
    this question in result.judge_calls (open/fill_blank only; mc/tf never
    call a judge). Off by default -- the field stays None and is omitted from
    saved output, since prompts include the retrieved chunk text.

    retrieval_only=True: runs retrieval (respecting reranker/budget) and every
    retrieval-tier metric, then returns WITHOUT calling a generator or judge:
    no answer, no token/cost tracking, no scoring (answer stays "", the
    generation-only fields keep their defaults). Mutually exclusive with
    baseline (ValueError). Off by default; existing callers are unaffected.

    n_generations>1: self-consistency. Retrieval runs once; generation
    repeats N times against the same context, then combined - majority vote
    on the raw predicted value for mc/tf, averaged judge-metric scores for
    open. RAG path only (baseline always makes one generation).

    generation_temperature must be non-zero for n_generations>1 to mean
    anything (else the repeats are byte-identical). None leaves it at 0.0.
    rag_max_tokens: raise for reasoning-capable OpenRouter models.
    prompt_mode: RAG branch only.

    context_token_budget=None (default): top_k/rerank_top_k control retrieval
    as before; candidate_pool/budget_overflow are only valid with a budget
    (ValueError otherwise, defaults 30 / "stop").
    Set: over-fetches candidate_pool candidates instead of top_k, reranks
    (or truncates) to candidate_pool instead of rerank_top_k, then fills
    the budget in rank order (see core.retrieval.fill_context_budget);
    top_k/rerank_top_k are ignored. Populates
    result.context_tokens/n_chunks_used/budget/fill_ratio.
    """
    context_token_budget, candidate_pool, budget_overflow = resolve_budget_args(
        context_token_budget, candidate_pool, budget_overflow)
    from open_telco_rag.core.pipeline import search, generate

    if baseline and retrieval_only:
        raise ValueError("baseline and retrieval_only are mutually exclusive")

    active_question = q.question
    if inject_metadata and q.canonical_source:
        active_question = f"According to {q.corpus.upper()} document {q.canonical_source}, {q.question}"

    # Retrieval query: no option text / format instructions.
    search_query = active_question
    if include_keywords and getattr(q, "keywords", None):
        kw_str = " ".join(q.keywords) if isinstance(q.keywords, list) else str(q.keywords)
        if kw_str.strip():
            search_query = f"{active_question} Keywords: {kw_str}"

    generation_question = _build_generation_question(q, active_question)
    response_schema = _response_schema_for(q)  # None for open questions

    result = QuestionResult(
        question=active_question, corpus=q.corpus,
        answer="", sources=[], chunks_used=0,
        answer_type=q.answer_type,
        difficulty=q.difficulty,
    )
    if record_judge_calls:
        result.judge_calls = []

    def _judge(metric: str):
        # off by default: the judge callable is passed through untouched
        if not record_judge_calls:
            return llm_call
        return _recording_llm_call(llm_call, metric, result.judge_calls)

    try:
        start_time = time.perf_counter()
        if baseline:
            retrieved = []
            generator = _build_baseline_generator(rag_llm_call or llm_call, max_tokens=rag_max_tokens)

            gen_start = time.perf_counter()
            gen       = generate(generation_question, retrieved, source_type=q.corpus, generator=generator,
                                  response_schema=response_schema)
            gens = [gen]  # baseline never repeats
            result.gen_latency = time.perf_counter() - gen_start
            result.search_latency = 0.0
        elif retrieval_only:
            retrieved_candidates, retrieved = _run_retrieval_pass(
                q, store, embedder, search_query, active_question, top_k, reranker, rerank_top_k,
                search_method, resolve_parents, context_token_budget, candidate_pool, budget_overflow,
                result,
            )
            result.latency = time.perf_counter() - start_time
            result.chunks_used = len(retrieved)
            _compute_retrieval_metrics(result, retrieved_candidates, retrieved, q, store, sim_threshold)
            return result
        else:
            retrieved_candidates, retrieved = _run_retrieval_pass(
                q, store, embedder, search_query, active_question, top_k, reranker, rerank_top_k,
                search_method, resolve_parents, context_token_budget, candidate_pool, budget_overflow,
                result,
            )

            gen_start = time.perf_counter()
            generator = _build_generator(
                rag_generator_name, model_name=rag_model_name,
                base_url=rag_base_url, api_key=rag_api_key,
                timeout=rag_generator_timeout,
                temperature=(generation_temperature or 0.0) if n_generations > 1 else 0.0,
                top_p=generation_top_p,
                max_tokens=rag_max_tokens,
                prompt_mode=prompt_mode,
            )
            gens = [
                generate(generation_question, retrieved, source_type=q.corpus, generator=generator,
                         response_schema=response_schema)
                for _ in range(max(1, n_generations))
            ]
            result.gen_latency = time.perf_counter() - gen_start

        # Pick the representative generation. mc/tf with >1 generation:
        # majority-vote on the raw predicted value; `gen` becomes the winner
        # so scoring below is unchanged.
        if len(gens) > 1 and q.answer_type in ("mc", "tf"):
            predicted = [_extract_predicted_value(g.get("answer", ""), q.answer_type) for g in gens]
            winning_index, vote_counts = _majority_vote(predicted)
            gen = gens[winning_index]
            result.self_consistency_n = len(gens)
            result.self_consistency_votes = {
                ("abstained/unparsed" if k is None else k): v for k, v in vote_counts.items()
            }
            result.self_consistency_agreement = max(vote_counts.values()) / len(gens)
        else:
            gen = gens[0]

        result.latency = time.perf_counter() - start_time

        # Prefer the generator's real usage over a tiktoken estimate;
        # token_source records which. Summed across every generation made.
        answer = gen.get("answer", "")
        usages = [g.get("usage") or {} for g in gens]
        prompt_token_list = [u.get("prompt_tokens") for u in usages]
        completion_token_list = [u.get("completion_tokens") for u in usages]

        if all(v is not None for v in prompt_token_list) and all(v is not None for v in completion_token_list):
            result.prompt_tokens = sum(prompt_token_list)
            result.completion_tokens = sum(completion_token_list)
            result.total_tokens = sum(
                u.get("total_tokens", (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0))
                for u in usages
            )
            result.token_source = "api"
            cached_list = [(u.get("prompt_tokens_details") or {}).get("cached_tokens") for u in usages]
            reasoning_list = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") for u in usages]
            result.cached_tokens = sum(c for c in cached_list if c is not None) if any(c is not None for c in cached_list) else None
            result.reasoning_tokens = sum(r for r in reasoning_list if r is not None) if any(r is not None for r in reasoning_list) else None
        else:
            # Matches what generate() was actually called with above, once per generation made.
            prompt_payload = generation_question + " ".join([c.text for c in retrieved])
            result.prompt_tokens = _local_token_count(prompt_payload) * len(gens)
            result.completion_tokens = sum(_local_token_count(g.get("answer", "")) for g in gens)
            result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.token_source = "estimated"
            result.cached_tokens = None
            result.reasoning_tokens = None

        # None only when every generation's cost was None (e.g. Claude, no
        # pricing table); else summed across all generations.
        costs = [g.get("cost") for g in gens]
        result.cost = sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None

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

            # accuracy is 1.0/0.0 (abstained = 0.0); mc_tf_outcome carries
            # the 3-way classification.
            result.mc_tf_outcome = outcome
            result.accuracy = 1.0 if outcome == "correct" else 0.0
            if debug:
                # Whether json_schema was honoured, or prose fallback.
                structured = _parse_structured_verdict(answer, q.answer_type) is not None
                print(f"  [DEBUG] [{q.answer_type}] reference={q.reference_answer!r} "
                      f"outcome={outcome} accuracy={result.accuracy} "
                      f"structured={'yes' if structured else 'NO (prose fallback)'} "
                      f"(answer: {answer[:120]!r})")
        else:
            # Bracket just the judge calls, so an earlier generation call on
            # the same llm_call isn't counted in the delta.
            judge_before = _snapshot_llm_usage(llm_call)
            if len(gens) > 1:
                # Self-consistency for open: each judge metric computed per
                # generation, then averaged (not voted). result.answer stays
                # gens[0]'s text; the scores reflect all generations.
                faithfulness_scores = []
                relevancy_scores = []
                recall_scores = []
                correctness_scores = []
                for g in gens:
                    g_answer = g.get("answer", "")
                    # faithfulness/context_recall are undefined with no
                    # retrieved context; skip for baseline so the -1.0 "not
                    # applicable" sentinel is kept rather than a misleading 0.0.
                    if not baseline:
                        faithfulness_scores.append(faithfulness(g_answer, retrieved, llm_call=_judge("faithfulness"), debug=debug))
                    relevancy_scores.append(
                        answer_relevancy(active_question, g_answer, embedder, llm_call=_judge("answer_relevancy"), debug=debug)
                    )
                    if q.reference_answer:
                        if not baseline:
                            recall_scores.append(
                                context_recall(q.reference_answer, retrieved, llm_call=_judge("context_recall"), debug=debug)
                            )
                        correctness_scores.append(
                            answer_correctness(g_answer, q.reference_answer, embedder, llm_call=_judge("answer_correctness"), debug=debug)
                        )
                if faithfulness_scores:
                    result.faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
                result.answer_relevancy = sum(relevancy_scores) / len(relevancy_scores)
                if recall_scores:
                    result.context_recall = sum(recall_scores) / len(recall_scores)
                if correctness_scores:
                    result.answer_correctness = sum(correctness_scores) / len(correctness_scores)
                result.self_consistency_n = len(gens)
            else:
                # See above: faithfulness/context_recall skipped for baseline.
                if not baseline:
                    result.faithfulness = faithfulness(answer, retrieved, llm_call=_judge("faithfulness"), debug=debug)
                result.answer_relevancy = answer_relevancy(active_question, answer, embedder, llm_call=_judge("answer_relevancy"), debug=debug)
                if q.reference_answer:
                    if not baseline:
                        result.context_recall = context_recall(q.reference_answer, retrieved, llm_call=_judge("context_recall"), debug=debug)
                    result.answer_correctness = answer_correctness(answer, q.reference_answer, embedder, llm_call=_judge("answer_correctness"), debug=debug)
            judge_after = _snapshot_llm_usage(llm_call)
            result.judge_prompt_tokens = judge_after[0] - judge_before[0]
            result.judge_completion_tokens = judge_after[1] - judge_before[1]

        if not baseline:
            _compute_retrieval_metrics(result, retrieved_candidates, retrieved, q, store, sim_threshold)

    except Exception as e:
        result.error = str(e)
        result.error_type = type(e).__name__
        result.latency = time.perf_counter() - start_time

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
    progress_bar:  bool  = False,
    max_concurrency: int = 1,
    rag_generator_model: str | None = None,
    rag_generator_name: str = "openrouter",
    rag_base_url:  str | None = None,
    rag_api_key:   str | None = None,
    rag_generator_timeout: int = 60,
    n_generations: int = 1,
    generation_temperature: float | None = None,
    generation_top_p: float | None = None,
    inject_metadata: bool = False,
    include_keywords: bool = False,
    search_method: str = "dense",
    resolve_parents:  bool = False,
    rag_max_tokens: int = 1000,
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    context_token_budget: int | None = None,
    candidate_pool: int | None = None,
    budget_overflow: str | None = None,
    retrieval_only: bool = False,
    record_judge_calls: bool = False,
) -> tuple[CorpusResult, CorpusResult | None]:
    """Run the full evaluation suite for one corpus.

    rag_max_tokens / prompt_mode: passed through to evaluate_question().
    context_token_budget/candidate_pool/budget_overflow: passed through to
    evaluate_question()'s RAG pass only (baseline never retrieves); optional,
    unset = plain top_k path.

    record_judge_calls: passed to evaluate_question() for the RAG pass; off by default.

    retrieval_only: passed to evaluate_question()'s RAG pass only (baseline is
    unaffected either way). Off by default. include_baseline=True with
    retrieval_only=True still runs a real baseline generation pass.

    Returns (rag_result, baseline_result); baseline_result is None if
    include_baseline=False.
    """
    # Label for the leaderboard/results row.
    questions_source_name = "default"
    # fail fast on bad budget settings (before any per-question work)
    context_token_budget, candidate_pool, budget_overflow = resolve_budget_args(
        context_token_budget, candidate_pool, budget_overflow)

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

    # Route logger.warning() through tqdm so it doesn't break the progress
    # bar. Root logger; removed at the end of this function.
    _tqdm_log_handler = None
    if progress_bar:
        _tqdm_log_handler = _TqdmLoggingHandler()
        logging.getLogger().addHandler(_tqdm_log_handler)

    # ── RAG pass ──────────────────────────────────────────────────────────────
    # max_concurrency>1 runs questions through a thread pool; the llm_call
    # wrappers are blocking requests calls, so the GIL releases during I/O
    # and N requests overlap. max_concurrency=1 uses the same machinery with
    # one worker (still sequential). store/embedder are shared read-only
    # across threads (chromadb supports concurrent readers).
    rag_results: list[Any] = [None] * len(qs)
    rag_wall_start = time.perf_counter()
    # tqdm.write() via emit() so output coexists with an active bar.
    pbar = tqdm(total=len(qs), desc=f"{corpus}/{model} RAG", unit="q") if progress_bar else None
    emit = pbar.write if pbar else print
    running_correct = 0
    running_scored = 0

    def _run_one(idx: int, q: EvalQuestion):
        display_q = q.question
        if inject_metadata and q.canonical_source:
            display_q = f"According to {corpus.upper()} document {q.canonical_source}, {q.question}"
        if not progress_bar:
            print(f"  [RAG {idx + 1}/{len(qs)}] {display_q[:65]}...")
        qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                               llm_call=llm_call, rag_llm_call=rag_llm_call,
                               baseline=False, retrieval_only=retrieval_only,
                               record_judge_calls=record_judge_calls, reranker=reranker,
                               rerank_top_k=rerank_top_k, debug=debug,
                               rag_model_name=resolved_rag_model,
                               rag_generator_name=rag_generator_name,
                               rag_base_url=rag_base_url, rag_api_key=rag_api_key,
                               rag_generator_timeout=rag_generator_timeout,
                               n_generations=n_generations,
                               generation_temperature=generation_temperature,
                               generation_top_p=generation_top_p,
                               inject_metadata=inject_metadata,
                               include_keywords=include_keywords,
                               search_method=search_method,
                               resolve_parents=resolve_parents,
                               rag_max_tokens=rag_max_tokens,
                               prompt_mode=prompt_mode,
                               context_token_budget=context_token_budget,
                               candidate_pool=candidate_pool,
                               budget_overflow=budget_overflow,)
        if delay > 0:
            time.sleep(delay)
        return idx, qr

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        futures = [executor.submit(_run_one, i, q) for i, q in enumerate(qs)]
        # as_completed() so the bar updates as questions finish; rag_results
        # is written back by index, so ordering is unaffected.
        for future in concurrent.futures.as_completed(futures):
            idx, qr = future.result()
            rag_results[idx] = qr
            if qr.error:
                emit(f"    [ERROR] {qr.error}")
            elif retrieval_only:
                # no generation happens, so print retrieval metrics and keep
                # the -1.0 accuracy sentinel out of the running accuracy
                if not progress_bar:
                    print(f"    sim={qr.mean_similarity:.3f}  n_chunks_used={qr.n_chunks_used}")
            elif qr.answer_type in ("mc", "tf"):
                if not progress_bar:
                    print(f"    accuracy={qr.accuracy:.2f}  sim={qr.mean_similarity:.3f}")
                running_scored += 1
                running_correct += qr.accuracy
            else:
                if not progress_bar:
                    print(f"    faithful={qr.faithfulness:.2f}  "
                          f"relevancy={qr.answer_relevancy:.2f}  "
                          f"sim={qr.mean_similarity:.3f}  "
                          f"recall={qr.context_recall:.2f}  "
                          f"correctness={qr.answer_correctness:.2f}")
            if pbar:
                postfix = {}
                if running_scored:
                    postfix["acc"] = f"{running_correct / running_scored:.2f}"
                if qr.error:
                    postfix["err"] = "yes"
                pbar.set_postfix(postfix)
                pbar.update(1)
    if pbar:
        pbar.close()
    rag_wall_clock = time.perf_counter() - rag_wall_start

    rag_result = CorpusResult(
        corpus=corpus, model=model, top_k=top_k,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        questions_file=questions_source_name,
        question_results=rag_results,
        rag_generator_model=resolved_rag_model,
        rag_generator_name=rag_generator_name,
        total_wall_clock_seconds=rag_wall_clock,
        max_concurrency=max(1, max_concurrency),
        # Mirrors evaluate_question()'s own resolution: temperature is forced
        # to 0.0 whenever n_generations==1, regardless of what was configured.
        rag_generator_temperature=(generation_temperature or 0.0) if n_generations > 1 else 0.0,
        rag_generator_top_p=generation_top_p,
        **aggregate(rag_results),
    )

    rag_result.reranker_stats = _reranker_stats(reranker)
    rag_result.embedder_stats = _embedder_stats(embedder)

    print_summary(rag_result)

    if output_dir:
        save_results(rag_result, Path(output_dir))

    # ── Baseline pass (optional) ──────────────────────────────────────────────
    baseline_result = None
    if include_baseline:
        print(f"\n  Running BASELINE pass ({len(qs)} questions, no retrieval)…")
        bl_results = []
        bl_wall_start = time.perf_counter()
        bl_pbar = tqdm(total=len(qs), desc=f"{corpus}/{model} baseline", unit="q") if progress_bar else None
        bl_emit = bl_pbar.write if bl_pbar else print
        for i, q in enumerate(qs, 1):
            if not progress_bar:
                print(f"  [BL  {i}/{len(qs)}] {q.question[:65]}...")
            qr = evaluate_question(q, store, embedder, top_k, sim_threshold,
                                   llm_call=llm_call, rag_llm_call=rag_llm_call,
                                   baseline=True, debug=debug, rag_max_tokens=rag_max_tokens)
            bl_results.append(qr)
            if qr.error:
                bl_emit(f"    [ERROR] {qr.error}")
            elif qr.answer_type in ("mc", "tf"):
                if not progress_bar:
                    print(f"    accuracy={qr.accuracy:.2f}")
            else:
                if not progress_bar:
                    print(f"    faithful={qr.faithfulness:.2f}  "
                          f"relevancy={qr.answer_relevancy:.2f}  "
                          f"recall={qr.context_recall:.2f}  "
                          f"correctness={qr.answer_correctness:.2f}")
            if bl_pbar:
                bl_pbar.update(1)
            if delay > 0 and i < len(qs):
                time.sleep(delay)
        if bl_pbar:
            bl_pbar.close()
        bl_wall_clock = time.perf_counter() - bl_wall_start

        baseline_result = CorpusResult(
            corpus=corpus, model=model, top_k=0,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            question_results=bl_results,
            rag_generator_model=resolved_rag_model,
            rag_generator_name=rag_generator_name,
            total_wall_clock_seconds=bl_wall_clock,
            max_concurrency=1,  # baseline pass is a plain sequential for-loop, always
            **aggregate(bl_results),
        )

        print(f"\n  BASELINE summary:")
        print_summary(baseline_result)

        if output_dir:
            save_results(baseline_result, Path(output_dir) / "baseline")

    if _tqdm_log_handler is not None:
        logging.getLogger().removeHandler(_tqdm_log_handler)

    return rag_result, baseline_result