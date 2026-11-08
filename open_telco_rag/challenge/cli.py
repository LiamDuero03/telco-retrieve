from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from open_telco_rag.challenge.settings import (
    FALLBACKS,
    describe,
    profile_defaults,
    resolve_settings,
)

__all__ = ["FALLBACKS", "resolve_settings"]

if TYPE_CHECKING:
    from open_telco_rag.challenge.client import ChatClient
    from open_telco_rag.challenge.export import ExportOptions
    from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder

EXAMPLE_COLLECTION = "fixed-grid/text_baseline/otel-0.6b"

EXPORT_EXAMPLE = f"""\
example:
  open-telco-rag challenge export --questions Test.csv --out submission.csv \\
      --db-path ./runs/{EXAMPLE_COLLECTION}/db \\
      --collection otel-0.6b_text_baseline_b512_v256_full \\
      --generator-model my-gemma --llm-base-url http://localhost:8000/v1
({EXAMPLE_COLLECTION} is the published text_baseline / OTel-0.6B cell;
the collection is always passed explicitly.)
"""


@lru_cache(maxsize=1)
def _defaults() -> dict:
    try:
        return profile_defaults()
    except (OSError, ValueError):
        return dict(FALLBACKS)


def _d(key: str):
    return _defaults()[key]


def _seed(text: str) -> int:
    value = int(text)
    if not 0 <= value < 2**32:
        raise argparse.ArgumentTypeError(f"{text} is outside 0..{2**32 - 1}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} must be at least 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} must be greater than 0")
    return value


def _non_negative(kind):
    def parse(text: str):
        value = kind(text)
        if value < 0:
            raise argparse.ArgumentTypeError(f"{text} must not be negative")
        return value

    parse.__name__ = f"non-negative {kind.__name__}"
    return parse


_seed.__name__ = "seed (an integer from 0 to 4294967295)"
_positive_int.__name__ = "positive integer"
_positive_float.__name__ = "positive number"


def _add_export_io(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--questions",
        required=True,
        help="Test CSV with header ID (or question_id), question, options",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Submission CSV to write (written to <out>.partial, then renamed)",
    )
    p.add_argument(
        "--db-path", required=True, help="Directory holding chroma.sqlite3 and <collection>_fts.db"
    )
    p.add_argument("--collection", required=True, help="Collection name inside --db-path")
    p.add_argument(
        "--generator-model",
        default=None,
        help=f"Served model name (default {_d('generator_model')}; LOCAL_LLM_MODEL overrides)",
    )
    p.add_argument(
        "--llm-base-url", default=None, help="Server URL (falls back to LOCAL_LLM_BASE_URL)"
    )


def _add_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--timeout",
        type=_positive_float,
        default=None,
        help=f"Seconds to wait for one generator request (default {_d('timeout'):g})",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <out>.partial of an interrupted run (refuses if it does not match "
        "the Test file); without it an existing .partial is replaced",
    )


def _add_embedder_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        default=None,
        help="Named settings preset (default challenge: OTel-0.6B, OTel reranker, hybrid, "
        "seed 1234); explicit flags win",
    )
    p.add_argument("--model", default=None, help=f"Embedder registry key (default {_d('model')})")
    _add_custom_embedder_options(p)
    p.add_argument(
        "--allow-embedder-mismatch",
        action="store_true",
        help="Continue (with a warning) when the embedder differs from the one the index was built with",
    )


def _add_custom_embedder_options(p: argparse.ArgumentParser) -> None:
    custom = p.add_mutually_exclusive_group()
    custom.add_argument(
        "--embedder-path",
        default=None,
        help="Local sentence-transformers model directory (replaces --model; must match the index)",
    )
    custom.add_argument(
        "--embedder-id",
        default=None,
        help="Hugging Face model id (replaces --model; must match the index)",
    )
    p.add_argument(
        "--query-prefix", default=None, help="Prefix added to queries (custom embedder only)"
    )
    p.add_argument(
        "--document-prefix",
        default=None,
        help="Recorded for consistency with ingest; no effect on queries (custom embedder only)",
    )
    p.add_argument(
        "--embedder-dimensions",
        type=_non_negative(int),
        default=0,
        help="Informational; the real vector size is always checked against the index",
    )


def _add_retrieval_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help=f"Chunks to retrieve (default {_d('top_k')})",
    )
    p.add_argument(
        "--search-method",
        choices=["dense", "bm25", "hybrid"],
        default=None,
        help=f"Retrieval method (default {_d('search_method')}); not dense needs <collection>_fts.db",
    )
    _add_rerank_options(p)


def _add_rerank_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rerank retrieved chunks",
    )
    p.add_argument(
        "--reranker-model",
        dest="reranker",
        metavar="MODEL",
        default=None,
        help=f"Reranker registry key (default {_d('reranker')})",
    )
    p.add_argument(
        "--resolve-parents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace child chunks by their parent chunks (parent_child collections only)",
    )
    p.add_argument(
        "--rerank-top-k",
        type=_positive_int,
        default=None,
        help=f"Chunks kept after reranking (default {_d('rerank_top_k')})",
    )


def _add_hook_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--prompt-builder",
        default=None,
        metavar="MODULE:FUNCTION",
        help="Replace the built-in prompt builder: f(question, passages, max_prompt_tokens) -> Prompt",
    )
    p.add_argument(
        "--answer-parser",
        default=None,
        metavar="MODULE:FUNCTION",
        help="Replace the built-in answer parser: f(model_text, options) -> answer string",
    )


def _add_generation_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help="Seed for python, numpy, torch and the generator request",
    )
    p.add_argument(
        "--temperature",
        type=_non_negative(float),
        default=None,
        help=f"Sampling temperature (default {_d('temperature')})",
    )
    p.add_argument(
        "--max-prompt-tokens",
        type=_positive_int,
        default=None,
        help="Whole-prompt cap in cl100k_base tokens (default: the official cap)",
    )
    p.add_argument(
        "--max-answer-tokens",
        dest="max_tokens",
        metavar="N",
        type=_positive_int,
        default=None,
        help=f"Generated tokens per answer (default {_d('max_tokens')})",
    )
    p.add_argument(
        "--max-tokens", dest="max_tokens", type=_positive_int, default=None, help=argparse.SUPPRESS
    )


def _add_export(sub) -> None:
    p = sub.add_parser(
        "export",
        help="Answer every question of a Test CSV and write ID,input_prompt,answer",
        description="Retrieve, build the prompt, call the generator and write the submission CSV. "
        "Never reads reference answers.",
        epilog=EXPORT_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for add_group in (
        _add_export_io,
        _add_run_options,
        _add_embedder_options,
        _add_retrieval_options,
        _add_generation_options,
        _add_hook_options,
    ):
        add_group(p)


def _add_check(sub) -> None:
    p = sub.add_parser(
        "check",
        help="Validate a submission CSV against its Test CSV (format, IDs, answers, "
        "official prompt cap)",
    )
    p.add_argument("--submission", required=True, help="Submission CSV to validate")
    p.add_argument("--questions", required=True, help="The Test CSV the submission answers")


def _add_score(sub) -> None:
    p = sub.add_parser(
        "score",
        help="Score a submission with the official metric scripts",
        description="Runs the vendored official accuracy, token efficiency and overall score "
        "(baseline, cap and minimum accuracy come from them). Separate from export.",
    )
    p.add_argument("--submission", required=True, help="Submission CSV (ID,input_prompt,answer)")
    p.add_argument("--reference", required=True, help="Reference CSV with header ID,answer")


def add_challenge_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "challenge", help="Challenge flow: export a submission CSV, check it, score it"
    )
    sub = parser.add_subparsers(dest="challenge_command", required=True)
    _add_export(sub)
    _add_check(sub)
    _add_score(sub)


def _build_embedder(settings: dict) -> SentenceTransformerEmbedder:
    from open_telco_rag.challenge.embedder import build_custom_embedder
    from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder

    custom = settings["embedder_path"] or settings["embedder_id"]
    if custom:
        return build_custom_embedder(
            custom,
            query_prefix=settings["query_prefix"],
            document_prefix=settings["document_prefix"],
            dimensions=settings["embedder_dimensions"] or 0,
        )
    if settings["query_prefix"] or settings["document_prefix"]:
        sys.exit(
            "error: --query-prefix and --document-prefix need --embedder-path or --embedder-id"
        )
    return SentenceTransformerEmbedder(model_key=settings["model"])


def _build_client(args: argparse.Namespace, settings: dict) -> ChatClient:
    from open_telco_rag.challenge.client import ChatClient

    return ChatClient(
        model=settings["generator_model"], base_url=args.llm_base_url, timeout=settings["timeout"]
    )


def _build_reranker(settings: dict):
    if not settings["rerank"]:
        return None
    from open_telco_rag.embedders.reranker import load_reranker

    return load_reranker(settings["reranker"])


def _warn_if_cap_exceeds_official(settings: dict) -> None:
    from open_telco_rag.challenge.caps import official_cap

    if settings["max_prompt_tokens"] > official_cap():
        print(
            f"[WARN] --max-prompt-tokens {settings['max_prompt_tokens']} is above the official cap "
            f"of {official_cap()}; challenge check and score reject prompts over it",
            file=sys.stderr,
        )


def _check_dimension(vector: list[float], store) -> None:
    held = store._collection.peek(1).get("embeddings")
    if held is not None and len(held) and len(held[0]) != len(vector):
        sys.exit(
            f"error: the embedder produces {len(vector)}-dimensional vectors but the index holds "
            f"{len(held[0])}-dimensional ones; pass the model the index was built with "
            f"(--model, --embedder-path/--embedder-id or --profile)"
        )


def _warm_up(embedder, reranker, store) -> None:
    from open_telco_rag.types import SearchResult

    try:
        vector = embedder.embed(["warm up"], is_query=True)[0]
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"error: cannot run the embedder: {error}")
    _check_dimension(vector, store)
    probe = [SearchResult(text="warm up", distance=0.0, metadata={}, doc_id="warm-up")]
    try:
        if reranker is not None:
            reranker.rerank("warm up", probe, top_k=1)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"error: cannot run the reranker: {error}")


def _load_models(settings: dict, store) -> tuple:
    try:
        embedder = _build_embedder(settings)
    except (ValueError, OSError, ImportError) as error:
        sys.exit(f"error: cannot load the embedder: {error}")
    try:
        reranker = _build_reranker(settings)
    except (ValueError, OSError, ImportError) as error:
        sys.exit(f"error: cannot load the reranker: {error}")
    _warm_up(embedder, reranker, store)
    return embedder, reranker


def _make_retrieve(settings: dict, store, embedder, reranker) -> Callable[[str], list]:
    from open_telco_rag.core.retrieval import execute_retrieval

    def retrieve(query: str) -> list:
        return execute_retrieval(
            query,
            store,
            embedder,
            top_k=settings["top_k"],
            reranker=reranker,
            rerank_top_k=settings["rerank_top_k"],
            search_method=settings["search_method"],
            resolve_parents=settings["resolve_parents"],
        )

    return retrieve


def _load_export_inputs(args: argparse.Namespace) -> tuple[dict, list, object]:
    from open_telco_rag.challenge.index import IndexProblem, open_index
    from open_telco_rag.challenge.questions import load_questions

    try:
        settings = resolve_settings(args)
        questions = load_questions(args.questions)
        store = open_index(args.db_path, args.collection, settings["search_method"])
    except (IndexProblem, ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    return settings, questions, store


def _export_options(args: argparse.Namespace, settings: dict) -> ExportOptions:
    from open_telco_rag.challenge.export import ExportOptions, stderr_progress
    from open_telco_rag.challenge.hooks import load_callable

    hooks = {}
    try:
        if args.prompt_builder:
            hooks["prompt_builder"] = load_callable(args.prompt_builder)
        if args.answer_parser:
            hooks["answer_parser"] = load_callable(args.answer_parser)
    except ValueError as error:
        sys.exit(f"error: {error}")
    return ExportOptions(
        max_prompt_tokens=settings["max_prompt_tokens"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        seed=settings["seed"],
        resume=args.resume,
        progress=stderr_progress,
        **hooks,
    )


def _failure_note(out: str) -> str:
    stale = f"\n{out} is from an earlier run and was not updated" if Path(out).exists() else ""
    return f"\npartial output kept at {out}.partial{stale}"


def _prepare_run(args: argparse.Namespace, settings: dict, store) -> None:
    from open_telco_rag.challenge.guards import (
        check_embedder_matches_index,
        check_flags_have_an_effect,
    )
    from open_telco_rag.challenge.seed import set_all_seeds

    print(describe(settings), file=sys.stderr)
    check_flags_have_an_effect(args, settings, store)
    _warn_if_cap_exceeds_official(settings)
    check_embedder_matches_index(args, settings, store)
    if settings["seed"] is not None:
        set_all_seeds(settings["seed"])


def _resume_note(result) -> str:
    if result.resumed_rows:
        return f"[resume] kept {result.resumed_rows} finished rows from the partial file"
    return "[resume] no usable partial file; started from the beginning"


def _run_export(args: argparse.Namespace) -> None:
    from open_telco_rag.challenge.export import ExportAborted, export_submission
    from open_telco_rag.challenge.prompt import PromptTooLong

    settings, questions, store = _load_export_inputs(args)
    _prepare_run(args, settings, store)
    options = _export_options(args, settings)
    embedder, reranker = _load_models(settings, store)
    retrieve = _make_retrieve(settings, store, embedder, reranker)
    try:
        result = export_submission(
            questions, retrieve, _build_client(args, settings), args.out, options
        )
    except (ExportAborted, PromptTooLong) as error:
        sys.exit(f"error: {error}{_failure_note(args.out)}")
    except OSError as error:
        sys.exit(f"error: cannot write {args.out}.partial: {error}")
    if args.resume:
        print(_resume_note(result), file=sys.stderr)
    print(
        f"Wrote {result.n_rows} rows to {result.path} "
        f"({result.total_prompt_tokens} prompt tokens, {result.empty_answers} empty answers)"
    )


def _run_check(args: argparse.Namespace) -> None:
    from open_telco_rag.challenge.check import check_submission

    try:
        report = check_submission(args.submission, args.questions)
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    for issue in report.issues[:50]:
        print(f"ERROR: {issue}")
    if len(report.issues) > 50:
        print(f"... {len(report.issues) - 50} more")
    print(
        f"{report.n_rows} rows, {report.total_prompt_tokens} prompt tokens, "
        f"{len(report.issues)} errors"
    )
    if not report.ok:
        sys.exit(1)


def _run_score(args: argparse.Namespace) -> None:
    from open_telco_rag.challenge.metrics import overall_score
    from open_telco_rag.challenge.score import MetricError, score_submission

    try:
        result = score_submission(args.submission, args.reference)
    except MetricError as error:
        print(f"error: {error}")
        sys.exit(1)
    except OSError as error:
        sys.exit(f"error: {error}")
    print(f"Accuracy:          {result.accuracy:.2f}")
    print(f"Token efficiency:  {result.token_efficiency:.2f}")
    print(f"Overall:           {result.overall:.2f}")
    verdict = "met" if result.minimum_met else "NOT met, not eligible for code evaluation"
    print(f"Minimum accuracy ({overall_score.MIN_ACCURACY}%): {verdict}")


def run_challenge(args: argparse.Namespace) -> None:
    {"export": _run_export, "check": _run_check, "score": _run_score}[args.challenge_command](args)
