"""Unified CLI for Open Telco RAG."""

from __future__ import annotations
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import sys
from pathlib import Path

# Package-relative configs/ fallback when the CWD-relative lookup misses.
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _resolve_config_path(filename: str) -> Path | None:
    """Finds `<filename>` under configs/: CWD-relative first (local override),
    then package-relative (see _CONFIGS_DIR above). None if neither exists.
    """
    cwd_relative = Path("configs") / filename
    if cwd_relative.exists():
        return cwd_relative
    package_relative = _CONFIGS_DIR / filename
    if package_relative.exists():
        return package_relative
    return None


def _load_chunking_strategies() -> list[str]:
    import json
    path = _resolve_config_path("chunking_strategies.json")
    if path is None:
        return ["text_baseline"]
    return list(json.loads(path.read_text(encoding="utf-8")).keys())


def _load_enrichment_strategies() -> list[str]:
    import json
    path = _resolve_config_path("enrichment_strategies.json")
    if path is None:
        return ["none"]
    return list(json.loads(path.read_text(encoding="utf-8")).keys())


def _strategy_requires_llm(strategy: str) -> bool:
    """Whether a chunking strategy needs an LLM, per its requires_llm flag in
    chunking_strategies.json.
    """
    import json
    path = _resolve_config_path("chunking_strategies.json")
    if path is None:
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get(strategy, {})
        return bool(meta.get("requires_llm", False))
    except Exception:
        return False


def _enrichment_requires_llm(enrichment: str) -> bool:
    """Same as _strategy_requires_llm, for enrichment strategies."""
    import json
    path = _resolve_config_path("enrichment_strategies.json")
    if path is None:
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get(enrichment, {})
        return bool(meta.get("requires_llm", False))
    except Exception:
        return False

# exit status of `evaluate` when every question of a corpus failed (1 stays "error")
EXIT_ALL_QUESTIONS_FAILED = 3


def main(argv: list[str] | None = None) -> None:
    # Force UTF-8 so emoji don't crash cp1252 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="open-telco-rag",
        description="Open RAG pipeline for telecom specifications.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest documents into vector database")
    p_ingest.add_argument("source_type",
                          choices=["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi"],
                          help="Document source type")
    p_ingest.add_argument("--data",       default=None,           help="Data directory")
    p_ingest.add_argument("--model",      default="minilm",       help="Embedding model key")
    p_ingest.add_argument("--db-path",    default="chroma_db",    help="ChromaDB path")
    # defaults (1024 / 200) applied in _run_ingest so explicit values can be detected
    p_ingest.add_argument("--chunk-size", type=int, default=None, help="Chunk size in chars (default 1024)")
    p_ingest.add_argument("--overlap",    type=int, default=None, help="Chunk overlap in chars (default 200)")
    p_ingest.add_argument("--target-tokens", type=int, default=None,
                          help="Optional: size chunks in tokens (cl100k_base) instead of characters -- "
                               "target size per chunk. Cannot be combined with --chunk-size/--overlap.")
    p_ingest.add_argument("--max-tokens", type=int, default=None,
                          help="Optional: hard cap on tokens per chunk (defaults to --target-tokens). "
                               "Cannot be combined with --chunk-size/--overlap.")
    p_ingest.add_argument("--overlap-tokens", type=int, default=None,
                          help="Optional: token overlap between chunks; text_baseline and "
                               "sliding_window_tokens only, and requires --target-tokens/--max-tokens.")
    p_ingest.add_argument("--no-resume",  action="store_true",    help="Disable resume")
    p_ingest.add_argument("--collection", default=None,
                          help="Override collection name (default: source_type_model)")
    p_ingest.add_argument("--use-marker", action="store_true", default=False,
                          help="Use Marker API for PDF conversion (better table/structure quality). "
                               "Requires DATALAB_API_KEY in .env. "
                               "Recommended for ETSI and TM Forum corpora.")
    p_ingest.add_argument("--enrichment",
                          choices=_load_enrichment_strategies(),
                          default="none",
                          help="Select a metadata/content enrichment strategy dynamically mapped from enrichment_strategies.json.")
    p_ingest.add_argument("--enrichment-model",
                          default="google/qwen-2.5-coder-32b",
                          help="The LLM model string to use when running an LLM-based enrichment strategy "
                               "(llm_metadata). Sent as the 'model' field to whichever server --llm-backend/"
                               "--llm-base-url point at (same connection lumber_chunker uses for chunking, "
                               "e.g. a local vLLM server) — distinct from --chunking-model, since chunking "
                               "and enrichment can request different model names from that same server.")
    p_ingest.add_argument("--strategy",
                          choices=_load_chunking_strategies(),
                          default="text_baseline",
                          help="The underlying text segmentation methodology mapped from chunking_strategies.json.")
    p_ingest.add_argument("--distance-metric",
                          choices=["l2", "cosine", "ip"],
                          default="l2",
                          help="Distance metric used for indexing HNSW vector space.")
    p_ingest.add_argument("--embed-backend", choices=["local", "openrouter"], default="local",
                      help="Backend for computing embeddings (default: local)")
    p_ingest.add_argument("--batch-size", type=int, default=None,
                          help="Number of chunks to send per embedding call (defaults to 256 for local, 500 for openrouter)")
    p_ingest.add_argument("--embed-max-workers", type=int, default=None,
                          help="OpenRouterEmbedder only: in-flight sub-batch requests per embed() call "
                               "(default: 10). Ignored for --embed-backend local.")
    p_ingest.add_argument("--llm-backend", choices=["local", "openrouter"], default="local",
                          help="Backend for chunking strategies (e.g. lumber_chunker) and/or enrichment "
                               "strategies (llm_metadata) that require an LLM — shared by both, since a "
                               "single vLLM server is the common case. 'local' talks to any OpenAI-compatible "
                               "server (vLLM, Ollama, etc.) on this machine — no GPU/server needed unless "
                               "your chosen strategy/enrichment actually requires an LLM. 'openrouter' reuses "
                               "OPENROUTER_API_KEY, useful for testing without a GPU.")
    p_ingest.add_argument("--llm-base-url", default=None,
                          help="Base URL for the local LLM server, used for both chunking's and enrichment's "
                               "LLM calls (default: env var LOCAL_LLM_BASE_URL, or http://localhost:8000/v1). "
                               "Ignored when --llm-backend=openrouter.")
    p_ingest.add_argument("--chunking-model", default=None,
                          help="Model name/ID to request from the LLM for chunking strategies that need one "
                               "(e.g. lumber_chunker). Default: env var LOCAL_LLM_MODEL, or a sensible "
                               "per-backend fallback. Distinct from --enrichment-model, which is for "
                               "enrichment strategies, not chunking.")

    # ── search ────────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Search the vector database")
    p_search.add_argument("query",                                  help="Search query")
    p_search.add_argument("--source-type",    default="oran",      help="Source type")
    p_search.add_argument("--model",          default="minilm",    help="Embedding model key")
    p_search.add_argument("--db-path",        default="chroma_db", help="ChromaDB path")
    p_search.add_argument("--top-k",          type=int, default=5, help="Number of results")
    p_search.add_argument("--rerank",         action="store_true", help="Enable reranking")
    p_search.add_argument("--rerank-top-k",   type=int, default=5, help="Chunks to keep after reranking")
    p_search.add_argument("--reranker-model", default="ms-marco-minilm",
                          help="Reranker model key from configs/reranker_models.json "
                               "(default: ms-marco-minilm, telecom: otel-reranker-0.6b)")
    p_search.add_argument("--collection",     default=None,        help="Override collection name")
    p_search.add_argument("--search-method", choices=["dense", "bm25", "hybrid"], default="dense",
                          help="Multi-method search routing pass algorithm selection.")
    p_search.add_argument("--embed-backend", choices=["local", "openrouter"], default="local",
                      help="Backend for computing embeddings (default: local)")
    p_search.add_argument("--resolve-parents", action="store_true",
                      help="Expand retrieved child chunks into full parent contexts")

    # ── ask ───────────────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a question (search + generate)")
    p_ask.add_argument("question",                                   help="Your question")
    p_ask.add_argument("--source-type",    default="oran",          help="Source type")
    p_ask.add_argument("--model",          default="minilm",        help="Embedding model key")
    p_ask.add_argument("--db-path",        default="chroma_db",     help="ChromaDB path")
    p_ask.add_argument("--top-k",          type=int, default=8,     help="Chunks to retrieve")
    p_ask.add_argument("--rerank",         action="store_true",     help="Enable reranking")
    p_ask.add_argument("--rerank-top-k",   type=int, default=5,     help="Chunks to keep after reranking")
    p_ask.add_argument("--reranker-model", default="ms-marco-minilm",
                       help="Reranker model key from configs/reranker_models.json "
                            "(default: ms-marco-minilm, telecom: otel-reranker-0.6b)")
    p_ask.add_argument("--collection",     default=None,            help="Override collection name")
    p_ask.add_argument("--search-method", choices=["dense", "bm25", "hybrid"], default="dense",
                          help="Multi-method search routing pass algorithm selection.")
    p_ask.add_argument("--embed-backend", choices=["local", "openrouter"], default="local",
                      help="Backend for computing embeddings (default: local)")
    p_ask.add_argument("--resolve-parents", action="store_true",
                   help="Expand retrieved child chunks into full parent contexts")
    p_ask.add_argument("--generator", choices=["claude", "openrouter", "local"], default="openrouter",
                       help="LLM backend for the answer (default: openrouter). 'local' talks to any "
                            "OpenAI-compatible server via --llm-base-url/--llm-api-key.")
    p_ask.add_argument("--generator-model", default=None,
                       help="Model id/key for --generator. Defaults: openrouter -> "
                            "meta-llama/llama-3.1-8b-instruct; local -> whatever your server serves; "
                            "claude -> the ClaudeGenerator default.")
    p_ask.add_argument("--llm-base-url", default=None,
                       help="Base URL for --generator local (default: env LOCAL_LLM_BASE_URL, "
                            "else http://localhost:8000/v1). Ignored for claude/openrouter.")
    p_ask.add_argument("--llm-api-key", default=None,
                       help="API key for --generator local (default: env LOCAL_LLM_API_KEY). "
                            "Ignored for claude/openrouter.")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Run evaluation suite on a corpus")
    p_eval.add_argument("corpus",
                        choices=["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi", "all"],
                        help="Corpus to evaluate, or 'all' to run all corpora sequentially")
    p_eval.add_argument("--model",          default="minilm",    help="Embedding model key")
    p_eval.add_argument("--db-path",        default="chroma_db", help="ChromaDB path")
    p_eval.add_argument("--collection",     default=None,
                        help="Override collection name (default: corpus_model). "
                             "Use this when your collection has a custom name, "
                             "e.g. 3gpp_r19_marker_bge")
    p_eval.add_argument("--questions-file", default=None,
                        help="Path to a custom QnA JSON file. "
                             "Must follow the same format as configs/eval_questions/{corpus}.json. "
                             "If omitted, loads from configs/eval_questions/{corpus}.json")
    p_eval.add_argument("--top-k",          type=int, default=5,  help="Chunks to retrieve per question")
    p_eval.add_argument("--search-method", choices=["dense", "bm25", "hybrid"], default="dense",
                        help="Retrieval method (default: dense), same choices as search/ask")
    p_eval.add_argument("--record-judge-responses", action="store_true",
                        help="Optional: store every judge prompt and raw response per question in the "
                             "results file (large: prompts include the retrieved chunks). Off by default.")
    p_eval.add_argument("--threshold",      type=float, default=0.5,
                        help="Similarity threshold for coverage_ratio metric (default: 0.5)")
    p_eval.add_argument("--output",         default="results",
                        help="Directory to save result JSON files (default: results/)")
    p_eval.add_argument("--delay",          type=float, default=1.0,
                        help="Seconds between questions — rate limits API calls (default: 1.0)")
    p_eval.add_argument("--no-save",        action="store_true",
                        help="Print summary only, do not save JSON results")
    p_eval.add_argument("--generator",      choices=["claude", "openrouter", "local"], default="openrouter",
                        help="LLM provider for evaluation metrics (default: openrouter). "
                             "'local' talks to any OpenAI-compatible server on this machine "
                             "(e.g. your GPU node) — see --llm-base-url/--llm-api-key.")
    p_eval.add_argument("--generator-model", default=None,
                        help=("Model for evaluation metrics. "
                              "Claude default: claude-sonnet-4-20250514. "
                              "OpenRouter default: meta-llama/llama-3.3-70b-instruct. "
                              "Local: required — whatever your server is serving."))
    p_eval.add_argument("--llm-base-url", default=None,
                        help="Base URL for --generator local (default: env var LOCAL_LLM_BASE_URL, "
                             "or http://localhost:8000/v1). Ignored for claude/openrouter.")
    p_eval.add_argument("--llm-api-key", default=None,
                        help="API key for --generator local, sent as 'Authorization: Bearer <key>' "
                             "(default: env var LOCAL_LLM_API_KEY). Ignored for claude/openrouter. "
                             "Omitted entirely if unset — unauthenticated local servers still work.")
    p_eval.add_argument("--rag-generator",   choices=["claude", "openrouter", "local"], default=None,
                        help="LLM provider for RAG answer generation. "
                             "If omitted, uses the same as --generator.")
    p_eval.add_argument("--rag-generator-model", default=None,
                        help="Model for RAG answer generation. "
                             "If omitted, uses the same as --generator-model.")
    p_eval.add_argument("--rag-llm-base-url", default=None,
                        help="Base URL for --rag-generator local. If omitted, uses --llm-base-url "
                             "(so a single GPU endpoint can serve both metrics judging and RAG "
                             "generation unless you point them at two different servers/ports).")
    p_eval.add_argument("--rag-llm-api-key", default=None,
                        help="API key for --rag-generator local. If omitted, uses --llm-api-key.")
    p_eval.add_argument("--debug",          action="store_true",
                        help="Log raw LLM responses for every metric call")
    p_eval.add_argument("--leaderboard",    default="results/leaderboard.json",
                        help="Path to leaderboard JSON file (default: results/leaderboard.json).")
    p_eval.add_argument("--no-leaderboard", action="store_true",
                        help="Do not update the leaderboard file")
    p_eval.add_argument("--include-baseline", action="store_true",
                        help="Run a second pass with no retrieval (LLM knowledge only) "
                             "and store baseline metrics alongside RAG metrics in the leaderboard")
    p_eval.add_argument("--rerank",         action="store_true",
                        help="Enable reranking during evaluation retrieval")
    p_eval.add_argument("--rerank-top-k",   type=int, default=5,
                        help="Chunks to keep after reranking (default: 5)")
    p_eval.add_argument("--retrieval-only", action="store_true",
                        help="Optional: run retrieval and the retrieval metrics only -- no answer "
                             "generation, no judge/LLM calls, no API keys needed. Does not update "
                             "the leaderboard. With --include-baseline the baseline pass still runs.")
    p_eval.add_argument("--context-token-budget", type=int, default=None,
                        help="Optional: fill the generator's context up to this many tokens (cl100k_base) "
                             "instead of a fixed --top-k/--rerank-top-k chunk count.")
    p_eval.add_argument("--candidate-pool", type=int, default=None,
                        help="With --context-token-budget: candidates to retrieve and rerank before "
                             "filling the budget (default 30).")
    p_eval.add_argument("--budget-overflow", choices=["stop", "skip", "truncate"], default=None,
                        help="With --context-token-budget: what to do with the first chunk that does not "
                             "fit (default stop).")
    p_eval.add_argument("--reranker-model", default="ms-marco-minilm",
                        help="Reranker model key from configs/reranker_models.json "
                             "(default: ms-marco-minilm, telecom: otel-reranker-0.6b)")
    p_eval.add_argument("--embed-backend", choices=["local", "openrouter"], default="local",
                      help="Backend for computing embeddings (default: local)")
    p_eval.add_argument("--inject-metadata", action="store_true",
                    help="Simulate domain expert queries by prepending canonical source metadata to the question.")
    p_eval.add_argument("--run-name", default=None,
                        help="Custom label to identify this evaluation run (e.g. 'exp_prompt_v2').")
    p_eval.add_argument("--include-keywords", action="store_true",
                        help="Append question keywords to the search query")
    p_eval.add_argument("--resolve-parents", action="store_true",
                    help="Expand retrieved child chunks into full parent contexts")

    # ── info ──────────────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show database info")
    p_info.add_argument("--db-path",    default="chroma_db", help="ChromaDB path")
    p_info.add_argument("--collection", default=None,        help="Collection name")


    # ── delete-collection ─────────────────────────────────────────────────────
    p_delete = sub.add_parser("delete", help="Delete a collection from the vector database")
    p_delete.add_argument("collection", help="The name of the collection to delete")
    p_delete.add_argument("--db-path", default="chroma_db", help="ChromaDB path (default: chroma_db)")


    # ── push-hf ───────────────────────────────────────────────────────────────
    p_push_hf = sub.add_parser("push-hf", help="Push local Chroma+FTS database folder to HF Hub")
    p_push_hf.add_argument("db_path", help="Path to local database directory (e.g. ./runs/run_name)")
    p_push_hf.add_argument("repo_id", help="Target HF dataset repo ID (e.g. <your-hf-username>/telco-rag-grid)")
    p_push_hf.add_argument("--path-in-repo", default=None, help="Folder name inside HF repository")
    p_push_hf.add_argument("--public", action="store_true", help="Make dataset public (default: private)")

    # ── pull-hf ───────────────────────────────────────────────────────────────
    p_pull_hf = sub.add_parser("pull-hf", help="Pull database folder from HF Hub for evaluation")
    p_pull_hf.add_argument("repo_id", help="HF dataset repo ID (e.g. <your-hf-username>/telco-rag-grid)")
    p_pull_hf.add_argument("path_in_repo", help="Folder name inside HF repository (e.g. run_name)")
    p_pull_hf.add_argument("--local-dir", default="./runs", help="Local directory to store downloaded DB")


    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "search":
        _run_search(args)
    elif args.command == "ask":
        _run_ask(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "info":
        _run_info(args)
    elif args.command == "delete":
        _run_delete_collection(args)
    elif args.command == "push-hf":
        from open_telco_rag.stores.hf import push_collection_to_hf
        push_collection_to_hf(
            db_path=args.db_path,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            private=not args.public,
        )
    elif args.command == "pull-hf":
        from open_telco_rag.stores.hf import pull_collection_from_hf
        pull_collection_from_hf(
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            local_dir=args.local_dir,
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_source(source_type: str, use_marker: bool = False):
    source_type_clean = source_type.lower()
    if source_type_clean in ("oran",):
        from open_telco_rag.sources.oran import ORANSource
        return ORANSource(use_marker=use_marker)
    elif source_type_clean in ("3gpp", "tpp"):
        from open_telco_rag.sources.tgpp import TPPSource
        return TPPSource()
    elif source_type_clean in ("camara",):
        from open_telco_rag.sources.camara import CAMARASource
        return CAMARASource()
    elif source_type_clean in ("tmforum",):
        from open_telco_rag.sources.tmforum import TMForumSource
        return TMForumSource(use_marker=use_marker)
    elif source_type_clean in ("itu",):
        from open_telco_rag.sources.itu import ITUSource
        return ITUSource(use_marker=use_marker)
    elif source_type_clean in ("etsi",):
        from open_telco_rag.sources.etsi import ETSISource
        return ETSISource(use_marker=use_marker)
    elif source_type_clean in ("gsma",):
        from open_telco_rag.sources.gsma import GSMASource
        return GSMASource(use_marker=use_marker)
    raise NotImplementedError(f"Source '{source_type}' not yet implemented.")



def _get_embedder(model_key: str, backend: str = "local", max_workers: int | None = None):
    if backend == "openrouter":
        from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
        kwargs = {"model_key": model_key}
        if max_workers is not None:
            kwargs["max_workers"] = max_workers
        return OpenRouterEmbedder(**kwargs)
    else:
        from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_key=model_key)


def _get_store(source_type: str, model_key: str, db_path: str, collection: str = None, distance_metric: str = "l2"):
    from open_telco_rag.stores.chroma import ChromaStore
    col = collection or f"{source_type}_{model_key}"

    metric_map = {"l2": "l2", "cosine": "cosine", "ip": "ip"}
    chosen_space = metric_map.get(distance_metric.lower(), "l2")

    return ChromaStore(
        path=db_path,
        collection_name=col,
        metadata={
            "hnsw:space": chosen_space,
            # Bulk-ingest tuning: fewer HNSW syncs on large ingests.
            # Collection-creation only; write-throughput only, not search.
            "hnsw:batch_size": 1000,
            "hnsw:sync_threshold": 10000,
        }
    )


def _get_reranker(model_key: str):
    """Load reranker by key from configs/reranker_models.json."""
    from open_telco_rag.embedders.reranker import load_reranker
    return load_reranker(model_key)


def _get_generator(name: str, model: str | None = None,
                   base_url: str | None = None, api_key: str | None = None):
    """RAG-answer generator for `ask` — claude | openrouter | local.
    Keep in sync with evaluation.evaluate._build_generator."""
    if name == "claude":
        from open_telco_rag.generators.claude import ClaudeGenerator
        return ClaudeGenerator(model=model) if model else ClaudeGenerator()
    if name == "local":
        from open_telco_rag.generators.local import LocalGenerator
        return LocalGenerator(model=model or "local-model", base_url=base_url, api_key=api_key)
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    return OpenRouterGenerator(model=model or "meta-llama/llama-3.1-8b-instruct")


def _get_llm_client(backend: str = "local", base_url: str | None = None, api_key: str | None = None):
    """LLM client for strategies needing one (e.g. lumber_chunker). api_key
    applies only to backend="local" (falls back to LOCAL_LLM_API_KEY).
    """
    if backend == "openrouter":
        from open_telco_rag.generators.local_llm import OpenRouterLLMClient
        return OpenRouterLLMClient()
    else:
        from open_telco_rag.generators.local_llm import LocalLLMClient
        return LocalLLMClient(base_url=base_url, api_key=api_key)


# ── Command runners ───────────────────────────────────────────────────────────

def _run_ingest(args) -> None:
    import json
    from pathlib import Path
    from open_telco_rag.core.pipeline import ingest

    use_marker = getattr(args, "use_marker", False)
    enrichment = getattr(args, "enrichment", "none")
    enrichment_model = getattr(args, "enrichment_model", "google/qwen-2.5-coder-32b")
    distance_metric  = getattr(args, "distance_metric", "l2")
    strategy = getattr(args, "strategy", "text_baseline")
    
    source     = _get_source(args.source_type, use_marker=use_marker)

    if use_marker and not os.environ.get("DATALAB_API_KEY"):
        # Fail fast; else each Phase 1 worker discovers it per file.
        print("\nDATALAB_API_KEY not set - required for --use-marker.")
        print("   Add it to your .env file: DATALAB_API_KEY=your_key_here")
        sys.exit(1)

    embed_backend = getattr(args, "embed_backend", "local")
    # 256 matches the GPU micro-batch size; override with --batch-size on tight hardware.
    default_batch_size = 500 if embed_backend == "openrouter" else 256
    batch_size_arg = getattr(args, "batch_size", None) or default_batch_size
    embed_max_workers = getattr(args, "embed_max_workers", None)
    embedder = _get_embedder(args.model, backend=embed_backend, max_workers=embed_max_workers)
    store      = _get_store(args.source_type, args.model, args.db_path,
                            collection=getattr(args, "collection", None),
                            distance_metric=distance_metric)
    data_dir   = args.data or f"data/{args.source_type}"
    

    print(f"\nIngesting {args.source_type.upper()} documents")
    print(f"  Data dir:   {data_dir}")
    print(f"  Model:      {args.model}")
    print(f"  Collection: {store.collection_name}")
    print(f"  PDF loader: {'Marker API (high-quality)' if use_marker else 'pypdf (default)'}")
    print(f"  Enrichment: {enrichment.upper()} ({enrichment_model})")
    print(f"  Strategy:   {strategy.upper()}")
    print(f"  Distance:   {distance_metric.upper()}")

    chunk_kwargs = {
        "source": source,
        "store": store,
        "embedder": embedder,
        "data_dir": data_dir,
        "resume": not args.no_resume,
        "enrichment": enrichment,
        "enrichment_model": enrichment_model,
        "strategy": strategy,
        "batch_size": batch_size_arg,
    }

    # LLM chunking/enrichment kick in per their requires_llm flags; both
    # share one --llm-backend/--llm-base-url connection.
    needs_chunking_llm = _strategy_requires_llm(strategy)
    needs_enrichment_llm = _enrichment_requires_llm(enrichment)

    if needs_chunking_llm or needs_enrichment_llm:
        llm_backend = getattr(args, "llm_backend", "local")
        llm_base_url = getattr(args, "llm_base_url", None)
        chunking_model = (
            getattr(args, "chunking_model", None)
            or os.environ.get("LOCAL_LLM_MODEL")
            or ("openai/gpt-4o-mini" if llm_backend == "openrouter" else "Qwen/Qwen2.5-32B-Instruct")
        )

        llm_client = _get_llm_client(backend=llm_backend, base_url=llm_base_url)

        resolved_base = llm_base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        needs_desc = " + ".join(
            d for d, needed in
            [("chunking", needs_chunking_llm), ("enrichment", needs_enrichment_llm)]
            if needed
        )
        print(f"  LLM backend: {llm_backend.upper()} "
              f"({resolved_base if llm_backend == 'local' else 'openrouter.ai'}) "
              f"— model: {chunking_model}  [{needs_desc}]")

        # Fail fast; else each worker retries a bad endpoint per file.
        print("  Checking LLM connectivity...", end=" ", flush=True)
        try:
            llm_client.completion.create(
                model=chunking_model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=5,
            )
            print("OK")
        except Exception as e:
            print("FAILED")
            print(f"\nCould not reach the LLM before starting ingestion: {e}")
            print(f"   Backend: {llm_backend}  |  Model: {chunking_model}"
                  + (f"  |  URL: {resolved_base}" if llm_backend == "local" else ""))
            if llm_backend == "local":
                print(f"   Check the server is running and reachable, e.g.:  curl {resolved_base}/models")
            sys.exit(1)

        if needs_chunking_llm:
            chunk_kwargs["llm_client"] = llm_client
            chunk_kwargs["model_name"] = chunking_model
        if needs_enrichment_llm:
            # Distinct key from chunking's llm_client; popped before chunk_document().
            chunk_kwargs["enrichment_llm_client"] = llm_client

    chunk_size = getattr(args, "chunk_size", None)
    overlap    = getattr(args, "overlap", None)
    target_tokens  = getattr(args, "target_tokens", None)
    max_tokens     = getattr(args, "max_tokens", None)
    overlap_tokens = getattr(args, "overlap_tokens", None)

    if target_tokens is not None or max_tokens is not None:
        # token sizing: don't inject the char defaults
        if chunk_size is not None or overlap is not None:
            sys.exit("--target-tokens/--max-tokens cannot be combined with --chunk-size/--overlap; "
                     "use token sizing or character sizing, not both.")
        if overlap_tokens is not None and strategy not in ("text_baseline", "sliding_window_tokens"):
            sys.exit(f"--overlap-tokens is not supported for the {strategy} strategy.")
        chunk_kwargs["target_tokens"] = target_tokens
        chunk_kwargs["max_tokens"] = max_tokens
        if overlap_tokens is not None:
            chunk_kwargs["overlap_tokens"] = overlap_tokens
        print(f"  Chunk size: TOKENS (target={target_tokens}, max={max_tokens or target_tokens})")
    else:
        if overlap_tokens is not None:
            sys.exit("--overlap-tokens requires --target-tokens or --max-tokens.")
        chunk_size = 1024 if chunk_size is None else chunk_size
        overlap    = 200 if overlap is None else overlap
        # pull structural constraints from the strategy definition json
        strat_path = Path("chunking_strategies.json")
        if strat_path.exists():
            try:
                strat_meta = json.loads(strat_path.read_text(encoding="utf-8")).get(strategy, {})
                if strat_meta.get("unit") == "tokens":
                    chunk_kwargs["chunk_tokens"] = chunk_size
                    chunk_kwargs["overlap_tokens"] = overlap
                else:
                    chunk_kwargs["chunk_size"] = chunk_size
                    chunk_kwargs["overlap"] = overlap
            except Exception as e:
                print(f"[Warning] Failed to parse chunking_strategies.json ({e}). Falling back to characters.")
                chunk_kwargs["chunk_size"] = chunk_size
                chunk_kwargs["overlap"] = overlap
        else:
            chunk_kwargs["chunk_size"] = chunk_size
            chunk_kwargs["overlap"] = overlap

    stats = ingest(**chunk_kwargs)

    print(f"\nFiles processed: {stats.files_processed}")
    print(f"Chunks stored:   {stats.chunks_stored}")
    print(f"Skipped:         {stats.skipped_files}")
    print(f"Errors:          {stats.error_files}")

def _run_search(args) -> None:
    import math
    from open_telco_rag.core.pipeline import search

    embed_backend = getattr(args, "embed_backend", "local")
    embedder = _get_embedder(args.model, backend=embed_backend)
    store    = _get_store(args.source_type, args.model, args.db_path,
                          collection=getattr(args, "collection", None))

    reranker = None
    if args.rerank:
        reranker = _get_reranker(args.reranker_model)
        print(f"Reranking enabled ({args.reranker_model}) — "
              f"{args.top_k * 3} candidates -> top {args.rerank_top_k}")

    results = search(
        args.query,
        store        = store,
        embedder     = embedder,
        top_k        = args.top_k,
        reranker     = reranker,
        rerank_top_k = args.rerank_top_k,
        search_method = getattr(args, "search_method", "dense"),
        resolve_parents = getattr(args, "resolve_parents", False),
    )

    if not results:
        print("No results found.")
        return

    seen = set()
    displayed = 0
    for r in results:
        key = f"{r.metadata.get('source','')}::{r.metadata.get('heading','')}"
        if key in seen:
            continue
        seen.add(key)
        displayed += 1
        heading      = r.metadata.get("heading", "")
        source       = r.metadata.get("source", "")
        wg           = r.metadata.get("work_group", "")
        release      = r.metadata.get("release", "")
        similarity   = round(math.exp(-r.distance), 3)
        rerank_score = r.metadata.get("rerank_score", "")
        rerank_str   = f" | rerank {rerank_score}" if rerank_score != "" else ""
        print(f"\n--- Result {displayed} | {source}"
              f"{' | ' + wg if wg else ''}"
              f"{' | ' + release if release else ''}"
              f"{' | ' + heading if heading else ''}"
              f" | similarity {similarity} | L2 {r.distance:.3f}{rerank_str} ---")
        print(r.text[:600])

    print(f"\n{displayed} results returned for: '{args.query}'")


def _run_ask(args) -> None:
    from open_telco_rag.core.pipeline import generate, search

    embed_backend = getattr(args, "embed_backend", "local")
    embedder = _get_embedder(args.model, backend=embed_backend)
    store    = _get_store(args.source_type, args.model, args.db_path,
                          collection=getattr(args, "collection", None))

    reranker = None
    if args.rerank:
        reranker = _get_reranker(args.reranker_model)
        print(f"Reranking enabled ({args.reranker_model}) — "
              f"{args.top_k * 3} candidates -> top {args.rerank_top_k}")

    results = search(
        args.question,
        store        = store,
        embedder     = embedder,
        top_k        = args.top_k,
        reranker     = reranker,
        rerank_top_k = args.rerank_top_k,
        search_method = getattr(args, "search_method", "dense"),
        resolve_parents = getattr(args, "resolve_parents", False),
    )
    generator = _get_generator(
        getattr(args, "generator", "openrouter"),
        model    = getattr(args, "generator_model", None),
        base_url = getattr(args, "llm_base_url", None),
        api_key  = getattr(args, "llm_api_key", None),
    )
    answer = generate(args.question, results, generator=generator, source_type=args.source_type)

    print(f"\nAnswer:\n{answer['answer']}")
    print(f"\nSources:     {answer['sources']}")
    print(f"Chunks used: {answer['chunks_used']}")


def _run_evaluate(args) -> None:
    from open_telco_rag.evaluation.evaluate import evaluate_corpus
    from open_telco_rag.evaluation.questions import SUPPORTED_CORPORA, load_questions, EvalQuestion
    from open_telco_rag.evaluation._claude import make_llm_call
    from open_telco_rag.core.retrieval import resolve_budget_args

    budget_kwargs = {
        "context_token_budget": getattr(args, "context_token_budget", None),
        "candidate_pool": getattr(args, "candidate_pool", None),
        "budget_overflow": getattr(args, "budget_overflow", None),
    }
    try:
        resolved = resolve_budget_args(**budget_kwargs)
    except ValueError as e:
        sys.exit(f"error: {e}")
    # resolved values (defaults filled in when a budget is set) for the run + leaderboard row
    budget_kwargs = dict(zip(budget_kwargs, resolved))

    embed_backend = getattr(args, "embed_backend", "local")
    embedder = _get_embedder(args.model, backend=embed_backend)
    output_dir = None if args.no_save else Path(args.output)

    retrieval_only = getattr(args, "retrieval_only", False)
    no_llm = retrieval_only and not args.include_baseline

    # RAG base_url/api_key fall back to the metrics ones unless given their own.
    rag_generator_name  = args.rag_generator       or args.generator
    rag_generator_model = args.rag_generator_model or args.generator_model
    rag_base_url = getattr(args, "rag_llm_base_url", None) or getattr(args, "llm_base_url", None)
    rag_api_key  = getattr(args, "rag_llm_api_key", None)  or getattr(args, "llm_api_key", None)

    # fail before any work if a Claude generator is selected without a key; the
    # per-question calls would otherwise all fail and the run would still exit 0
    if not no_llm and "claude" in (args.generator, rag_generator_name)             and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: --generator/--rag-generator claude needs ANTHROPIC_API_KEY "
                 "(set it in the environment or a .env file).")

    if no_llm:
        # nothing calls an LLM in retrieval-only mode, so don't build clients
        def _no_llm_call(prompt, max_tokens):
            raise RuntimeError("LLM call attempted in --retrieval-only mode")
        llm_call = rag_llm_call = _no_llm_call
        print("  Mode         : retrieval only (no generation, no LLM calls)")
    else:
        llm_call = make_llm_call(
            args.generator, args.generator_model,
            base_url=getattr(args, "llm_base_url", None),
            api_key=getattr(args, "llm_api_key", None),
        )
        rag_llm_call = make_llm_call(rag_generator_name, rag_generator_model, base_url=rag_base_url, api_key=rag_api_key)

        print(f"  Metrics LLM  : {args.generator} / {args.generator_model or 'default'}")
        print(f"  RAG gen LLM  : {rag_generator_name} / {rag_generator_model or 'default'}")

    custom_questions = None
    if args.questions_file:
        import json
        qpath = Path(args.questions_file)
        if not qpath.exists():
            print(f"[ERROR] Questions file not found: {qpath}")
            sys.exit(1)
        data  = json.loads(qpath.read_text(encoding="utf-8"))
        items = data.get("questions", [])
        if not items:
            print(f"[ERROR] No questions found in {qpath}")
            sys.exit(1)
        custom_questions = [EvalQuestion.from_dict(q) for q in items]
        print(f"[INFO] Loaded {len(custom_questions)} questions from {qpath}")

    corpora = SUPPORTED_CORPORA if args.corpus == "all" else [args.corpus]

    failed_corpora: list[str] = []

    for corpus in corpora:
        collection = args.collection if len(corpora) == 1 else None
        store      = _get_store(corpus, args.model, args.db_path, collection=collection)

        count    = store.count()
        col_name = collection or f"{corpus}_{args.model}"
        if count == 0:
            print(f"\n[SKIP] {corpus.upper()} — collection '{col_name}' "
                  f"is empty. Run: open-telco-rag ingest {corpus}")
            continue

        print(f"\n[INFO] {corpus.upper()} — {count:,} chunks in '{col_name}'")

        questions = custom_questions or None

        reranker = None
        if args.rerank:
            reranker = _get_reranker(args.reranker_model)
            print(f"  Reranker     : {args.reranker_model} -> top {args.rerank_top_k}")

        result, baseline_result = evaluate_corpus(
            corpus            = corpus,
            store             = store,
            embedder          = embedder,
            questions         = questions,
            model             = args.model,
            top_k             = args.top_k,
            sim_threshold     = args.threshold,
            delay             = args.delay,
            output_dir        = output_dir,
            llm_call          = llm_call,
            rag_llm_call      = rag_llm_call,
            include_baseline  = args.include_baseline,
            reranker          = reranker,
            rerank_top_k      = args.rerank_top_k,
            debug             = args.debug,
            rag_generator_model=rag_generator_model,
            rag_generator_name=rag_generator_name,
            rag_base_url      = rag_base_url,
            rag_api_key       = rag_api_key,
            inject_metadata   = args.inject_metadata,
            include_keywords  = getattr(args, "include_keywords", False),
            search_method=getattr(args, "search_method", "dense"),
            resolve_parents   = getattr(args, "resolve_parents", False),
            **budget_kwargs,
            **({"retrieval_only": True} if retrieval_only else {}),
            **({"record_judge_calls": True} if getattr(args, "record_judge_responses", False) else {}),
        )

        if result.question_results and all(q.error for q in result.question_results):
            # nothing was evaluated: say so, keep the saved results, skip the leaderboard row
            failed_corpora.append(corpus)
            print(f"\n[ERROR] Every one of the {len(result.question_results)} questions failed for "
                  f"{corpus.upper()} (see the per-question errors above); no metrics were computed. "
                  f"Results were still saved.", file=sys.stderr)
            if not args.no_leaderboard:
                print("  Leaderboard not updated.")
        elif retrieval_only and not args.no_leaderboard:
            print("\n  Leaderboard not updated (retrieval-only runs have no generation metrics).")
        elif not args.no_leaderboard:
            _update_leaderboard(
                leaderboard_path = Path(args.leaderboard),
                result           = result,
                baseline_result  = baseline_result,
                rag_model        = rag_generator_model or "default",
                evaluator_model  = args.generator_model or "default",
                corpus           = corpus,
                collection       = col_name,
                reranker_model   = args.reranker_model if args.rerank else None,
                rerank_top_k     = getattr(args, "rerank_top_k", result.top_k),
                questions_file   = getattr(args, "questions_file", None),
                inject_metadata  = args.inject_metadata,
                include_keywords = getattr(args, "include_keywords", False),
                search_method=getattr(args, "search_method", "dense"),
                embed_backend=embed_backend,
                run_name         = getattr(args, "run_name", None),
                **budget_kwargs,
            )

    if failed_corpora:
        sys.exit(EXIT_ALL_QUESTIONS_FAILED)




def _update_leaderboard(
    leaderboard_path: Path,
    result,
    rag_model:        str,
    evaluator_model:  str,
    corpus:           str,
    collection:       str,
    baseline_result   = None,
    reranker_model:   str | None = None,
    rerank_top_k:     int = 0,
    questions_file:   str | None = None,
    inject_metadata:  bool = False,
    include_keywords: bool = False,
    search_method:    str = "dense", 
    embed_backend:    str = "local",
    run_name:         str | None = None,
    context_token_budget: int | None = None,
    candidate_pool:   int | None = None,
    budget_overflow:  str | None = None,) -> None:
    """Append one row to the leaderboard JSON file."""
    import json
    from datetime import datetime

    def _avg(r):
        # mean_accuracy included so "average" covers mixed mc+tf+open runs.
        scores = [getattr(r, "mean_accuracy", -1.0), r.mean_faithfulness, r.mean_answer_relevancy,
                  r.mean_similarity, r.mean_context_recall, r.mean_answer_correctness]
        valid = [s for s in scores if s >= 0]
        return round(sum(valid) / len(valid), 3) if valid else -1.0
    
    q_results = getattr(result, "question_results", [])
    
    import statistics
    avg_total = statistics.mean([getattr(q, "latency", 0.0) for q in q_results]) if q_results else 0.0
    avg_search = statistics.mean([getattr(q, "search_latency", 0.0) for q in q_results]) if q_results else 0.0
    avg_rerank = statistics.mean([getattr(q, "rerank_latency", 0.0) for q in q_results]) if q_results else 0.0
    avg_gen = statistics.mean([getattr(q, "gen_latency", 0.0) for q in q_results]) if q_results else 0.0

    total_prompt_tokens = sum([getattr(q, "prompt_tokens", 0) for q in q_results])
    total_completion_tokens = sum([getattr(q, "completion_tokens", 0) for q in q_results])
    total_rerank_tokens = sum([getattr(q, "rerank_tokens", 0) for q in q_results])

    # cached_tokens/reasoning_tokens can be None; `or 0` handles it.
    total_cached_tokens = sum([(getattr(q, "cached_tokens", 0) or 0) for q in q_results])
    total_reasoning_tokens = sum([(getattr(q, "reasoning_tokens", 0) or 0) for q in q_results])

    # Fall back to the pricing-matrix estimate only when no question has a known cost.
    known_costs = [getattr(q, "cost", None) for q in q_results]
    known_costs = [c for c in known_costs if c is not None]
    if known_costs:
        exact_cost = sum(known_costs)
    else:
        PRICING_MATRIX = {
            "anthropic/claude-sonnet-4.5": {"prompt": 3.00, "completion": 15.00},
            "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.70, "completion": 0.90},
            "google/qwen-2.5-coder-32b": {"prompt": 0.35, "completion": 0.35},
            "default": {"prompt": 0.70, "completion": 0.90}
        }

        pricing = PRICING_MATRIX.get(rag_model, PRICING_MATRIX["default"])

        prompt_cost = (total_prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (total_completion_tokens / 1_000_000) * pricing["completion"]
        exact_cost = prompt_cost + completion_cost
    
    q_file_label = "default"
    if questions_file:
        q_file_label = Path(questions_file).name
    elif hasattr(result, "questions_file") and result.questions_file:
        q_file_label = Path(result.questions_file).name
    else:
        q_file_label = f"{corpus}.json"

    chunks_to_llm = rerank_top_k if reranker_model else result.top_k

    row = {
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name":              run_name.strip() if run_name else "default",
        "rag_model":             rag_model,
        "evaluator_model":       evaluator_model,
        "embedder":              result.model,
        "corpus":                corpus,
        "questions_file":        q_file_label,
        "collection":            collection,
        "top_k":                 result.top_k,
        "n_questions":           result.n_questions,
        "reranker":              reranker_model or "none",
        "chunks_to_llm":          chunks_to_llm,
        "include_keywords":      include_keywords,
        "search_method":         search_method,
        "embed_backend":         embed_backend,
        "avg_search_latency_sec": round(avg_search, 2), 
        "avg_rerank_latency_sec": round(avg_rerank, 2), 
        "avg_gen_latency_sec":    round(avg_gen, 2),     
        "avg_total_latency_sec":  round(avg_total, 2),   
        "total_api_tokens":      total_prompt_tokens + total_completion_tokens,
        "total_rerank_tokens":    total_rerank_tokens,
        "total_cached_tokens":   total_cached_tokens,       
        "total_reasoning_tokens": total_reasoning_tokens,   
        "run_cost_usd":          round(exact_cost, 6),      
        "inject_metadata":       inject_metadata,
        "accuracy":              round(max(0, getattr(result, "mean_accuracy", -1.0)),  3),
        "faithfulness":          round(max(0, result.mean_faithfulness),       3),
        "answer_relevancy":      round(max(0, result.mean_answer_relevancy),   3),
        "mean_similarity":       round(result.mean_similarity,                 3),
        "coverage_ratio":        round(max(0, getattr(result, "mean_coverage_ratio", -1.0)), 3),
        "context_recall":        round(max(0, result.mean_context_recall),     3),
        "answer_correctness":    round(max(0, result.mean_answer_correctness), 3),
        "average":               _avg(result),
        "by_format":             getattr(result, "by_format", {}),
        "bl_accuracy":           round(max(0, getattr(baseline_result, "mean_accuracy", -1.0)), 3) if baseline_result else None,
        "bl_faithfulness":       round(max(0, baseline_result.mean_faithfulness),     3) if baseline_result else None,
        "bl_answer_relevancy":   round(max(0, baseline_result.mean_answer_relevancy), 3) if baseline_result else None,
        "bl_context_recall":     round(max(0, baseline_result.mean_context_recall),     3) if baseline_result else None,
        "bl_answer_correctness": round(max(0, baseline_result.mean_answer_correctness), 3) if baseline_result else None,
        "bl_average":            _avg(baseline_result) if baseline_result else None,
    }

    # budget keys only appear on rows from budgeted runs
    if context_token_budget is not None:
        row["context_token_budget"] = context_token_budget
        row["candidate_pool"] = candidate_pool
        row["budget_overflow"] = budget_overflow

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if leaderboard_path.exists():
        try:
            rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        except Exception as e:
            # Back up a corrupted leaderboard.json rather than overwriting its history.
            import shutil
            from datetime import datetime
            backup_path = leaderboard_path.with_suffix(
                f".corrupted-{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            shutil.copy2(leaderboard_path, backup_path)
            print(f"Could not parse existing leaderboard at {leaderboard_path} "
                  f"({type(e).__name__}: {e}). Backed up the unreadable file to "
                  f"{backup_path} and starting a fresh leaderboard rather than "
                  f"silently discarding its history.")
            rows = []

    rows.append(row)
    leaderboard_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Leaderboard updated -> {leaderboard_path}  (avg={_avg(result):.3f})")


def _run_info(args) -> None:
    import chromadb
    client      = chromadb.PersistentClient(path=args.db_path)
    collections = client.list_collections()
    print(f"\nDatabase: {args.db_path}")
    print(f"Collections ({len(collections)}):")
    for col in collections:
        c = client.get_collection(col.name)
        print(f"  {col.name}: {c.count()} chunks")


def _run_delete_collection(args) -> None:
    import chromadb
    from open_telco_rag.stores.fts import FTSStore
    
    client = chromadb.PersistentClient(path=args.db_path)
    
    print(f"\nAttempting to delete collection '{args.collection}' from '{args.db_path}'...")
    
    try:
        existing_collections = [c.name for c in client.list_collections()]
        if args.collection not in existing_collections:
            print(f"Error: Collection '{args.collection}' does not exist.")
            print(f"Available collections: {', '.join(existing_collections) if existing_collections else 'None'}")
            return

        client.delete_collection(name=args.collection)
        FTSStore(path=args.db_path, collection_name=args.collection).delete()
        print(f"Deleted collection '{args.collection}'.")
    except Exception as e:
        print(f"Failed to delete collection: {e}")


if __name__ == "__main__":
    main()