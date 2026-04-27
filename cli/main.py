"""Unified CLI for Open Telco RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
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
    p_ingest.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in chars")
    p_ingest.add_argument("--overlap",    type=int, default=200,  help="Chunk overlap in chars")
    p_ingest.add_argument("--no-resume",  action="store_true",    help="Disable resume")
    p_ingest.add_argument("--use-marker", action="store_true", default=False,
                          help="Use Marker API for PDF conversion (better table/structure quality). "
                               "Requires DATALAB_API_KEY in .env. "
                               "Recommended for ETSI and TM Forum corpora.")

    # ── search ────────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Search the vector database")
    p_search.add_argument("query",                                  help="Search query")
    p_search.add_argument("--source-type", default="oran",         help="Source type")
    p_search.add_argument("--model",       default="minilm",       help="Embedding model key")
    p_search.add_argument("--db-path",     default="chroma_db",    help="ChromaDB path")
    p_search.add_argument("--top-k",       type=int, default=5,    help="Number of results")
    p_search.add_argument("--rerank",      action="store_true",    help="Enable cross-encoder reranking")
    p_search.add_argument("--rerank-top-k",type=int, default=5,    help="Chunks to keep after reranking")

    # ── ask ───────────────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a question (search + generate)")
    p_ask.add_argument("question",                                   help="Your question")
    p_ask.add_argument("--source-type", default="oran",             help="Source type")
    p_ask.add_argument("--model",       default="minilm",           help="Embedding model key")
    p_ask.add_argument("--db-path",     default="chroma_db",        help="ChromaDB path")
    p_ask.add_argument("--top-k",       type=int, default=8,        help="Chunks to retrieve")
    p_ask.add_argument("--rerank",      action="store_true",        help="Enable cross-encoder reranking")
    p_ask.add_argument("--rerank-top-k",type=int, default=5,        help="Chunks to keep after reranking")

    # ── evaluate (phase 2 — not yet released) ────────────────────────────────
    # Uncomment the block below when open_telco_rag/evaluation/ is published
    #
    # p_eval = sub.add_parser("evaluate", help="Run evaluation suite on a corpus")
    # p_eval.add_argument("corpus",
    #                     choices=["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi", "all"],
    #                     help="Corpus to evaluate, or 'all' to run all corpora sequentially")
    # p_eval.add_argument("--model",     default="minilm",    help="Embedding model key")
    # p_eval.add_argument("--db-path",   default="chroma_db", help="ChromaDB path")
    # p_eval.add_argument("--top-k",     type=int, default=5, help="Chunks to retrieve per question")
    # p_eval.add_argument("--threshold", type=float, default=0.5,
    #                     help="Similarity threshold for coverage_ratio metric (default: 0.5)")
    # p_eval.add_argument("--output",    default="results",
    #                     help="Directory to save result JSON files (default: results/)")
    # p_eval.add_argument("--delay",     type=float, default=1.0,
    #                     help="Seconds between questions — rate limits API calls (default: 1.0)")
    # p_eval.add_argument("--no-save",   action="store_true",
    #                     help="Print summary only, do not save JSON results")
    # p_eval.add_argument("--generator", choices=["claude", "openrouter"], default="claude",
    #                     help="LLM provider for evaluation metrics (default: claude)")
    # p_eval.add_argument("--generator-model", default=None,
    #                     help=("Model for evaluation metrics. "
    #                           "Claude default: claude-sonnet-4-20250514. "
    #                           "OpenRouter default: meta-llama/llama-3.3-70b-instruct."))
    # p_eval.add_argument("--debug", action="store_true",
    #                     help="Log raw LLM responses for every metric call")

    # ── info ──────────────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show database info")
    p_info.add_argument("--db-path",    default="chroma_db", help="ChromaDB path")
    p_info.add_argument("--collection", default=None,        help="Collection name")

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
    elif args.command == "info":
        _run_info(args)
    # ── evaluate (phase 2 — not yet released) ────────────────────────────────
    # elif args.command == "evaluate":
    #     _run_evaluate(args)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_source(source_type: str, use_marker: bool = False):
    if source_type == "oran":
        from open_telco_rag.sources.oran import ORANSource
        return ORANSource(use_marker=use_marker)
    elif source_type == "3gpp":
        from open_telco_rag.sources.tgpp import TPPSource
        return TPPSource()                          # already Markdown — no Marker needed
    elif source_type == "camara":
        from open_telco_rag.sources.camara import CAMARASource
        return CAMARASource()                       # already YAML/MD — no Marker needed
    elif source_type == "tmforum":
        from open_telco_rag.sources.tmforum import TMForumSource
        return TMForumSource(use_marker=use_marker)
    elif source_type == "itu":
        from open_telco_rag.sources.itu import ITUSource
        return ITUSource(use_marker=use_marker)
    elif source_type == "etsi":
        from open_telco_rag.sources.etsi import ETSISource
        return ETSISource(use_marker=use_marker)
    elif source_type == "gsma":
        from open_telco_rag.sources.gsma import GSMASource
        return GSMASource(use_marker=use_marker)
    raise NotImplementedError(f"Source '{source_type}' not yet implemented.")


def _get_embedder(model_key: str):
    from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_key=model_key)


def _get_store(source_type: str, model_key: str, db_path: str, collection: str = None):
    from open_telco_rag.stores.chroma import ChromaStore
    col = collection or f"{source_type}_{model_key}"
    return ChromaStore(path=db_path, collection_name=col)


# ── Command runners ───────────────────────────────────────────────────────────

def _run_ingest(args) -> None:
    from open_telco_rag.core.pipeline import ingest

    use_marker = getattr(args, "use_marker", False)
    source     = _get_source(args.source_type, use_marker=use_marker)
    embedder   = _get_embedder(args.model)
    store      = _get_store(args.source_type, args.model, args.db_path)
    data_dir   = args.data or f"data/{args.source_type}"

    print(f"\nIngesting {args.source_type.upper()} documents")
    print(f"  Data dir:   {data_dir}")
    print(f"  Model:      {args.model}")
    print(f"  Collection: {args.source_type}_{args.model}")
    if use_marker:
        print(f"  PDF loader: Marker API (high-quality)")
    else:
        print(f"  PDF loader: pypdf (default)")

    stats = ingest(
        source     = source,
        store      = store,
        embedder   = embedder,
        data_dir   = data_dir,
        chunk_size = args.chunk_size,
        overlap    = args.overlap,
        resume     = not args.no_resume,
    )

    print(f"\nFiles processed: {stats.files_processed}")
    print(f"Chunks stored:   {stats.chunks_stored}")
    print(f"Skipped:         {stats.skipped_files}")
    print(f"Errors:          {stats.error_files}")


def _run_search(args) -> None:
    import math
    from open_telco_rag.core.pipeline import search

    embedder = _get_embedder(args.model)
    store    = _get_store(args.source_type, args.model, args.db_path)

    reranker = None
    if args.rerank:
        from open_telco_rag.embedders.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        print(f"Reranking enabled — retrieving {args.top_k * 3} candidates → top {args.rerank_top_k}")

    results = search(
        args.query,
        store        = store,
        embedder     = embedder,
        top_k        = args.top_k,
        reranker     = reranker,
        rerank_top_k = args.rerank_top_k,
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

    embedder = _get_embedder(args.model)
    store    = _get_store(args.source_type, args.model, args.db_path)

    reranker = None
    if args.rerank:
        from open_telco_rag.embedders.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        print(f"Reranking enabled — retrieving {args.top_k * 3} candidates → top {args.rerank_top_k}")

    results = search(
        args.question,
        store        = store,
        embedder     = embedder,
        top_k        = args.top_k,
        reranker     = reranker,
        rerank_top_k = args.rerank_top_k,
    )
    answer = generate(args.question, results, source_type=args.source_type)

    print(f"\nAnswer:\n{answer['answer']}")
    print(f"\nSources:     {answer['sources']}")
    print(f"Chunks used: {answer['chunks_used']}")


# ── evaluate (phase 2 — not yet released) ────────────────────────────────────
# Uncomment when open_telco_rag/evaluation/ is published
#
# def _run_evaluate(args) -> None:
#     from open_telco_rag.evaluation.evaluate import evaluate_corpus
#     from open_telco_rag.evaluation.questions import SUPPORTED_CORPORA
#     from open_telco_rag.evaluation._claude import make_llm_call
#
#     embedder   = _get_embedder(args.model)
#     output_dir = None if args.no_save else Path(args.output)
#     llm_call   = make_llm_call(args.generator, args.generator_model)
#
#     corpora = SUPPORTED_CORPORA if args.corpus == "all" else [args.corpus]
#
#     for corpus in corpora:
#         store = _get_store(corpus, args.model, args.db_path)
#         count = store.count()
#         if count == 0:
#             print(f"\n[SKIP] {corpus.upper()} — collection '{corpus}_{args.model}' "
#                   f"is empty. Run: open-telco-rag ingest {corpus}")
#             continue
#         print(f"\n[INFO] {corpus.upper()} — {count:,} chunks in '{corpus}_{args.model}'")
#         evaluate_corpus(
#             corpus        = corpus,
#             store         = store,
#             embedder      = embedder,
#             model         = args.model,
#             top_k         = args.top_k,
#             sim_threshold = args.threshold,
#             delay         = args.delay,
#             output_dir    = output_dir,
#             llm_call      = llm_call,
#             debug         = args.debug,
#         )


def _run_info(args) -> None:
    import chromadb
    client      = chromadb.PersistentClient(path=args.db_path)
    collections = client.list_collections()
    print(f"\nDatabase: {args.db_path}")
    print(f"Collections ({len(collections)}):")
    for col in collections:
        c = client.get_collection(col.name)
        print(f"  {col.name}: {c.count()} chunks")


if __name__ == "__main__":
    main()