"""Check chunks against token limits, per tokenizer, and calibrate the cl100k ceiling. Read-only: nothing is written.

Inputs (one of them):
  --collection NAME [--db-path chroma_db]   stored chunk texts of a Chroma collection, read straight from its sqlite
                                            file in immutable read-only mode (never opened through chromadb)
  --data-dir DIR --source-type 3gpp         a staged directory, chunked in memory with the same load / clean /
                                            chunk_document calls as ingest (--strategy, --chunk-size, --overlap,
                                            --target-tokens, --max-tokens, --bound-tokens, --bound-margin as for `ingest`)

--tokenizers lists the verification tokenizers (an embedding model key such as bge-m3, a Hugging Face model name, a
tokenizer.json path or tiktoken:<encoding>; only tokenizer files are downloaded). cl100k is always measured as the
reference. Each tokenizer's limit is --limit NAME=N, else max_seq_length from configs/embedding_models.json, else a
built-in table of the models' documented limits, else --bound. Per tokenizer the report gives chunks over its limit and
over the bound (--bound minus --margin), the ratio of its token count to the cl100k count (p50, p99, p99.9, max), and the
worst chunks (spec, token counts, kind, first 60 characters). In staged mode it also gives the text retained after the
bound pass and after the min_length filter that ingest applies afterwards.

--calibrate finds, for each tokenizer and each limit in --calibrate-limits, the largest cl100k ceiling C for which no
chunk (and at most --calibrate-tolerance of the chunks) exceeds the limit. Staged mode re-runs the bound pass for each
candidate C, so the answer comes from the data; for a collection the chunks are fixed, so C is an estimate from the ratios.
C is the cl100k limit itself: pass --bound-tokens C+margin to `ingest` (margin 2 by default).

Usage:
    python scripts/check_chunk_bound.py --collection 3gpp_bge --tokenizers minilm bge bge-m3 --bound 512
    python scripts/check_chunk_bound.py --data-dir data/3gpp/3GPP-clean/Rel-18 --source-type 3gpp --strategy hierarchical_markdown \
        --target-tokens 400 --bound-tokens 512 --tokenizers minilm bge bge-m3 qwen3-embedding-0.6b embeddinggemma-300m --calibrate
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from open_telco_rag.core.chunk_bound import (
    DEFAULT_MARGIN,
    DEFAULT_TOKENIZER,
    enforce_bound,
    get_counter,
    parse_verify,
)

# documented max_seq_length of the models in configs/embedding_models.json, used when the registry has no value
KNOWN_LIMITS = {"minilm": 256, "mpnet": 384, "e5": 512, "bge": 512, "bge-m3": 8192, "qwen3-embedding-0.6b": 32768,
                "embeddinggemma-300m": 2048}
REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "embedding_models.json"
CL100K = "cl100k"
_QUERY = """select e.embedding_id, m.string_value from embeddings e
            join segments s on s.id = e.segment_id and s.scope = 'METADATA'
            join collections c on c.id = s.collection
            join embedding_metadata m on m.id = e.id and m.key = 'chroma:document'
            where c.name = ?"""


def squash_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def counter_for(name: str):
    return get_counter(DEFAULT_TOKENIZER if name in (CL100K, "cl100k_base") else name)


def model_limit(name: str, overrides: dict[str, int]) -> tuple[int | None, str]:
    """(limit, where it came from)"""
    if name in overrides:
        return overrides[name], "--limit"
    try:
        value = json.loads(REGISTRY.read_text(encoding="utf8")).get(name, {}).get("max_seq_length")
    except (OSError, ValueError):
        value = None
    if value:
        return int(value), "registry"
    if name in KNOWN_LIMITS:
        return KNOWN_LIMITS[name], "built-in"
    return None, "--bound"


def kind(text: str) -> str:
    """a rough label for what makes a chunk long in tokens"""
    lines = [line for line in text.split("\n") if line.strip()]
    if lines and sum(line.lstrip().startswith("|") for line in lines) >= 0.5 * len(lines):
        return "table"
    n = max(len(text), 1)
    if sum(text.count(c) for c in "$\\{}^_") / n >= 0.08:
        return "latex/math"
    if sum(len(m) for m in re.findall(r"[0-9A-Fa-f_:.\-]{12,}", text)) / n >= 0.3:
        return "hex/ids"
    if sum(ord(c) > 127 for c in text) / n >= 0.2:
        return "symbols"
    return "prose"


def read_collection(db_path: Path, name: str) -> list[tuple[str, str]]:
    """(chunk id, text) for a collection; the file is opened read-only and immutable, so it is never modified"""
    sqlite_file = (db_path / "chroma.sqlite3").resolve()
    if not sqlite_file.exists():
        raise SystemExit(f"error: {sqlite_file} not found")
    con = sqlite3.connect(f"file:{sqlite_file.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute(_QUERY, (name,)).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"error: collection {name!r} not found or empty in {sqlite_file}")
    return rows


def staged_documents(args) -> dict[str, str]:
    from cli.main import _get_source
    source = _get_source(args.source_type, use_marker=False)
    data_dir = Path(args.data_dir)
    docs = {}
    for ext in source.supported_extensions():
        for path in sorted(data_dir.rglob(f"*{ext}")):
            text = source.load(path)
            if text and text.strip():
                docs[str(path.relative_to(data_dir))] = source.clean(text)
    if not docs:
        raise SystemExit(f"error: no {source.supported_extensions()} files under {data_dir}")
    return docs


def base_chunks(docs: dict[str, str], args) -> dict[str, list]:
    """the strategy's chunks per document, before any bound pass"""
    from open_telco_rag.core.chunking import chunk_document
    kwargs = {"strategy": args.strategy, "source_type": args.source_type}
    for key in ("chunk_size", "overlap", "target_tokens", "max_tokens", "overlap_tokens"):
        if getattr(args, key) is not None:
            kwargs[key] = getattr(args, key)
    if args.target_tokens is None and args.max_tokens is None:
        kwargs.setdefault("chunk_size", 1024)
        kwargs.setdefault("overlap", 200)
    return {rel: chunk_document(text, **kwargs) for rel, text in docs.items()}


def bound_pass(chunks_by_doc: dict[str, list], ceiling: int | None, min_length: int, verify=None, outliers: str = "warn",
               margin: int = DEFAULT_MARGIN):
    """apply the cl100k bound pass (ceiling None = no pass; verify / outliers as for ingest) and the min_length filter:
    [(id, text)] plus retention figures"""
    after_bound, final = {}, []
    for rel, chunks in chunks_by_doc.items():
        out = enforce_bound(list(chunks), ceiling, None, verify, outliers, margin) if ceiling is not None else list(chunks)
        after_bound[rel] = sum(squash_len(c.text) for c in out)
        final.extend((f"{rel}::{i}", c.text) for i, c in enumerate(out) if len(c.text.strip()) >= min_length)
    return final, after_bound


def retention(chunks: list[tuple[str, str]], docs: dict[str, str]) -> float | None:
    """non-whitespace characters of the chunks of the files present in docs over those of the documents"""
    norm = {re.sub(r"[\\/]+", "/", rel): text for rel, text in docs.items()}
    by_file: dict[str, int] = {}
    for chunk_id, text in chunks:
        rel = re.sub(r"[\\/]+", "/", chunk_id.split("::")[0])
        if rel in norm:
            by_file[rel] = by_file.get(rel, 0) + squash_len(text)
    total = sum(squash_len(norm[rel]) for rel in by_file)
    return sum(by_file.values()) / total if total else None


def pct(values: list[float], q: float) -> float:
    return values[min(len(values) - 1, max(0, int(q * len(values) + 0.999999) - 1))]


def measure(chunks: list[tuple[str, str]], name: str, limit: int | None, source: str, bound_eff: int, margin: int, n_outliers: int,
            cl_counts: list[int]) -> dict:
    count = counter_for(name)
    counts = [count(text) for _, text in chunks]
    eff = None if limit is None else limit - margin
    ratios = sorted(c / max(1, k) for c, k in zip(counts, cl_counts))
    order = sorted(range(len(chunks)), key=lambda i: (counts[i], counts[i] / max(1, cl_counts[i])), reverse=True)[:n_outliers]
    sizes = sorted(counts)
    return {
        "tokenizer": name, "limit": limit, "limit_source": source, "chunks": len(counts), "mean": sum(counts) / len(counts),
        "p95": pct(sizes, 0.95), "max": sizes[-1],
        "over_limit": None if eff is None else sum(c > eff for c in counts),
        "over_bound": sum(c > bound_eff for c in counts),
        "ratio": {"p50": pct(ratios, 0.5), "p99": pct(ratios, 0.99), "p99.9": pct(ratios, 0.999), "max": ratios[-1]},
        "outliers": [{"spec": chunks[i][0].split("::")[0], "cl100k": cl_counts[i], "tokens": counts[i], "kind": kind(chunks[i][1]),
                      "text": " ".join(chunks[i][1][:60].split())} for i in order],
    }


def calibrate_staged(chunks_by_doc: dict[str, list], names: list[str], limits: list[int], margin: int, tolerance: float,
                     step: int, min_length: int) -> dict:
    """largest cl100k ceiling C per (tokenizer, limit) with 0 chunks over, and with at most `tolerance` over, by re-running
    the bound pass for each candidate C from limit - margin downwards"""
    counters = {n: counter_for(n) for n in names}
    memo: dict[tuple[str, str], int] = {}

    def tokens(name: str, text: str) -> int:
        key = (name, text)
        if key not in memo:
            memo[key] = counters[name](text)
        return memo[key]

    result: dict = {}
    for limit in limits:
        eff = limit - margin
        pending = {(n, "zero") for n in names} | {(n, "tol") for n in names} | {("all", "zero"), ("all", "tol")}
        found: dict = {}
        c = eff
        while pending and c >= 16:
            final, _ = bound_pass(chunks_by_doc, c, min_length)
            texts = [t for _, t in final]
            fractions = {n: sum(tokens(n, t) > eff for t in texts) / max(1, len(texts)) for n in names}
            checks = {n: fractions[n] for n in names}
            checks["all"] = max(fractions.values()) if fractions else 0.0
            for name, frac in checks.items():
                if (name, "zero") in pending and frac == 0:
                    found[(name, "zero")] = c
                    pending.discard((name, "zero"))
                if (name, "tol") in pending and frac <= tolerance:
                    found[(name, "tol")] = c
                    pending.discard((name, "tol"))
            c -= step
        result[limit] = {name: {"zero": found.get((name, "zero")), "tolerance": found.get((name, "tol"))} for name in names + ["all"]}
    return result


def calibrate_estimate(measured: list[dict], limits: list[int], margin: int, tolerance: float) -> dict:
    """for a collection: C estimated as (limit - margin) divided by the max / p99.9 ratio; the chunks are not re-split"""
    out = {}
    for limit in limits:
        out[limit] = {}
        for r in measured:
            if r["tokenizer"] == CL100K:
                continue
            q = r["ratio"]["p99.9"] if tolerance >= 0.001 else r["ratio"]["max"]
            out[limit][r["tokenizer"]] = {"zero": int((limit - margin) / max(r["ratio"]["max"], 1e-9)),
                                          "tolerance": int((limit - margin) / max(q, 1e-9))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection")
    p.add_argument("--db-path", default="chroma_db")
    p.add_argument("--data-dir", help="staged source directory (also the source files when --collection is used)")
    p.add_argument("--source-type", default="3gpp")
    p.add_argument("--tokenizers", nargs="+", default=[CL100K], help="verification tokenizers (cl100k is always measured too)")
    p.add_argument("--limit", action="append", default=[], metavar="NAME=N", help="override a tokenizer's limit (repeatable)")
    p.add_argument("--bound", type=int, default=512, help="the token limit chunks are held to (default 512)")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="tokens reserved for special tokens (default 2)")
    p.add_argument("--min-length", type=int, default=100, help="staged mode: drop chunks shorter than this many characters, as ingest does")
    p.add_argument("--outliers", type=int, default=20, help="how many of the largest chunks to list per tokenizer (default 20)")
    p.add_argument("--calibrate", action="store_true", help="find the cl100k ceiling C for each tokenizer and limit")
    p.add_argument("--calibrate-limits", nargs="+", type=int, default=[256, 512])
    p.add_argument("--calibrate-tolerance", type=float, default=0.001, help="share of chunks allowed over the limit (default 0.001)")
    p.add_argument("--calibrate-step", type=int, default=8, help="step between candidate ceilings (default 8)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--strategy", default="text_baseline")
    p.add_argument("--bound-verify", default=None, help="staged mode: as for `ingest --bound-verify` (name=limit,...)")
    p.add_argument("--bound-outliers", choices=["warn", "split"], default="warn", help="staged mode: as for `ingest`")
    for name in ("chunk-size", "overlap", "target-tokens", "max-tokens", "overlap-tokens", "bound-tokens", "bound-margin"):
        p.add_argument(f"--{name}", type=int, default=None)
    args = p.parse_args()
    if not args.collection and not args.data_dir:
        p.error("give --collection or --data-dir")
    if args.calibrate and not (args.data_dir or args.collection):
        p.error("--calibrate needs a source")
    overrides = {}
    for item in args.limit:
        name, _, value = item.rpartition("=")
        if not name or not value.isdigit():
            p.error(f"--limit expects NAME=N, got {item!r}")
        overrides[name] = int(value)

    docs = staged_documents(args) if args.data_dir else None
    stages: dict | None = None
    chunks_by_doc = None
    if args.collection:
        chunks = read_collection(Path(args.db_path), args.collection)
    else:
        chunks_by_doc = base_chunks(docs, args)
        ceiling = None if args.bound_tokens is None else args.bound_tokens - (DEFAULT_MARGIN if args.bound_margin is None else args.bound_margin)
        chunks, after_bound = bound_pass(chunks_by_doc, ceiling, args.min_length, parse_verify(args.bound_verify),
                                         args.bound_outliers, DEFAULT_MARGIN if args.bound_margin is None else args.bound_margin)
        total = sum(squash_len(t) for t in docs.values())
        _, before = bound_pass(chunks_by_doc, None, args.min_length)
        stages = {"strategy": sum(before.values()) / total, "after_bound_pass": sum(after_bound.values()) / total,
                  "after_min_length": sum(squash_len(t) for _, t in chunks) / total,
                  "dropped_by_min_length_chars": sum(after_bound.values()) - sum(squash_len(t) for _, t in chunks)}

    bound_eff = args.bound - args.margin
    cl_counter = counter_for(CL100K)
    cl_counts = [cl_counter(t) for _, t in chunks]
    measured = []
    for name in dict.fromkeys([CL100K, *args.tokenizers]):
        limit, source = model_limit(name, overrides) if name != CL100K else (args.bound, "--bound")
        try:
            measured.append(measure(chunks, name, limit if limit is not None else args.bound, source, bound_eff, args.margin,
                                    args.outliers, cl_counts))
        except Exception as e:  # noqa: BLE001 - one unavailable tokenizer must not hide the others
            measured.append({"tokenizer": name, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"})
    good = [r for r in measured if "error" not in r]
    calibration = None
    if args.calibrate:
        names = [n for n in args.tokenizers if n != CL100K and any(r["tokenizer"] == n for r in good)]
        calibration = (calibrate_staged(chunks_by_doc, names, args.calibrate_limits, args.margin, args.calibrate_tolerance,
                                        args.calibrate_step, args.min_length) if chunks_by_doc is not None
                       else calibrate_estimate(good, args.calibrate_limits, args.margin, args.calibrate_tolerance))
    kept = retention(chunks, docs) if docs else None

    report = {"chunks": len(chunks), "bound": args.bound, "margin": args.margin, "limit": bound_eff, "text_retained": kept,
              "stages": stages, "tokenizers": measured, "calibration": calibration,
              "calibration_is_estimate": bool(args.calibrate and chunks_by_doc is None)}
    if args.json:
        print(json.dumps(report, indent=2))
        return

    source = f"collection {args.collection!r}" if args.collection else f"staged {args.data_dir}"
    print(f"{source}: {len(chunks):,} chunks; bound {args.bound} minus margin {args.margin} = {bound_eff} tokens")
    print(f"{'tokenizer':26} {'limit':>7} {'over limit':>14} {'over bound':>14}   ratio vs cl100k: p50    p99  p99.9    max   mean    p95    max")
    for r in measured:
        if "error" in r:
            print(f"{r['tokenizer'][:26]:26} unavailable: {r['error']}")
            continue
        n = r["chunks"]
        over_limit = "-" if r["over_limit"] is None else f"{r['over_limit']} ({r['over_limit'] / n:.1%})"
        rt = r["ratio"]
        print(f"{r['tokenizer'][:26]:26} {r['limit']:>7} {over_limit:>14} {r['over_bound']:>6} ({r['over_bound'] / n:.1%})"
              f"   {rt['p50']:>22.2f} {rt['p99']:>6.2f} {rt['p99.9']:>6.2f} {rt['max']:>6.2f} {r['mean']:>6.1f} {r['p95']:>6} {r['max']:>6}")
    if stages:
        print(f"text retained (non-whitespace characters of the cleaned documents): strategy output {stages['strategy']:.1%}, "
              f"after the bound pass {stages['after_bound_pass']:.1%}, after the min_length filter {stages['after_min_length']:.1%} "
              f"({stages['dropped_by_min_length_chars']:,} characters dropped by min_length {args.min_length})")
    elif kept is not None:
        print(f"text retained: {kept:.1%} of the non-whitespace characters of the cleaned documents (stored chunks)")
    else:
        print("text retained: n/a (give --data-dir with the source files to compare)")
    for r in good:
        if r["outliers"] and args.outliers:
            print(f"\nlargest {len(r['outliers'])} chunks by {r['tokenizer']} tokens (spec, cl100k, tokens, kind, first 60 characters):")
            for o in r["outliers"]:
                print(f"  {o['spec'][:34]:34} {o['cl100k']:>6} {o['tokens']:>6}  {o['kind']:<10} {o['text']}")
    if calibration:
        note = " (estimate from the per-chunk ratios; the chunks were not re-split)" if report["calibration_is_estimate"] else ""
        print(f"\ncalibration{note}: largest cl100k ceiling C with no chunk over the limit / at most {args.calibrate_tolerance:.1%} over "
              f"(ingest: --bound-tokens C+{args.margin})")
        for limit, per in calibration.items():
            print(f"  limit {limit} (cut at {limit - args.margin}):")
            for name, c in per.items():
                zero = "none >= 16" if c["zero"] is None else c["zero"]
                tol = "none >= 16" if c["tolerance"] is None else c["tolerance"]
                print(f"    {name:26} C = {zero!s:>10} / {tol!s:>10}")


if __name__ == "__main__":
    main()
