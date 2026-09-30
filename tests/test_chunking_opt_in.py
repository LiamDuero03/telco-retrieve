"""Token-based chunk sizing (target_tokens/max_tokens) is OPT-IN.

  (a) default / character-based chunking is unchanged and never touches the tokenizer
  (b) token-based chunking works for every strategy when explicitly requested
  (c) conflicting or invalid combinations raise a clear ValueError
  (d) CLI: token flags never inject the character defaults; old invocations unchanged
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict, fields
from types import SimpleNamespace

import pytest
import tiktoken

import open_telco_rag.core.chunking as chunking
from open_telco_rag.core.chunking import ChunkingStrategy, chunk_document
from open_telco_rag.types import Chunk

DOC = "\n\n".join(
    f"## Section {i}\n\n"
    + "\n\n".join(
        f"Paragraph {i}.{j}: the AMF processes N{j} requests and replies within timer T{i}{j}. "
        f"A second sentence about procedure {i * 10 + j}."
        for j in range(1, 6)
    )
    for i in range(1, 6)
)

CHAR_STRATEGIES = ["text_baseline", "hierarchical_markdown", "parent_child"]
ALL_STRATEGIES = CHAR_STRATEGIES + ["sliding_window_tokens", "lumber_chunker"]


class _FakeLLM:
    def __init__(self):
        msg = SimpleNamespace(content="Answer: ID 2")
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        self.completion = SimpleNamespace(create=lambda **kw: resp)


def _extra(strategy):
    return {"llm_client": _FakeLLM()} if strategy == "lumber_chunker" else {}


def _ntok(text):
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# ── (a) default mode ──────────────────────────────────────────────────────────

def test_chunk_dataclass_fields_unchanged():
    assert [f.name for f in fields(Chunk)] == ["text", "heading", "chunk_index", "source_type"]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_none_token_options_are_identical_to_omitting_them(strategy):
    plain = chunk_document(DOC, strategy, **_extra(strategy))
    with_none = chunk_document(DOC, strategy, target_tokens=None, max_tokens=None, **_extra(strategy))
    assert [asdict(c) for c in plain] == [asdict(c) for c in with_none]
    # Default-mode chunks carry no token info and serialize without new keys.
    assert all(set(asdict(c)) == {"text", "heading", "chunk_index", "source_type"} for c in plain)
    assert all(c.char_start == -1 and c.char_end == -1 and c.n_tokens == 0 for c in plain)


@pytest.mark.parametrize("strategy", CHAR_STRATEGIES)
def test_default_paths_never_initialize_the_tokenizer(strategy, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tiktoken.get_encoding called on a character-sized path")

    monkeypatch.setattr(tiktoken, "get_encoding", boom)
    monkeypatch.setattr(chunking, "_ENCODING_CACHE", {})
    assert chunk_document(DOC, strategy, chunk_size=800, overlap=100) if strategy == "text_baseline" \
        else chunk_document(DOC, strategy)


def test_importing_chunking_module_does_not_initialize_the_tokenizer():
    # fresh interpreter so an already-imported module can't mask it
    code = ";".join([
        "import tiktoken",
        "tiktoken.get_encoding = lambda *a, **k: 1 / 0",
        "import open_telco_rag.core.chunking",
    ])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_token_path_initializes_tokenizer_lazily_and_once(monkeypatch):
    calls = []
    real = tiktoken.get_encoding

    def counting(name):
        calls.append(name)
        return real(name)

    monkeypatch.setattr(tiktoken, "get_encoding", counting)
    monkeypatch.setattr(chunking, "_ENCODING_CACHE", {})
    chunk_document(DOC, "text_baseline", target_tokens=60)
    chunk_document(DOC, "text_baseline", target_tokens=60)
    assert calls == ["cl100k_base"]


# ── (b) token mode ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_token_mode_respects_max_tokens_and_exact_slices(strategy):
    chunks = chunk_document(DOC, strategy, target_tokens=40, max_tokens=60, **_extra(strategy))
    assert chunks
    for c in chunks:
        assert c.n_tokens == _ntok(c.text)
        assert c.n_tokens <= 60, (strategy, c.n_tokens)
        assert DOC[c.char_start:c.char_end] == c.text


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_max_tokens_alone_is_enough_to_opt_in(strategy):
    chunks = chunk_document(DOC, strategy, max_tokens=50, **_extra(strategy))
    assert chunks and all(c.n_tokens <= 50 and c.char_end > c.char_start for c in chunks)


def test_token_mode_differs_from_character_mode():
    char_mode = chunk_document(DOC, "text_baseline")
    token_mode = chunk_document(DOC, "text_baseline", target_tokens=40)
    assert len(token_mode) > len(char_mode)


# ── (c) conflicting / invalid combinations ────────────────────────────────────

@pytest.mark.parametrize("strategy,extra", [
    ("text_baseline", {"chunk_size": 1000}),
    ("text_baseline", {"overlap": 100}),
    ("hierarchical_markdown", {"chunk_size": 1000}),
    ("parent_child", {"parent_size": 2000}),
    ("parent_child", {"child_size": 400}),
    ("sliding_window_tokens", {"chunk_tokens": 256}),
    ("sliding_window_tokens", {"chunk_size": 256}),
    ("lumber_chunker", {"chunk_size": 1500}),
])
@pytest.mark.parametrize("token_opt", [{"target_tokens": 50}, {"max_tokens": 50}])
def test_mixing_token_and_character_sizes_raises(strategy, extra, token_opt):
    with pytest.raises(ValueError, match="cannot be combined"):
        chunk_document(DOC, strategy, **token_opt, **extra, **_extra(strategy))


@pytest.mark.parametrize("bad", [0, -5, 1.5, "10", True])
def test_invalid_token_values_raise(bad):
    with pytest.raises(ValueError, match="positive integer"):
        chunk_document(DOC, "text_baseline", target_tokens=bad)


def test_max_below_target_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_document(DOC, "text_baseline", target_tokens=100, max_tokens=50)


def test_sliding_window_overlap_tokens_is_allowed_with_target_tokens():
    chunks = chunk_document(DOC, "sliding_window_tokens", target_tokens=40, overlap_tokens=5)
    assert chunks and all(c.n_tokens <= 40 for c in chunks)


def test_legacy_sliding_window_arguments_still_work():
    assert chunk_document(DOC, "sliding_window_tokens", chunk_tokens=64, overlap_tokens=8)
    assert chunk_document(DOC, ChunkingStrategy.SLIDING_WINDOW_TOKENS, chunk_size=64, overlap=8)


# ── (d) CLI ───────────────────────────────────────────────────────────────────

def _cli_args(**over):
    base = dict(source_type="oran", use_marker=False, model="minilm", strategy="text_baseline",
                enrichment="none", data=None, db_path="db", no_resume=True, collection=None,
                chunk_size=None, overlap=None, target_tokens=None, max_tokens=None,
                overlap_tokens=None)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def cli_ingest(monkeypatch):
    import cli.main as cli_main
    import open_telco_rag.core.pipeline as pipeline

    captured = {}

    def fake_ingest(**kw):
        captured.update(kw)
        return SimpleNamespace(files_processed=0, chunks_stored=0, skipped_files=0, error_files=0)

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    monkeypatch.setattr(cli_main, "_get_source", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_embedder", lambda *a, **k: object())
    monkeypatch.setattr(cli_main, "_get_store", lambda *a, **k: SimpleNamespace(collection_name="c"))
    return cli_main, captured


def test_cli_old_invocation_still_gets_character_defaults(cli_ingest):
    cli_main, captured = cli_ingest
    cli_main._run_ingest(_cli_args())
    assert captured["chunk_size"] == 1024 and captured["overlap"] == 200
    assert "target_tokens" not in captured and "max_tokens" not in captured


def test_cli_explicit_character_sizes_still_pass_through(cli_ingest):
    cli_main, captured = cli_ingest
    cli_main._run_ingest(_cli_args(chunk_size=512, overlap=50))
    assert captured["chunk_size"] == 512 and captured["overlap"] == 50


def test_cli_token_flags_do_not_inject_character_defaults(cli_ingest):
    cli_main, captured = cli_ingest
    cli_main._run_ingest(_cli_args(target_tokens=256, max_tokens=300))
    assert captured["target_tokens"] == 256 and captured["max_tokens"] == 300
    assert "chunk_size" not in captured and "overlap" not in captured
    assert "chunk_tokens" not in captured


def test_cli_token_flags_with_explicit_character_size_exit_with_error(cli_ingest):
    cli_main, _ = cli_ingest
    with pytest.raises(SystemExit, match="cannot be combined"):
        cli_main._run_ingest(_cli_args(target_tokens=256, chunk_size=512))


def test_cli_overlap_tokens_requires_token_sizing(cli_ingest):
    cli_main, _ = cli_ingest
    with pytest.raises(SystemExit, match="requires"):
        cli_main._run_ingest(_cli_args(overlap_tokens=10))


# ── token-mode overlap must be smaller than the chunk size ────────────────────

@pytest.mark.parametrize("strategy", ["text_baseline", "sliding_window_tokens"])
@pytest.mark.parametrize("overlap", [50, 80, -1])
def test_token_overlap_not_smaller_than_the_chunk_size_is_an_error(strategy, overlap):
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_document(DOC, strategy, target_tokens=50, overlap_tokens=overlap)


@pytest.mark.parametrize("strategy", ["text_baseline", "sliding_window_tokens"])
def test_valid_token_overlap_still_works(strategy):
    chunks = chunk_document(DOC, strategy, target_tokens=60, overlap_tokens=10)
    assert chunks and all(c.n_tokens <= 60 for c in chunks)
    assert len(chunk_document(DOC, strategy, target_tokens=60, overlap_tokens=0)) <= len(chunks)
