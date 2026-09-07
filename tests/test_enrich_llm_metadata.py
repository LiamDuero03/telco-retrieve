"""Unit tests for enrich_llm_metadata / _enrich_chunks_async — the LLM-based
enrichment strategy.

httpx.AsyncClient is mocked throughout, no real network calls. Uses real
LocalLLMClient/OpenRouterLLMClient instances (not fakes) so a regression in
their .base_url/.api_key attributes is caught here too; exercises the
backend-agnostic URL/auth-header construction specifically.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_telco_rag.core.enrichment import enrich_chunks, enrich_llm_metadata
from open_telco_rag.generators.local_llm import LocalLLMClient, OpenRouterLLMClient
from open_telco_rag.types import Chunk


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, heading=None, chunk_index=idx, source_type="oran")


def _json_response(content: str, status_ok: bool = True):
    resp = MagicMock()
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = RuntimeError("HTTP error")
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _patch_async_client(post_side_effect):
    """Patches httpx.AsyncClient as an async context manager driven by post_side_effect.
    Returns (patcher, inner client mock) so tests can assert on call_args."""
    client_instance = MagicMock()
    client_instance.post = AsyncMock(side_effect=post_side_effect)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client_instance)
    ctx.__aexit__ = AsyncMock(return_value=False)

    return patch("httpx.AsyncClient", return_value=ctx), client_instance


# ── No / broken client -> graceful skip, never raises ──────────────────────

def test_no_llm_client_skips_and_returns_chunks_unchanged(capsys):
    chunks = [_chunk("Some technical content about the AMF.")]
    result = enrich_llm_metadata(chunks, llm_client=None)

    assert result[0].text == "Some technical content about the AMF."
    assert "No LLM client provided" in capsys.readouterr().out


def test_client_without_base_url_skips_and_returns_chunks_unchanged(capsys):
    chunks = [_chunk("Some technical content about the AMF.")]
    result = enrich_llm_metadata(chunks, llm_client=object())

    assert result[0].text == "Some technical content about the AMF."
    assert "no base_url attribute" in capsys.readouterr().out


# ── URL/header construction - the core vLLM-compatibility fix ──────────────

def test_local_client_builds_url_from_base_url_not_hardcoded_openrouter(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    client = LocalLLMClient(base_url="http://localhost:8000/v1", api_key=None)
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response('["Access and Mobility Function"]'))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client, model_name="qwen-test")

    url, kwargs = client_mock.post.call_args
    assert url[0] == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in kwargs["headers"]  # no api_key -> no header


def test_max_tokens_defaults_to_1500():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client)

    _, kwargs = client_mock.post.call_args
    assert kwargs["json"]["max_tokens"] == 1500


def test_timeout_defaults_to_180_and_is_overridable_via_kwarg():
    """Regression: a flat 60s httpx timeout under real concurrency was silently
    dropping chunks whose GPU time had already been spent ("failed for chunk N"
    with no truncation warning) -- must default higher and stay overridable."""
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Some technical content about the AMF.")]

    with patch("httpx.AsyncClient") as mock_client_cls:
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=AsyncMock(return_value=_json_response("[]"))
        ))
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = ctx

        enrich_llm_metadata(chunks, llm_client=client)
        assert mock_client_cls.call_args.kwargs["timeout"] == 180

        enrich_llm_metadata(chunks, llm_client=client, timeout=300)
        assert mock_client_cls.call_args.kwargs["timeout"] == 300


def test_frequency_penalty_defaults_to_0_4_and_is_overridable_via_kwarg():
    """Regression: some truncations' JSONDecodeError offsets ran 100k+ chars
    into a 3000-token budget -- a repetition loop, not genuinely long content.
    frequency_penalty must be sent by default to discourage that."""
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client)
    _, kwargs = client_mock.post.call_args
    assert kwargs["json"]["frequency_penalty"] == 0.4

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client, frequency_penalty=0.8)
    _, kwargs = client_mock.post.call_args
    assert kwargs["json"]["frequency_penalty"] == 0.8


def test_max_tokens_override_via_kwarg():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client, max_tokens=800)

    _, kwargs = client_mock.post.call_args
    assert kwargs["json"]["max_tokens"] == 800


def test_truncated_response_prints_specific_warning_and_leaves_chunk_unenriched(capsys):
    """finish_reason=="length" means truncation; must warn specifically, not report
    a generic parse failure, and leave the chunk safely unenriched."""
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    original = "Some technical content about the AMF."
    chunks = [_chunk(original)]

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{
            "message": {"content": '["AMF: Access and Mobility Fun'},  # cut off mid-string
            "finish_reason": "length",
        }]
    }
    patcher, _ = _patch_async_client(AsyncMock(return_value=resp))
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client, max_tokens=50)

    output = capsys.readouterr().out
    assert "was truncated at max_tokens=50" in output
    assert result[0].text == original  # left unenriched, not corrupted


def test_local_client_with_api_key_sends_bearer_header():
    client = LocalLLMClient(base_url="http://localhost:8000/v1", api_key="secret")
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client)

    _, kwargs = client_mock.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_openrouter_client_hits_openrouter_base_url(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterLLMClient(api_key="or-key")
    chunks = [_chunk("Some technical content about the AMF.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response("[]"))
    )
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client)

    url, kwargs = client_mock.post.call_args
    assert url[0] == "https://openrouter.ai/api/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer or-key"


# ── Content behavior ─────────────────────────────────────────────────────────

def test_successful_response_prepends_definitions_to_chunk_text():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    original = "The AMF handles registration."
    chunks = [_chunk(original)]

    patcher, _ = _patch_async_client(
        AsyncMock(return_value=_json_response(
            '["AMF: Access and Mobility Management Function"]'
        ))
    )
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    assert "Technical Context & Definitions:" in result[0].text
    assert "AMF: Access and Mobility Management Function" in result[0].text
    assert f"Original Content:\n{original}" in result[0].text


def test_dict_wrapped_response_unwraps_first_key():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Body text.")]

    patcher, _ = _patch_async_client(
        AsyncMock(return_value=_json_response('{"definitions": ["RIC: Radio Intelligent Controller"]}'))
    )
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    assert "RIC: Radio Intelligent Controller" in result[0].text


def test_empty_dict_response_is_treated_as_no_explanations_not_a_crash(capsys):
    """Regression: a model asked for "nothing notable" sometimes returns {}
    instead of the requested [] -- list(explanations.keys())[0] on an empty
    dict raised an unhandled IndexError ("list index out of range"), masking
    a legitimate empty result as a parse failure. Must leave the chunk
    unenriched, same as an empty list, with no error printed at all."""
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    original = "Body text with nothing notable in it."
    chunks = [_chunk(original)]

    patcher, _ = _patch_async_client(
        AsyncMock(return_value=_json_response("{}"))
    )
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    assert result[0].text == original
    assert "list index out of range" not in capsys.readouterr().out


def test_empty_chunk_text_is_skipped_without_a_request():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("   ")]

    patcher, client_mock = _patch_async_client(AsyncMock())
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    client_mock.post.assert_not_called()
    assert result[0].text == "   "


# ── Per-chunk fault isolation ────────────────────────────────────────────────

def test_one_chunk_failing_does_not_affect_others(capsys):
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("First chunk.", idx=0), _chunk("Second chunk.", idx=1)]

    async def post_side_effect(*args, **kwargs):
        payload = kwargs["json"]
        if "First" in payload["messages"][1]["content"]:
            raise RuntimeError("connection reset")
        return _json_response('["Second: a definition"]')

    patcher, _ = _patch_async_client(post_side_effect)
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    # Failed chunk: untouched, error logged, no exception propagated.
    assert result[0].text == "First chunk."
    assert "LLM Enrichment failed for chunk 0" in capsys.readouterr().out
    # Other chunk: still enriched normally.
    assert "Second: a definition" in result[1].text


def test_malformed_json_response_leaves_chunk_unchanged():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Body text.")]

    patcher, _ = _patch_async_client(
        AsyncMock(return_value=_json_response("not valid json"))
    )
    with patcher:
        result = enrich_llm_metadata(chunks, llm_client=client)

    assert result[0].text == "Body text."


# ── Concurrency cap ───────────────────────────────────────────────────────────

def test_concurrency_is_capped_by_max_concurrency_kwarg():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk(f"Chunk number {i}.", idx=i) for i in range(10)]

    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def post_side_effect(*args, **kwargs):
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _json_response("[]")

    patcher, _ = _patch_async_client(post_side_effect)
    with patcher:
        enrich_llm_metadata(chunks, llm_client=client, llm_concurrency=3)

    assert max_observed <= 3
    assert max_observed > 1  # confirms it actually ran concurrently, not serially


# ── enrich_chunks() dispatch reaches the same code path ─────────────────────

def test_enrich_chunks_dispatches_llm_metadata_by_string_name():
    client = LocalLLMClient(base_url="http://localhost:8000/v1")
    chunks = [_chunk("Body text.")]

    patcher, client_mock = _patch_async_client(
        AsyncMock(return_value=_json_response('["term: definition"]'))
    )
    with patcher:
        result = enrich_chunks(chunks, "llm_metadata", llm_client=client)

    client_mock.post.assert_called_once()
    assert "term: definition" in result[0].text
