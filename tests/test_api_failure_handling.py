"""API failure handling: transient errors are retried with backoff, permanent
ones fail fast, and a bad answer never silently misaligns embeddings."""
from __future__ import annotations

import time

import pytest
import requests

from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
from open_telco_rag.evaluation._claude import _post_with_retry


class _Resp:
    def __init__(self, status=200, payload=None, headers=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class _Session:
    """returns/raises the scripted outcomes in order, then repeats the last one"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, **kw):
        self.calls.append(kw)
        outcome = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ok(n, usage=None, offset=0):
    body = {"data": [{"index": i, "embedding": [float(offset + i)]} for i in range(n)]}
    if usage is not None:
        body["usage"] = usage
    return _Resp(200, body)


@pytest.fixture
def embedder(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    e = OpenRouterEmbedder(model_key="or-3-small", batch_size=4, max_workers=1)
    e.sleeps = sleeps
    return e


def _run(embedder, script, n=3):
    embedder.session = _Session(script)
    return embedder.embed([f"t{i}" for i in range(n)])


@pytest.mark.parametrize("bad", [_Resp(429), _Resp(500), _Resp(502), _Resp(503), _Resp(504)])
def test_transient_http_errors_are_retried_with_backoff(embedder, bad):
    out = _run(embedder, [bad, _ok(3)])
    assert out == [[0.0], [1.0], [2.0]] and embedder.sleeps == [2]


def test_dropped_connections_and_timeouts_are_retried(embedder):
    out = _run(embedder, [requests.ConnectionError("reset"), requests.Timeout("slow"), _ok(3)])
    assert len(out) == 3 and embedder.sleeps == [2, 4]


def test_provider_errors_in_a_200_body_are_retried_whatever_their_shape(embedder):
    out = _run(embedder, [_Resp(200, {"error": {"message": "overloaded"}}),
                          _Resp(200, {"error": "plain string error"}), _ok(3)])
    assert len(out) == 3 and embedder.sleeps == [2, 4]


def test_a_non_json_200_is_retried(embedder):
    assert len(_run(embedder, [_Resp(200, bad_json=True), _ok(3)])) == 3


@pytest.mark.parametrize("wrong_count", [0, 2, 5])
def test_a_wrong_number_of_embeddings_is_never_accepted(embedder, wrong_count):
    out = _run(embedder, [_ok(wrong_count), _ok(3)])
    assert len(out) == 3 and embedder.sleeps == [2]


def test_persistent_failure_raises_after_six_attempts_with_the_last_error(embedder):
    session = _Session([_Resp(503)])
    embedder.session = session
    with pytest.raises(Exception, match=r"after 6 attempts.*HTTP 503"):
        embedder.embed(["a", "b"])
    assert len(session.calls) == 6 and embedder.sleeps == [2, 4, 8, 16, 32, 64]


def test_client_errors_fail_immediately_without_retrying(embedder):
    session = _Session([_Resp(401)])
    embedder.session = session
    with pytest.raises(requests.HTTPError):
        embedder.embed(["a"])
    assert len(session.calls) == 1 and embedder.sleeps == []


def test_every_request_has_a_timeout(embedder):
    _run(embedder, [_ok(3)])
    assert embedder.timeout == 120.0
    assert all(c["timeout"] == 120.0 for c in embedder.session.calls)


def test_null_usage_cost_does_not_break_a_batch(embedder):
    _run(embedder, [_ok(3, usage={"prompt_tokens": 7, "cost": None})])
    assert embedder.total_prompt_tokens == 7 and embedder.total_cost_usd == 0.0


def test_batches_keep_their_order_when_split(embedder):
    embedder.session = _Session([_ok(4, offset=0), _ok(3, offset=4)])
    out = embedder.embed([f"t{i}" for i in range(7)])          # batch_size=4 -> 2 requests
    assert [v[0] for v in out] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ── the shared LLM retry helper ───────────────────────────────────────────────

def test_llm_retry_honours_retry_after_and_recovers(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    session = _Session([_Resp(429, headers={"retry-after": "3"}), _Resp(200, {"ok": 1})])
    resp = _post_with_retry(session, "http://x", headers={}, json_payload={}, timeout=5)
    assert resp.status_code == 200 and 3.0 <= sleeps[0] <= 4.0


def test_llm_retry_gives_up_with_the_final_error_and_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    session = _Session([_Resp(503)])
    with pytest.raises(requests.HTTPError):
        _post_with_retry(session, "http://x", headers={}, json_payload={}, timeout=5, max_retries=2)
    assert len(session.calls) == 3
    session = _Session([_Resp(400)])
    with pytest.raises(requests.HTTPError):
        _post_with_retry(session, "http://x", headers={}, json_payload={}, timeout=5)
    assert len(session.calls) == 1
