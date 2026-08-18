"""Tests for _locked_add / _raise_first_write_failure, the fix for a
confirmed-in-production bug: ingest()'s background DB-writer thread pool
allowed concurrent writes to the same store when flushes overlapped, and
wait(db_futures) only waits for done state (not success), so a failed write
was silently discarded and the run still declared success. Confirmed via
audit_ingestion_grid.py: runs where _ingest_complete.json claimed hundreds
of thousands of chunks while the real ChromaDB collection held a few
hundred. Tested in isolation here; see test_ingest_multi_enrichment.py for
the full ingest() failure path (needs a real embedder + source).
"""
from __future__ import annotations

import threading
import time

import pytest

from open_telco_rag.core.pipeline import _locked_add, _raise_first_write_failure


# ── _locked_add ──────────────────────────────────────────────────────────────

def test_locked_add_calls_fn_with_given_args():
    calls = []
    lock = threading.Lock()

    _locked_add(lock, lambda a, b: calls.append((a, b)), "x", "y")

    assert calls == [("x", "y")]


def test_locked_add_serializes_concurrent_calls_to_the_same_lock():
    """Two calls sharing a lock must never overlap, even across threads."""
    lock = threading.Lock()
    in_flight = 0
    max_observed = 0
    guard = threading.Lock()  # protects in_flight/max_observed themselves

    def slow_write():
        nonlocal in_flight, max_observed
        with guard:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        time.sleep(0.05)
        with guard:
            in_flight -= 1

    threads = [threading.Thread(target=lambda: _locked_add(lock, slow_write)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_observed == 1, "two writes under the same lock overlapped"


def test_locked_add_does_not_serialize_different_locks():
    """Different locks must still run concurrently; fix must not force
    full serialization."""
    lock_a = threading.Lock()
    lock_b = threading.Lock()
    in_flight = 0
    max_observed = 0
    guard = threading.Lock()

    def slow_write():
        nonlocal in_flight, max_observed
        with guard:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        time.sleep(0.05)
        with guard:
            in_flight -= 1

    t1 = threading.Thread(target=lambda: _locked_add(lock_a, slow_write))
    t2 = threading.Thread(target=lambda: _locked_add(lock_b, slow_write))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert max_observed == 2, "writes under DIFFERENT locks should be able to overlap"


def test_locked_add_propagates_the_wrapped_function_exception():
    lock = threading.Lock()

    def boom():
        raise ValueError("simulated write failure")

    with pytest.raises(ValueError, match="simulated write failure"):
        _locked_add(lock, boom)


# ── _raise_first_write_failure ──────────────────────────────────────────────

class _FakeFuture:
    def __init__(self, exc: Exception | None):
        self._exc = exc

    def exception(self):
        return self._exc


def test_raise_first_write_failure_no_op_when_all_succeeded():
    futures = [_FakeFuture(None), _FakeFuture(None)]
    _raise_first_write_failure(futures)  # must not raise


def test_raise_first_write_failure_raises_on_a_failed_future():
    futures = [_FakeFuture(None), _FakeFuture(RuntimeError("db is locked")), _FakeFuture(None)]

    with pytest.raises(RuntimeError, match="Database write failed during ingest"):
        _raise_first_write_failure(futures)


def test_raise_first_write_failure_preserves_the_original_exception_as_cause():
    original = ValueError("original write error")
    futures = [_FakeFuture(original)]

    with pytest.raises(RuntimeError) as exc_info:
        _raise_first_write_failure(futures)

    assert exc_info.value.__cause__ is original
