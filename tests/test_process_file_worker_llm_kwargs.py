"""Unit tests for _process_file_worker's enrichment_llm_client kwarg
threading: it must reach enrich_chunks() (for llm_metadata) but must never
leak into chunk_document()'s kwargs — some chunking strategies (e.g.
chunk_sliding_window_tokens) have no **kwargs catch-all and would TypeError
on an unexpected key. Uses a real tiny local file — chunking/enrichment
here use the real (non-LLM) code paths; only enrich_llm_metadata's actual
LLM call would need a real client, and that's covered separately in
test_enrich_llm_metadata.py.
"""
from __future__ import annotations

from unittest.mock import patch

from open_telco_rag.core.pipeline import _process_file_worker


def _write_doc(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "23501-j00.md"
    f.write_text(
        "# Scope\n\nThis document specifies the AMF procedures for 5G "
        "registration and connection management. The AMF exposes the N1 "
        "interface to the UE for registration signalling.\n",
        encoding="utf-8",
    )
    return d, f


def test_enrichment_llm_client_reaches_enrich_chunks_not_chunk_document(tmp_path):
    data_dir, file_path = _write_doc(tmp_path)
    fake_client = object()

    with patch("open_telco_rag.core.pipeline.enrich_chunks") as mock_enrich:
        mock_enrich.side_effect = lambda chunks, **kw: chunks  # passthrough

        args = (
            str(file_path), str(data_dir), "3gpp", False,
            {
                "strategy": "text_baseline",
                "chunk_size": 200,
                "overlap": 20,
                "enrichment": "llm_metadata",
                "enrichment_model": "some-model",
                "enrichment_llm_client": fake_client,
            },
            10, "minilm",
        )
        result = _process_file_worker(args)

    assert result["status"] == "success", result.get("error")
    mock_enrich.assert_called_once()
    _, kwargs = mock_enrich.call_args
    assert kwargs["llm_client"] is fake_client
    assert kwargs["strategy"] == "llm_metadata"
    assert kwargs["model_name"] == "some-model"


def test_enrichment_llm_client_key_never_reaches_chunk_document_kwargs(tmp_path):
    """chunk_document()'s active_kwargs must not contain
    enrichment_llm_client — patch chunk_document itself and assert on the
    kwargs it actually received.
    """
    data_dir, file_path = _write_doc(tmp_path)
    fake_client = object()

    with patch("open_telco_rag.core.pipeline.chunk_document") as mock_chunk_doc, \
         patch("open_telco_rag.core.pipeline.enrich_chunks") as mock_enrich:
        mock_chunk_doc.return_value = []  # -> "no valid chunks", short-circuits cleanly
        mock_enrich.side_effect = lambda chunks, **kw: chunks

        args = (
            str(file_path), str(data_dir), "3gpp", False,
            {
                "strategy": "text_baseline",
                "chunk_size": 200,
                "overlap": 20,
                "enrichment": "none",
                "enrichment_llm_client": fake_client,
            },
            10, "minilm",
        )
        _process_file_worker(args)

    mock_chunk_doc.assert_called_once()
    _, kwargs = mock_chunk_doc.call_args
    assert "enrichment_llm_client" not in kwargs
    assert "llm_client" not in kwargs  # chunking strategy here doesn't need one either


def test_no_kwargs_catchall_strategy_does_not_crash_with_enrichment_llm_client_present(tmp_path):
    """chunk_sliding_window_tokens has no **kwargs catch-all -- if
    enrichment_llm_client ever leaked into chunk_document()'s kwargs this
    would TypeError. Runs for real (no mocking) to prove it end-to-end.
    """
    data_dir, file_path = _write_doc(tmp_path)
    fake_client = object()

    args = (
        str(file_path), str(data_dir), "3gpp", False,
        {
            "strategy": "sliding_window_tokens",
            "chunk_tokens": 50,
            "overlap_tokens": 5,
            "enrichment": "none",
            "enrichment_llm_client": fake_client,  # present even though enrichment is "none"
        },
        5, "minilm",
    )
    result = _process_file_worker(args)

    assert result["status"] == "success", result.get("error")
