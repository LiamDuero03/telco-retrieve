"""dense / bm25 / hybrid retrieval, threshold, reranking, parent resolution and
budgeting against a real ChromaDB + FTS5 index built by ingest(), using a small
deterministic bag-of-words embedder (no model download)."""
from __future__ import annotations

import hashlib
import math
import re

import pytest
import tiktoken

from open_telco_rag.core.pipeline import ingest, search
from open_telco_rag.core.retrieval import execute_retrieval, reciprocal_rank_fusion
from open_telco_rag.sources.tgpp import TPPSource
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.types import SearchResult

DOCS = {
    "23501-j00.md": "# System\n\nThe AMF performs registration and mobility management for the UE. "
                    "The SMF manages PDU sessions and the UPF forwards user plane traffic.\n",
    "33501-j00.md": "# Security\n\nAuthentication uses 5G-AKA. The SEAF holds the anchor key and "
                    "integrity protection of signalling is mandatory between the UE and the gNB.\n",
    "38300-j00.md": "# NR\n\nHandover in NR is network controlled. The source gNB sends a handover "
                    "request to the target gNB over the Xn interface during handover preparation.\n",
    "38401-j00.md": "# RAN\n\nThe gNB consists of a gNB-CU and gNB-DUs connected over the F1 interface. "
                    "The E1 interface connects the CU-CP and the CU-UP.\n",
}
DIM = 64


class BagOfWordsEmbedder:
    model_key = "bow"
    dimensions = DIM

    def embed(self, texts, is_query=False):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retrieval")
    data = tmp / "data"
    data.mkdir()
    for name, text in DOCS.items():
        (data / name).write_text(text, encoding="utf-8")
    emb = BagOfWordsEmbedder()
    store = ChromaStore(path=tmp / "db", collection_name="t_3gpp_bow", metadata={"hnsw:space": "cosine"})
    stats = ingest(source=TPPSource(), store=store, embedder=emb, data_dir=str(data),
                   strategy="text_baseline", chunk_size=400, overlap=0, min_length=10,
                   resume=False, log_dir=str(tmp / "logs"))
    assert stats.error_files == 0 and stats.chunks_stored == store.count() >= 4
    return store, emb


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_each_search_method_finds_the_handover_document(index, method):
    store, emb = index
    results = execute_retrieval("handover request from the source gNB to the target gNB over Xn",
                                store, emb, top_k=2, search_method=method)
    assert 0 < len(results) <= 2
    assert "38300" in results[0].doc_id


def test_bm25_finds_an_exact_keyword_that_dense_may_miss(index):
    store, emb = index
    results = execute_retrieval("SEAF", store, emb, top_k=1, search_method="bm25")
    assert results and "33501" in results[0].doc_id


def test_search_method_is_case_insensitive_and_defaults_to_dense(index):
    store, emb = index
    a = execute_retrieval("PDU session", store, emb, top_k=2)
    b = execute_retrieval("PDU session", store, emb, top_k=2, search_method="DENSE")
    assert [r.doc_id for r in a] == [r.doc_id for r in b]


def test_threshold_only_filters_dense_results(index):
    store, emb = index
    loose = execute_retrieval("F1 interface CU DU", store, emb, top_k=4, threshold=0.05)
    tight = execute_retrieval("F1 interface CU DU", store, emb, top_k=4, threshold=0.95)
    assert len(tight) <= len(loose)


def test_hybrid_reranking_and_top_k(index):
    store, emb = index

    class ReverseReranker:
        def rerank(self, query, candidates, top_k=5):
            return list(reversed(candidates))[:top_k]

    results = execute_retrieval("gNB interface", store, emb, top_k=3, search_method="hybrid",
                                reranker=ReverseReranker(), rerank_top_k=2)
    assert len(results) == 2


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("overflow", ["stop", "skip", "truncate"])
def test_budgeted_retrieval_never_exceeds_the_budget(index, method, overflow):
    store, emb = index
    enc = tiktoken.get_encoding("cl100k_base")
    budget = 60
    results = execute_retrieval("gNB interface handover", store, emb, search_method=method,
                                context_token_budget=budget, candidate_pool=8, budget_overflow=overflow)
    assert sum(len(enc.encode(r.text)) for r in results) <= budget


def test_reciprocal_rank_fusion_rewards_agreement():
    def sr(i):
        return SearchResult(text=str(i), distance=0.0, metadata={}, doc_id=f"d{i}")

    dense = [sr(1), sr(2), sr(3)]
    bm25 = [sr(3), sr(2), sr(4)]
    fused = [r.doc_id for r in reciprocal_rank_fusion(dense, bm25)]
    assert fused[0] in ("d2", "d3")            # ranked by both lists
    assert set(fused) == {"d1", "d2", "d3", "d4"}
    assert reciprocal_rank_fusion([], []) == []


def test_pipeline_search_passes_the_method_through(index):
    store, emb = index
    assert search("SEAF", store=store, embedder=emb, top_k=1, search_method="bm25")[0].doc_id.startswith("33501")
