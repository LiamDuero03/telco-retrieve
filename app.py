"""Streamlit web UI for TelcoLens."""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TelcoLens",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Prevent sidebar from being permanently hidden by showing a reopen hint
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] { min-width: 260px !important; }
</style>
""", unsafe_allow_html=True)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 1rem; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .stTabs [data-baseweb='tab-panel'] { padding-top: 1rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #888; }
    .source-chip {
        display: inline-block; font-size: 0.72rem; padding: 2px 10px;
        border-radius: 99px; border: 1px solid #ddd; color: #555; margin: 2px 3px 2px 0;
    }
    .model-tag {
        display: inline-block; font-size: 0.72rem; padding: 2px 9px;
        border-radius: 99px; background: #fff8e1; color: #8a6200;
        border: 1px solid #f0d080; margin-left: 6px;
    }
    .answer-box {
        background: var(--secondary-background-color, #f8f8f8); 
        color: var(--text-color, #31333F); /* <-- Uses Streamlit's native text color */
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
        border-radius: 10px; 
        padding: 1rem 1.2rem; 
        margin-top: 0.5rem;
    }
    .result-card {
        background: #ffffff; border: 1px solid #e8e8e8;
        border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    }
    .result-source { font-size: 0.72rem; font-weight: 600; color: #1a6bbf; }
    .result-heading { font-size: 0.88rem; font-weight: 500; margin: 4px 0 2px; }
    .result-snippet { font-size: 0.82rem; color: #555; line-height: 1.55; }
    .sim-score { float: right; font-size: 0.72rem; color: #999; }
    .log-box {
        background: #1e1e1e; color: #d4d4d4; border-radius: 8px;
        padding: 0.75rem 1rem; font-family: monospace;
        font-size: 0.8rem; line-height: 1.7; white-space: pre-wrap;
    }
    .col-row {
        display: flex; justify-content: space-between; padding: 8px 12px;
        background: var(--secondary-background-color, #f5f5f5); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px; margin-bottom: 5px; font-size: 0.85rem;
    }
    .col-name { 
        font-weight: 500;
        color: var(--text-color, #31333F); 
    }
    .col-count { 
        color: var(--text-color, #31333F); 
        opacity: 0.65; 
    }
    .eval-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-radius: 8px; margin-bottom: 4px; font-size: 0.85rem;
    }
    .eval-row-header { background: #f0f0f0; font-weight: 600; }
    .eval-row-data { background: #fafafa; border: 1px solid #eee; }
    .eval-metric { flex: 2; }
    .eval-score { flex: 1; text-align: right; font-family: monospace; }
    .eval-type { flex: 1; text-align: right; font-size: 0.72rem; color: #999; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _get_source(source_type: str, use_marker: bool = False):
    if source_type == "oran":
        from open_telco_rag.sources.oran import ORANSource
        return ORANSource(use_marker=use_marker)
    elif source_type == "3gpp":
        from open_telco_rag.sources.tgpp import TPPSource
        return TPPSource()
    elif source_type == "camara":
        from open_telco_rag.sources.camara import CAMARASource
        return CAMARASource()
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
    raise NotImplementedError(f"Source '{source_type}' not implemented.")


@st.cache_resource(show_spinner=False)
def _get_embedder(model_key: str):
    from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_key=model_key)

@st.cache_resource(show_spinner=False)
def _get_reranker(model_key: str):
    from open_telco_rag.embedders.reranker import load_reranker
    return load_reranker(model_key)


@st.cache_data(show_spinner=False)
def _load_embedder_keys() -> dict:
    import json
    from pathlib import Path
    path = Path("configs/embedding_models.json")
    if not path.exists():
        return {"minilm": "minilm (default)"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: f"{k} — {v.get('description', v['model_name'])}" for k, v in data.items()}

@st.cache_data(show_spinner=False)
def _load_llm_keys() -> dict:
    import json
    from pathlib import Path
    path = Path("configs/llm_models.json")
    
    if not path.exists():
        return {"Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5"}
        
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["model_name"] for k, v in data.items()}

    
@st.cache_data(show_spinner=False)
def _load_chunking_strategies() -> dict:
    import json
    from pathlib import Path
    path = Path("configs/chunking_strategies.json")
    if not path.exists():
        return {"text_baseline": {"display_name": "Text Baseline", "help_text": "", "unit": "characters", "default_size": 1024, "default_overlap": 200}}
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def _load_enrichment_strategies() -> dict:
    import json
    from pathlib import Path
    path = Path("configs/enrichment_strategies.json")
    if not path.exists():
        return {"none": {"display_name": "No Enrichment", "help_text": "", "requires_llm": False}}
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_resource(show_spinner=False)
def _get_embedder(model_key: str, backend: str = "local"):
    if backend == "openrouter":
        from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
        return OpenRouterEmbedder(model_key=model_key)
    else:
        from open_telco_rag.embedders.sentense_transformers import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_key=model_key)


def _get_store(source_type: str, model_key: str, db_path: str,
               collection: str | None = None, distance_metric: str = "l2"):
    from open_telco_rag.stores.chroma import ChromaStore
    col = collection.strip() if collection and collection.strip() else f"{source_type}_{model_key}"
    
    
    return ChromaStore(
        path=db_path, 
        collection_name=col, 
        metadata={"hnsw:space": distance_metric.lower()}
    )


def _similarity(distance: float) -> float:
    import math
    return round(math.exp(-distance), 3)


def _build_generator(llm_model: str):
    from open_telco_rag.generators.openrouter import OpenRouterGenerator
    gen = OpenRouterGenerator(model=llm_model)
    if not hasattr(gen, "site_url"):  gen.site_url  = ""
    if not hasattr(gen, "site_name"): gen.site_name = "TelcoLens"
    return gen


def _render_chunk_cards(results):
    seen = set()
    displayed = 0
    for r in results:
        key = f"{r.metadata.get('source','')}::{r.metadata.get('heading','')}"
        if key in seen:
            continue
        seen.add(key)
        displayed += 1
        source  = r.metadata.get("source", "—")
        heading = r.metadata.get("heading", "")
        wg      = r.metadata.get("work_group", "")
        sim     = _similarity(r.distance)
        snippet = r.text[:300].replace("\n", " ")
        extra   = f" · {wg}" if wg else ""
        st.markdown(
            f"<div class='result-card'>"
            f"<span class='result-source'>{source}{extra}</span>"
            f"<span class='sim-score'>sim {sim}</span>"
            f"{'<div class=\"result-heading\">' + heading + '</div>' if heading else ''}"
            f"<div class='result-snippet'>{snippet}…</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    return displayed


# ── Sidebar ───────────────────────────────────────────────────────────────────

_load_env()

st.sidebar.markdown("## 🔭 TelcoLens")
st.sidebar.caption("v0.1.0")
st.sidebar.markdown("---")

# Pipeline selector — styled nav menu
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = "rag"

is_rag = st.session_state["pipeline"] == "rag"

if st.sidebar.button("🔍  RAG pipeline",
    key="btn_rag", width="stretch",
    type="primary" if is_rag else "secondary"):
    st.session_state["pipeline"] = "rag"
    st.rerun()

st.sidebar.caption("Ask · Search · Ingest · Evaluate · Leaderboard")

if st.sidebar.button("🧪  QnA generation",
    key="btn_qna", width="stretch",
    type="primary" if not is_rag else "secondary"):
    st.session_state["pipeline"] = "qna"
    st.rerun()

st.sidebar.caption("Generate · Review")

st.sidebar.markdown("---")

# ── RAG sidebar controls ──────────────────────────────────────────────────────
if is_rag:
    SOURCE_TYPES = ["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi"]
    source_type = st.sidebar.selectbox("Corpus", SOURCE_TYPES)

    _embedder_options = _load_embedder_keys()
    model_key = st.sidebar.selectbox(
        "Embedder",
        options=list(_embedder_options.keys()),
        format_func=lambda k: _embedder_options[k],
    )

    embed_backend = st.sidebar.radio(
        "Embedding Backend",
        options=["local", "openrouter"],
        horizontal=True,
        help="Use 'local' to run embeddings on your local CPU/GPU, or 'openrouter' to offload execution to OpenRouter's cloud GPUs.",
    )

    LLM_OPTIONS = _load_llm_keys()
    llm_label = st.sidebar.selectbox("LLM", list(LLM_OPTIONS.keys()))
    llm_model  = LLM_OPTIONS[llm_label]

    st.sidebar.markdown("---")
    DB_PATHS = ["chroma_db", "chroma_db_r19_marker"]
    db_path = st.sidebar.selectbox("DB path", DB_PATHS)

    collection_override = st.sidebar.text_input(
        "Collection override",
        placeholder=f"{source_type}_{model_key}",
        help="Leave blank for default (corpus_embedder). "
             "Use for custom names e.g. 3gpp_r19_marker_bge",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📚 Cookbook**")
    st.sidebar.markdown(
        "[TelcoLens Cookbook](https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf)",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("**▶ Try it yourself**")
    NOTEBOOKS = [
        ("O-RAN",    "01_oran_quickstart.ipynb"),
        ("3GPP",     "02_3gpp_quickstart.ipynb"),
        ("CAMARA",   "03_camara_quickstart.ipynb"),
        ("TM Forum", "04_tmforum_quickstart.ipynb"),
        ("GSMA",     "05_gsma_quickstart.ipynb"),
        ("ITU-T",    "06_itu_quickstart.ipynb"),
        ("ETSI",     "07_etsi_quickstart.ipynb"),
    ]
    NB_BASE = "https://github.com/Znbne/telcolens/blob/main/notebooks"
    for name, fname in NOTEBOOKS:
        st.sidebar.markdown(f"[{name}]({NB_BASE}/{fname})", unsafe_allow_html=True)

# ── QnA Generation sidebar controls ──────────────────────────────────────────
else:
    SOURCE_TYPES = ["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi"]
    source_type = st.sidebar.selectbox("Corpus", SOURCE_TYPES, key="qna_corpus")

    qna_source = st.sidebar.selectbox("Chunk source", ["chroma", "jsonl"], key="qna_source")

    DB_PATHS = ["chroma_db", "chroma_db_r19_marker"]
    db_path = st.sidebar.selectbox("DB path", DB_PATHS, key="qna_db_path")

    if qna_source == "chroma":
        _embedder_options = _load_embedder_keys()
        model_key = st.sidebar.selectbox(
            "Embedder (collection)",
            options=list(_embedder_options.keys()),
            format_func=lambda k: _embedder_options[k],
            key="qna_embedder",
        )
        qna_collection = st.sidebar.text_input(
            "Collection override", key="qna_collection",
            placeholder=f"{source_type}_minilm",
        )
        qna_jsonl_path = ""
    else:
        qna_jsonl_path = st.sidebar.text_input(
            "JSONL path", key="qna_jsonl_path",
            placeholder=f"data/{source_type}.jsonl",
        )
        model_key = "minilm"
        qna_collection = ""

    st.sidebar.markdown("---")
    LLM_OPTIONS = _load_llm_keys()
    
    qna_gen_label = st.sidebar.selectbox(
        "Generation model", 
        options=list(LLM_OPTIONS.keys()),
        key="qna_gen_label",
    )
    qna_gen_model = LLM_OPTIONS[qna_gen_label]
    
    qna_val_label = st.sidebar.selectbox(
        "Validation model", 
        options=list(LLM_OPTIONS.keys()),
        key="qna_val_label",
        help="Used by validation judges — can differ from generation model.",
    )
    qna_val_model = LLM_OPTIONS[qna_val_label]
    st.sidebar.markdown("---")
    st.sidebar.caption("Output goes to draft_questions/. Review in the Review QnA tab.")


# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE TABS
# ══════════════════════════════════════════════════════════════════════════════
if is_rag:
    tab_home, tab_ask, tab_search, tab_ingest, tab_eval, tab_lb, tab_info = st.tabs(
        ["Home", "Ask", "Search", "Ingest", "Evaluate", "Leaderboard", "DB info"]
    )

    # ── HOME ─────────────────────────────────────────────────────────────────
    with tab_home:
        st.markdown("""
<div style='text-align:center; padding: 2rem 1rem 1rem;'>
    <div style='font-size:2.8rem; font-weight:700; letter-spacing:-0.5px;'>🔭 TelcoLens</div>
    <div style='font-size:1.1rem; color:#666; margin-top:0.5rem; max-width:600px; margin-left:auto; margin-right:auto;'>
        Ask natural language questions directly against authoritative telecom standards
        with source citations, across 7 standards bodies.
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Indexed corpora")

        CORPUS_STATS = [
            ("O-RAN",    "oran",    "📡", "O-RAN Alliance R003–R005",       169130),
            ("3GPP",     "3gpp",    "📶", "Release 8 through 19",            2229734),
            ("ETSI",     "etsi",    "🇪🇺", "ENI, ZSM, MEC, NFV, SAI, F5G",  984728),
            ("ITU-T",    "itu",     "🌐", "All 27 active series",            575762),
            ("TM Forum", "tmforum", "🔧", "TMF Open APIs, ODA",             6905),
            ("CAMARA",   "camara",  "🔌", "Fall25 — 75+ API repos",         3919),
            ("GSMA",     "gsma",    "📱", "Open Gateway & security series", 121118),
        ]
        CARD_BG     = "#1a1a1a"
        CARD_BORDER = "#cc0000"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Standards bodies", "7")
        m2.metric("Total published chunks", "4,592,971")
        m3.metric("Total tokens", "1.09 billion")
        m4.metric("Embedding models", "6")

        st.markdown("")
        cols1 = st.columns(4)
        cols2 = st.columns(4)
        all_cols = cols1 + cols2 + [None]
        for col, (name, key, icon, desc, chunks) in zip(all_cols, CORPUS_STATS):
            if col is None:
                continue
            with col:
                chunk_str = f"{chunks:,} chunks" if chunks > 0 else "🔜 coming soon"
                st.markdown(
                    f"<div style='background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px;"
                    f"padding:14px 14px 12px; text-align:center; min-height:120px;'>"
                    f"<div style='font-size:1.6rem;'>{icon}</div>"
                    f"<div style='font-weight:600; font-size:0.9rem; margin-top:4px; color:#ffffff;'>{name}</div>"
                    f"<div style='font-size:0.75rem; color:#cccccc; margin-top:2px; line-height:1.3;'>{desc}</div>"
                    f"<div style='font-size:0.72rem; color:{CARD_BORDER}; margin-top:6px; font-family:monospace;'>{chunk_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("#### Get started")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "<div style='background:#f0f7ff;border:1px solid #cce0ff;border-radius:10px;padding:14px;'>"
                "<div style='font-weight:600; margin-bottom:6px;'>1. Select corpus & model</div>"
                "<div style='font-size:0.82rem;color:#555;'>Use the sidebar to choose a corpus, "
                "embedder, LLM, and DB path.</div></div>", unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                "<div style='background:#f0fff4;border:1px solid #b2f0c8;border-radius:10px;padding:14px;'>"
                "<div style='font-weight:600; margin-bottom:6px;'>2. Ask a question</div>"
                "<div style='font-size:0.82rem;color:#555;'>Go to the <b>Ask</b> tab and type any "
                "technical question. The pipeline retrieves relevant chunks and generates a grounded answer.</div>"
                "</div>", unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                "<div style='background:#fff8f0;border:1px solid #ffdcb2;border-radius:10px;padding:14px;'>"
                "<div style='font-weight:600; margin-bottom:6px;'>3. Evaluate & compare</div>"
                "<div style='font-size:0.82rem;color:#555;'>Use the <b>Evaluate</b> tab to score RAG quality "
                "across 5 metrics. Results appear on the <b>Leaderboard</b> automatically.</div>"
                "</div>", unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; font-size:0.82rem; color:#888;'>"
            "📄 <a href='https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf' target='_blank'>Cookbook</a> &nbsp;·&nbsp; "
            "💻 <a href='https://github.com/Znbne/telcolens' target='_blank'>GitHub</a> &nbsp;·&nbsp; "
            "🤗 <a href='https://huggingface.co/datasets/GSMA/telcolens-chunks' target='_blank'>HuggingFace Dataset</a>"
            "</div>", unsafe_allow_html=True,
        )

    # ── ASK ──────────────────────────────────────────────────────────────────
    with tab_ask:
        st.caption("Retrieval + generation — asks the LLM to answer using retrieved chunks.")
        question = st.text_input("Question",
            placeholder="What are the xApp conflict mitigation strategies in O-RAN?",
            label_visibility="collapsed")
        col_ask1, col_ask2, col_ask3 = st.columns(3) 
        with col_ask1:
            top_k_ask = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=8)
        with col_ask2:
            threshold_ask = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)
        with col_ask3:
            selected_search_method = st.selectbox(
                "Search Method",
                options=["dense", "bm25", "hybrid"],
                index=0,
                key="ask_search_method"
            )
        ask_btn = st.button("Ask ✦", type="primary")

        if ask_btn and question.strip():
            with st.spinner("Retrieving and generating…"):
                try:
                    from open_telco_rag.core.pipeline import generate, search
                    embedder = _get_embedder(model_key, backend=embed_backend)
                    store    = _get_store(source_type, model_key, db_path, collection_override)
                    results  = search(question, store=store, embedder=embedder,
                                      top_k=top_k_ask, threshold=threshold_ask,
                                      search_method=selected_search_method)
                    answer   = generate(question, results, source_type=source_type,
                                        generator=_build_generator(llm_model))

                    import math as _math
                    mean_sim = (sum(_math.exp(-r.distance) for r in results) / len(results)
                                if results else 0.0)
                    if mean_sim >= 0.65:   conf_color, conf_label = "#2ecc71", "High confidence"
                    elif mean_sim >= 0.45: conf_color, conf_label = "#f39c12", "Medium confidence"
                    else:                  conf_color, conf_label = "#e74c3c", "Low confidence"

                    st.markdown(
                        f"<div style='font-size:0.75rem;color:#888;margin-bottom:6px;"
                        f"display:flex;align-items:center;gap:8px;'>"
                        f"<span style='width:8px;height:8px;border-radius:50%;"
                        f"background:{conf_color};display:inline-block;'></span>"
                        f"<span>{conf_label} · sim {mean_sim:.3f} · {answer['chunks_used']} chunks</span>"
                        f"<span class='model-tag'>{llm_label}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"<div class='answer-box'>{answer['answer']}</div>",
                                unsafe_allow_html=True)

                    if answer.get("sources"):
                        chips = "".join(f"<span class='source-chip'>📄 {s}</span>"
                                        for s in answer["sources"])
                        st.markdown(chips, unsafe_allow_html=True)

                    if results:
                        with st.expander(f"Retrieved chunks ({len(results)})"):
                            for i, r in enumerate(results, 1):
                                source  = r.metadata.get("source", "—")
                                heading = r.metadata.get("heading", "")
                                sim     = _similarity(r.distance)
                                st.markdown(f"**{i}. {source}**"
                                            + (f" — {heading}" if heading else "")
                                            + f"  `sim {sim}`")
                                st.text(r.text)
                                st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")
        elif ask_btn:
            st.warning("Please enter a question.")

    # ── SEARCH ───────────────────────────────────────────────────────────────
    with tab_search:
        st.caption("Pure vector retrieval — returns the most relevant chunks, no LLM involved.")
        query = st.text_input("Query",
            placeholder="near-RT RIC E2 interface latency requirements",
            label_visibility="collapsed", key="search_query")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            top_k_search = st.number_input("Top K", 1, 20, 5, key="sk_topk")
        with col_s2:
            threshold_search = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05, key="sk_threshold")
        with col_s3:
            selected_search_method_s = st.selectbox(
                "Search Method",
                options=["dense", "bm25", "hybrid"],
                index=0,
                key="search_search_method"
            )
        search_btn = st.button("Search", type="primary", key="search_btn")

        if search_btn and query.strip():
            with st.spinner("Searching…"):
                try:
                    from open_telco_rag.core.pipeline import search
                    embedder = _get_embedder(model_key, backend=embed_backend)
                    store    = _get_store(source_type, model_key, db_path, collection_override)
                    results  = search(query, store=store, embedder=embedder,
                                      top_k=top_k_search, threshold=threshold_search,
                                      search_method=selected_search_method_s)
                    if not results:
                        st.info("No results found.")
                    else:
                        displayed = _render_chunk_cards(results)
                        st.caption(f'{displayed} results for: "{query}"')
                        with st.expander(f"Full chunk text ({displayed} chunks)"):
                            for i, r in enumerate(results, 1):
                                source  = r.metadata.get("source", "—")
                                heading = r.metadata.get("heading", "")
                                sim     = _similarity(r.distance)
                                st.markdown(f"**{i}. {source}**"
                                            + (f" — {heading}" if heading else "")
                                            + f"  `sim {sim}`")
                                st.text(r.text)
                                st.divider()
                except Exception as e:
                    st.error(f"Error: {e}")
        elif search_btn:
            st.warning("Please enter a query.")

# ── INGEST ───────────────────────────────────────────────────────────────
    with tab_ingest:
        # ── SECTION 1: GLOBAL DATA SETUP ──────────────────────────────────────
        col_i0_1, col_i0_2 = st.columns(2)
        with col_i0_1:
            selected_source_type = st.selectbox(
                "Document Source Type",
                options=["3gpp", "oran", "camara", "tmforum", "gsma", "itu", "etsi"],
                index=0,
                help="Selects the parsing class and expected file formats."
            )
        with col_i0_2:
            st.text(f"Active Model: {model_key}")

        col_data1, col_data2 = st.columns(2)
        with col_data1:
            data_dir = st.text_input("Data directory", value=f"data/{selected_source_type}")
        with col_data2:
            ingest_collection = st.text_input("Collection name",
                                              placeholder=f"{selected_source_type}_{model_key}")

        # ── SECTION 2: CHUNKING CONFIGURATION ─────────────────────────────────
        st.markdown("---")
        st.markdown("##### Chunking Parameters")

        # Load dynamic strategies from JSON configs
        chunk_strat_dict = _load_chunking_strategies()
        enrich_strat_dict = _load_enrichment_strategies()
        
        # ── ROW 1: Strategy & Distance Metric Side-by-Side ──
        col_strat, col_dist = st.columns(2)
        
        with col_strat:
            selected_strategy = st.selectbox(
                "Chunking Strategy", 
                options=list(chunk_strat_dict.keys()), 
                index=0,
                format_func=lambda k: chunk_strat_dict[k]["display_name"],
                help="Select layout splitting parameters."
            )
            strat_meta = chunk_strat_dict[selected_strategy]
            st.caption(f"💡 {strat_meta['help_text']}")
            
        with col_dist:
            selected_distance = st.selectbox(
                "Distance Metric (Index Vector Space)",
                options=["l2", "cosine", "ip"],
                index=0,
                help="l2 = Euclidean, cosine = angular matching (recommended for prepended text descriptions)."
            )

        # Parse labels based on the selected strategy row above
        unit_label = strat_meta["unit"].capitalize()

        # ── ROW 2: Chunk Size & Overlap Side-by-Side ──
        col_size, col_overlap = st.columns(2)
        
        with col_size:
            chunk_size = st.number_input(
                f"Chunk size ({unit_label})", 
                min_value=strat_meta.get("min_size", 64), 
                max_value=strat_meta.get("max_size", 4096), 
                value=strat_meta.get("default_size", 1024)
            )
                
        with col_overlap:
            if "default_overlap" in strat_meta and strat_meta["default_overlap"] > 0:
                overlap = st.number_input(
                    f"Overlap ({unit_label})", 
                    min_value=0, 
                    max_value=strat_meta.get("max_size", 4096) // 2, 
                    value=strat_meta["default_overlap"]
                )
            else:
                overlap = st.number_input(
                    f"Overlap", 
                    min_value=0, max_value=0, value=0, 
                    disabled=True, 
                    help="Not utilized by this configuration strategy."
                )

        # ── DYNAMIC ENRICHMENT CONFIGURATIONS ───────────────────────
        st.markdown("---")
        st.markdown("##### Chunk Enrichment (Pre-Embedding Process)")
        col_en1, col_en2 = st.columns(2)
        #col_en1, col_en2 = col_i1, col_i2 
        
        with col_en1:
            selected_enrichment = st.selectbox(
                "Enrichment Strategy",
                options=list(enrich_strat_dict.keys()),
                index=0,
                format_func=lambda k: enrich_strat_dict[k]["display_name"],
                help="Appends additional domain context, vocabulary mappings, or structural headers before text embedding."
            )
            st.caption(f"💡 {enrich_strat_dict[selected_enrichment]['help_text']}")
            
        with col_en2:
            # Dynamically gray out the model selection if the chosen strategy doesn't require an LLM
            req_llm = enrich_strat_dict[selected_enrichment].get("requires_llm", False)
            selected_enrichment_model = st.selectbox(
                "Enrichment Model (If LLM strategy chosen)",
                options=list(LLM_OPTIONS.keys()),
                format_func=lambda k: k.split("/")[-1],
                disabled=not req_llm,
                help="Requires functional LLM processing credentials when active."
            )
                
        col_i3, col_i4 = st.columns(2)
        with col_i3:
            use_marker = st.checkbox("Use Marker API (better PDF quality)")
        with col_i4:
            no_resume  = st.checkbox("Disable resume (reprocess all files)")
        ingest_btn = st.button("Run ingest ▶", type="primary")

        if ingest_btn:
            col_name  = ingest_collection.strip() or f"{selected_source_type}_{model_key}"
            log_lines = [
                f"Ingesting {selected_source_type.upper()} documents",
                f"  Data dir:   {data_dir}", 
                f"  Model:      {model_key}",
                f"  Collection: {col_name}",
                f"  Strategy:   {selected_strategy}",
                f"  Embedder: {embed_backend.upper()} + {model_key.upper()}", 
                f"  Enrichment: {selected_enrichment.upper()}", 
                f"  PDF loader: {'Marker API' if use_marker else 'pypdf (default)'}",
                ""
            ]
            log_box = st.empty()
            log_box.markdown("<div class='log-box'>" + "\n".join(log_lines) + "Running…</div>",
                             unsafe_allow_html=True)
            try:
                from open_telco_rag.core.pipeline import ingest
                from open_telco_rag.stores.chroma import ChromaStore
                
                src      = _get_source(selected_source_type, use_marker=use_marker)
                embedder = _get_embedder(model_key, backend=embed_backend)
                store    = _get_store(
                    source_type=selected_source_type, 
                    model_key=model_key, 
                    db_path=db_path, 
                    collection=col_name,
                    distance_metric=selected_distance
                )
                
                chunk_kwargs = {
                    "strategy": selected_strategy,
                    "enrichment": selected_enrichment,
                    "enrichment_model": LLM_OPTIONS[selected_enrichment_model]
                }
                
                if strat_meta["unit"] == "tokens":
                    chunk_kwargs["chunk_tokens"] = chunk_size
                    chunk_kwargs["overlap_tokens"] = overlap
                elif selected_strategy in ["lumber_chunker"]:
                    if "llm_client" in st.session_state:
                        chunk_kwargs["llm_client"] = st.session_state["llm_client"]
                    else:
                        from open_telco_rag.generators.openrouter import OpenRouterGenerator
                        chunk_kwargs["llm_client"] = OpenRouterGenerator(model=LLM_OPTIONS[selected_enrichment_model])
                else:
                    chunk_kwargs["chunk_size"] = chunk_size
                    chunk_kwargs["overlap"] = overlap
                    
                stats = ingest(
                    source=src, 
                    store=store, 
                    embedder=embedder,
                    data_dir=data_dir, 
                    resume=not no_resume,
                    **chunk_kwargs
                )
                
                log_lines += [f"Files processed: {stats.files_processed}",
                              f"Chunks stored:   {stats.chunks_stored}",
                              f"Skipped:         {stats.skipped_files}",
                              f"Errors:          {stats.error_files}"]
                log_box.markdown("<div class='log-box'>" + "\n".join(log_lines) + "</div>",
                                 unsafe_allow_html=True)
                st.success(f"Done — {stats.chunks_stored:,} chunks stored, "
                           f"{stats.skipped_files} skipped, {stats.error_files} errors.")
                
                from open_telco_rag.core.efficiency import render_efficiency_dashboard
                render_efficiency_dashboard(stats)
                
            except Exception as e:
                log_lines.append(f"ERROR: {e}")
                log_box.markdown("<div class='log-box'>" + "\n".join(log_lines) + "</div>",
                                 unsafe_allow_html=True)
                st.error(f"Ingest failed: {e}")

    # ── EVALUATE ─────────────────────────────────────────────────────────────
    with tab_eval:
        st.caption(f"RAG generation uses the sidebar LLM ({llm_label}). Metrics scoring uses the model selected below.")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            from pathlib import Path
            eval_q_dir = Path("configs/eval_questions")
            available_eval_files = [str(p) for p in sorted(eval_q_dir.glob("*.json"))] if eval_q_dir.exists() else []

            if available_eval_files:
                eval_options = available_eval_files + ["Custom path..."]
                selected_eval_file = st.selectbox(
                    "Questions file (.json)",
                    options=eval_options,
                    key="eval_questions_file_select",
                    help="Select a validated QnA dataset from configs/eval_questions/"
                )
                if selected_eval_file == "Custom path...":
                    questions_file = st.text_input("Custom Questions file path", value="configs/eval_questions/3gpp_r19_qna.json", key="eval_questions_custom_path")
                else:
                    questions_file = selected_eval_file
            else:
                questions_file = st.text_input("Questions file (.json)", value="configs/eval_questions/3gpp_r19_qna.json", key="eval_questions_text_input")

            top_k_eval = st.number_input("Top K", 1, 20, 5, key="eval_topk")
            output_dir = st.text_input("Output directory", value="results/gui")
        with col_e2:
            eval_run_name = st.text_input(
                "Run name",
                value="",
                placeholder="e.g. exp_prompt_v2",
                help="Custom label to identify this experiment run on the leaderboard."
            )
            METRICS_LLM_OPTIONS = _load_llm_keys() 
            metrics_llm_label = st.selectbox("Metrics evaluation model",
                                             list(METRICS_LLM_OPTIONS.keys()))
            eval_delay = st.slider("Delay between questions (s)", 0.0, 5.0, 1.0, 0.5)
            selected_search_method_eval = st.selectbox(
                "Evaluation Search Pass Method",
                options=["dense", "bm25", "hybrid"],
                index=0,
                key="eval_search_method"
            )

        rag_llm_model     = llm_model
        metrics_llm_model = METRICS_LLM_OPTIONS[metrics_llm_label]

        col_eval_opts1, col_eval_opts2, col_eval_opts3 = st.columns(3)
        with col_eval_opts1:
            include_baseline = st.checkbox("Include baseline (no retrieval)", value=False)
            inject_metadata  = st.checkbox("Inject domain metadata", value=False)
        with col_eval_opts2:
            include_keywords = st.checkbox("Append question keywords to query", value=False)
            use_reranker     = st.checkbox("Enable Reranker", value=False)
        with col_eval_opts3:
            if use_reranker:
                reranker_model = st.selectbox("Reranker Model", ["ms-marco-minilm", "otel-reranker-0.6b", "bge-reranker"])
                rerank_top_k   = st.number_input("Chunks to pass to LLM", 1, 20, 3, key="eval_rerank_topk")
            else:
                reranker_model = "0"
                rerank_top_k   = top_k_eval  


        eval_btn = st.button("Run evaluation ▶", type="primary", key="eval_btn")

        if eval_btn:
            from pathlib import Path
            import json
            qpath = Path(questions_file)
            if not qpath.exists():
                st.error(f"Questions file not found: {qpath}")
            else:
                try:
                    data  = json.loads(qpath.read_text(encoding="utf-8"))
                    qs    = data.get("questions", [])
                    n     = len(qs)
                    col_name = collection_override.strip() or f"{source_type}_{model_key}"
                    st.info(f"Running evaluation on **{n} questions** · collection `{col_name}` · "
                            f"RAG: {llm_label} · Metrics: {metrics_llm_label}")

                    from open_telco_rag.evaluation.evaluate import evaluate_corpus
                    from open_telco_rag.evaluation.questions import EvalQuestion
                    from open_telco_rag.evaluation._claude import make_llm_call

                    embedder = _get_embedder(model_key, backend=embed_backend)
                    store        = _get_store(source_type, model_key, db_path, collection_override)
                    questions    = [EvalQuestion.from_dict(q) for q in qs]
                    llm_call     = make_llm_call("openrouter", metrics_llm_model)
                    rag_llm_call = make_llm_call("openrouter", rag_llm_model)

                    progress_bar = st.progress(0, text="Starting…")
                    log_area     = st.empty()
                    log_lines    = []

                    import open_telco_rag.evaluation.evaluate as _eval_mod
                    if hasattr(_eval_mod, "_true_orig_eq"):
                        _orig_eq = _eval_mod._true_orig_eq
                    else:
                        _orig_eq = _eval_mod.evaluate_question
                        _eval_mod._true_orig_eq = _orig_eq

                    q_counter = [0]
                    total_expected = n * 2 if include_baseline else n

                    def _patched_eq(q, *args, **kwargs):
                        qr = _orig_eq(q, *args, **kwargs)
                        q_counter[0] += 1
                        idx = q_counter[0]
                        
                        # Clamp ratio between 0.0 and 1.0 so it never exceeds 100%
                        progress_val = min(1.0, idx / total_expected)
                        progress_bar.progress(progress_val, text=f"Question {idx}/{total_expected}")

                        if qr.error:
                            log_lines.append(f"[{idx}/{total_expected}] ERROR: {qr.error}")
                        else:
                            log_lines.append(
                                f"[{idx}/{total_expected}] faithful={qr.faithfulness:.2f}  "
                                f"relevancy={qr.answer_relevancy:.2f}  "
                                f"sim={qr.mean_similarity:.3f}  "
                                f"recall={qr.context_recall:.2f}  "
                                f"correct={qr.answer_correctness:.2f}"
                            )
                        log_area.markdown("<div class='log-box'>" + "\n".join(log_lines[-20:]) + "</div>",
                                         unsafe_allow_html=True)
                        return qr

                    _eval_mod.evaluate_question = _patched_eq
                    
                    try:
                        # Load the actual reranker object if enabled
                        active_reranker = _get_reranker(reranker_model) if use_reranker else None
                        
                        result, baseline_result = evaluate_corpus(
                            corpus=source_type, store=store, embedder=embedder,
                            questions=questions, model=model_key, top_k=top_k_eval,
                            delay=eval_delay, output_dir=Path(output_dir),
                            llm_call=llm_call, rag_llm_call=rag_llm_call,
                            include_baseline=include_baseline, rag_generator_model=rag_llm_model,
                            inject_metadata=inject_metadata, search_method=selected_search_method_eval,
                            include_keywords=include_keywords, 
                            reranker=active_reranker,  
                            rerank_top_k=rerank_top_k,
                        )
                    finally:
                        # 2. ALWAYS restore the original function, even if the pipeline crashes
                        _eval_mod.evaluate_question = _orig_eq

                    scores = [result.mean_faithfulness, result.mean_answer_relevancy,
                              result.mean_similarity, result.mean_context_recall,
                              result.mean_answer_correctness]
                    valid = [s for s in scores if s >= 0]
                    avg   = round(sum(valid) / len(valid), 3) if valid else -1.0

                    def _bl(v):
                        return round(max(0, v), 3) if v is not None and v >= 0 else None
                    
                    q_filename = Path(questions_file).name if questions_file else f"{source_type}.json"

                    total_cost = sum(getattr(q, "cost", 0.0) for q in result.question_results)
                    total_cached = sum(getattr(q, "cached_tokens", 0) for q in result.question_results)
                    total_reasoning = sum(getattr(q, "reasoning_tokens", 0) for q in result.question_results)

                    avg_search = round(sum(getattr(q, "search_latency", 0.0) for q in result.question_results) / max(1, len(result.question_results)), 2)
                    avg_gen = round(sum(getattr(q, "gen_latency", 0.0) for q in result.question_results) / max(1, len(result.question_results)), 2)
                    avg_total = round(sum(getattr(q, "latency", 0.0) for q in result.question_results) / max(1, len(result.question_results)), 2)

                    lb_row = {
                        "timestamp":             __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "run_name":              eval_run_name.strip() or "default", 
                        "rag_model":             rag_llm_model,
                        "evaluator_model":       metrics_llm_model,
                        "embedder":              model_key,
                        "corpus":               source_type,
                        "collection":            col_name,
                        "questions_file":         q_filename,
                        "top_k":                 top_k_eval,
                        "chunks_to_llm":         rerank_top_k,                    
                        "reranker":              reranker_model,
                        "n_questions":           result.n_questions,
                        "inject_metadata":       inject_metadata,
                        "search_method":         selected_search_method_eval,
                        "embed_backend":         embed_backend,
                        "avg_search_latency_sec": avg_search,            
                        "avg_gen_latency_sec":    avg_gen,              
                        "avg_total_latency_sec":  avg_total,             
                        "total_api_tokens":      sum(getattr(q, "prompt_tokens", 0) + getattr(q, "completion_tokens", 0) for q in result.question_results),
                        "total_cached_tokens":   total_cached,
                        "total_reasoning_tokens": total_reasoning,
                        "run_cost_usd":          round(total_cost, 6),
                        "faithfulness":          round(result.mean_faithfulness, 3),
                        "answer_relevancy":      round(result.mean_answer_relevancy, 3),
                        "mean_similarity":       round(result.mean_similarity, 3),
                        "context_recall":        round(max(0, result.mean_context_recall), 3),
                        "answer_correctness":    round(max(0, result.mean_answer_correctness), 3),
                        "average":               avg,
                        "bl_faithfulness":       _bl(baseline_result.mean_faithfulness)       if baseline_result else None,
                        "bl_answer_relevancy":   _bl(baseline_result.mean_answer_relevancy)   if baseline_result else None,
                        "bl_context_recall":     _bl(baseline_result.mean_context_recall)     if baseline_result else None,
                        "bl_answer_correctness": _bl(baseline_result.mean_answer_correctness) if baseline_result else None,
                        "bl_average":            round(sum(v for v in [
                                                     baseline_result.mean_faithfulness,
                                                     baseline_result.mean_answer_relevancy,
                                                     baseline_result.mean_context_recall,
                                                     baseline_result.mean_answer_correctness,
                                                 ] if v >= 0) / max(1, sum(1 for v in [
                                                     baseline_result.mean_faithfulness,
                                                     baseline_result.mean_answer_relevancy,
                                                     baseline_result.mean_context_recall,
                                                     baseline_result.mean_answer_correctness,
                                                 ] if v >= 0)), 3) if baseline_result else None,
                    }
                    lb_path = Path("results/leaderboard.json")
                    lb_path.parent.mkdir(parents=True, exist_ok=True)
                    lb_rows = []
                    if lb_path.exists():
                        try:
                            lb_rows = json.loads(lb_path.read_text(encoding="utf-8"))
                        except Exception:
                            lb_rows = []
                    lb_rows.append(lb_row)
                    lb_path.write_text(json.dumps(lb_rows, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

                    st.markdown("---")
                    st.markdown(f"**Results — {result.corpus.upper()} · model={result.model} · "
                                f"top-k={result.top_k} · n={result.n_questions}**")

                    rows = [
                        ("Faithfulness",          result.mean_faithfulness,       "Generation"),
                        ("Answer relevancy",       result.mean_answer_relevancy,   "Generation"),
                        ("Mean similarity",        result.mean_similarity,         "Retrieval"),
                        ("Source coverage (mean)", result.mean_source_coverage,    "Retrieval"),
                        ("Context recall",         result.mean_context_recall,     "Retrieval+GT"),
                        ("Answer correctness",     result.mean_answer_correctness, "Generation+GT"),
                    ]
                    st.markdown("<div class='eval-row eval-row-header'>"
                                "<span class='eval-metric'>Metric</span>"
                                "<span class='eval-score'>Score</span>"
                                "<span class='eval-type'>Type</span></div>",
                                unsafe_allow_html=True)
                    for label, value, kind in rows:
                        score_str = "N/A" if value < 0 else (
                            f"{value:.1f}" if "coverage" in label.lower() else f"{value:.3f}")
                        st.markdown(f"<div class='eval-row eval-row-data'>"
                                    f"<span class='eval-metric'>{label}</span>"
                                    f"<span class='eval-score'>{score_str}</span>"
                                    f"<span class='eval-type'>{kind}</span></div>",
                                    unsafe_allow_html=True)

                    import plotly.graph_objects as go
                    qrs     = result.question_results
                    qlabels = [f"Q{i+1}" for i in range(len(qrs))]

                    def _metric_chart(title, values, color, mean_val):
                        clean = [max(0.0, v) if v >= 0 else 0.0 for v in values]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=qlabels, y=clean, marker_color=color,
                                             text=[f"{v:.2f}" for v in clean],
                                             textposition="outside", name=title))
                        if mean_val >= 0:
                            fig.add_hline(y=mean_val, line_dash="dash", line_color="#555",
                                          line_width=1.5,
                                          annotation_text=f"avg {mean_val:.3f}",
                                          annotation_position="top right",
                                          annotation_font_size=11)
                        fig.update_layout(title=dict(text=title, font=dict(size=13)),
                                          yaxis=dict(range=[0, 1.15], gridcolor="#eee",
                                                     tickformat=".2f"),
                                          xaxis=dict(tickangle=-45 if len(qrs) > 20 else 0),
                                          height=300, margin=dict(t=45, b=50, l=30, r=20),
                                          plot_bgcolor="white", paper_bgcolor="white",
                                          font=dict(size=11), showlegend=False)
                        return fig

                    metrics_to_plot = [
                        ("Faithfulness",       [q.faithfulness       for q in qrs], "#4C9BE8", result.mean_faithfulness),
                        ("Answer relevancy",   [q.answer_relevancy   for q in qrs], "#82C5A0", result.mean_answer_relevancy),
                        ("Mean similarity",    [q.mean_similarity    for q in qrs], "#F5A623", result.mean_similarity),
                        ("Context recall",     [q.context_recall     for q in qrs], "#9B59B6", result.mean_context_recall),
                        ("Answer correctness", [q.answer_correctness for q in qrs], "#E74C3C", result.mean_answer_correctness),
                    ]
                    r1c1, r1c2, r1c3 = st.columns(3)
                    r2c1, r2c2, _    = st.columns([1, 1, 1])
                    for col, (title, values, color, mean_val) in zip(
                        [r1c1, r1c2, r1c3, r2c1, r2c2], metrics_to_plot
                    ):
                        with col:
                            st.plotly_chart(_metric_chart(title, values, color, mean_val),
                                            width="stretch")
                    st.success(f"Results saved → {output_dir}/")

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ── LEADERBOARD ──────────────────────────────────────────────────────────
    with tab_lb:
        st.caption("Evaluation results across generator models, embedders, and corpora — sorted by average score.")
        import json as _json
        from pathlib import Path as _Path
        lb_path = _Path("results/leaderboard.json")
        col_lb1, col_lb2 = st.columns([3, 1])
        with col_lb2:
            if st.button("Refresh", key="lb_refresh"):
                st.rerun()

        if not lb_path.exists():
            st.info("No leaderboard data yet. Run an evaluation to populate it.")
        else:
            try:
                rows = _json.loads(lb_path.read_text(encoding="utf-8"))
                if not rows:
                    st.info("Leaderboard file is empty.")
                else:
                    import pandas as pd
                    import pandas as _pd
                    import plotly.graph_objects as go
                    df = pd.DataFrame(rows)

                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        filter_collection = st.selectbox("Collection",
                            ["All"] + sorted(df["collection"].unique().tolist()), key="lb_collection")
                    with col_f2:
                        filter_embedder = st.selectbox("Embedder",
                            ["All"] + sorted(df["embedder"].unique().tolist()), key="lb_embedder")
                    with col_f3:
                        filter_model = st.selectbox("Generator",
                            ["All"] + sorted(df["rag_model"].unique().tolist()), key="lb_model")

                    if filter_collection != "All": df = df[df["collection"] == filter_collection]
                    if filter_embedder   != "All": df = df[df["embedder"]   == filter_embedder]
                    if filter_model      != "All": df = df[df["rag_model"]  == filter_model]

                    if df.empty:
                        st.info("No results match the selected filters.")
                    else:
                        df = df.sort_values("average", ascending=False).reset_index(drop=True)
                        df.insert(0, "rank", range(1, len(df) + 1))
                        has_baseline = "bl_average" in df.columns and df["bl_average"].notna().any()
                        if has_baseline:
                            df["Δ correctness"] = (df["answer_correctness"] - df["bl_answer_correctness"]).round(3)
                            df["Δ recall"]      = (df["context_recall"]     - df["bl_context_recall"]).round(3)
                            df["Δ avg"]         = (df["average"]            - df["bl_average"]).round(3)

                        display_cols = {
                            "rank": "Rank", "rag_model": "Generator",
                            "run_name": "Run Name",
                            "evaluator_model": "Evaluator", "embedder": "Embedder",
                            "collection": "Collection","questions_file" :"Question Set",
                            "inject_metadata": "Metadata Injected",
                            "include_keywords": "Keywords Added",       
                            "top_k": "Retrieved",                       
                            "reranker": "Reranker",                     
                            "chunks_to_llm": "LLM Chunks",
                            "search_method": "Search Method", 
                            "embed_backend": "Vector Backend",
                            "avg_search_latency_sec": "Search Latency",  
                            "avg_gen_latency_sec": "Gen Latency",        
                            "avg_total_latency_sec": "Total Latency",
                            "total_api_tokens": "Total Tokens",
                            "total_cached_tokens": "Cached Tokens",         
                            "total_reasoning_tokens": "Reasoning Tokens",   
                            "run_cost_usd": "Cost (USD)",
                            "faithfulness": "Faithful",
                            "answer_relevancy": "Relevancy", "mean_similarity": "Similarity",
                            "context_recall": "Recall", "answer_correctness": "Correctness",
                            "average": "RAG Avg",
                        }
                        if has_baseline:
                            display_cols.update({
                                "bl_answer_correctness": "BL Correctness",
                                "bl_context_recall": "BL Recall", "bl_average": "BL Avg",
                                "Δ correctness": "Δ Correctness", "Δ recall": "Δ Recall",
                                "Δ avg": "Δ Avg",
                            })
                        df_display = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
                        col_config = {k: st.column_config.NumberColumn(format="%.3f")
                                    for k in ["RAG Avg","Faithful","Relevancy","Similarity",
                                                "Recall","Correctness","BL Correctness","BL Recall","BL Avg"]}
                        col_config.update({k: st.column_config.NumberColumn(format="%+.3f")
                                        for k in ["Δ Correctness","Δ Recall","Δ Avg"]})
                        col_config.update({
                                        "Search Latency": st.column_config.NumberColumn(format="%.2f s"),  # Updated
                                        "Gen Latency": st.column_config.NumberColumn(format="%.2f s"),     # Updated
                                        "Total Latency": st.column_config.NumberColumn(format="%.2f s"),   # Updated
                                        "Total Tokens": st.column_config.NumberColumn(format="%d"),
                                        "Cached Tokens": st.column_config.NumberColumn(format="%d"),     
                                        "Reasoning Tokens": st.column_config.NumberColumn(format="%d"),  
                                        "Cost (USD)": st.column_config.NumberColumn(format="$ %.6f")     
                        })
                        st.dataframe(df_display, width="stretch", hide_index=True,
                                     column_config=col_config)

                        with st.expander("🗑 Remove a row"):
                            all_rows = _json.loads(lb_path.read_text(encoding="utf-8"))
                            row_labels = {
                                i: f"{r.get('timestamp','?')}  ·  {r.get('rag_model','?')}  ·  "
                                   f"{r.get('corpus','?')}  ·  avg={r.get('average','?')}"
                                for i, r in enumerate(all_rows)
                            }
                            selected_label = st.selectbox("Select row to remove",
                                options=list(row_labels.keys()),
                                format_func=lambda i: row_labels[i], key="lb_delete_select")
                            if st.button("Remove selected row", type="primary", key="lb_delete_btn"):
                                all_rows.pop(selected_label)
                                lb_path.write_text(_json.dumps(all_rows, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
                                st.success("Row removed.")
                                st.rerun()

                        st.markdown("---")
                        avg_by_model = (df.groupby("rag_model")["average"].mean()
                                        .sort_values(ascending=False).reset_index())
                        fig = go.Figure(go.Bar(
                            x=avg_by_model["rag_model"], y=avg_by_model["average"],
                            marker_color="#4C9BE8",
                            text=[f"{v:.3f}" for v in avg_by_model["average"]],
                            textposition="outside"))
                        fig.update_layout(title="Average score by generator model",
                                          yaxis=dict(range=[0, 1.15], gridcolor="#eee", tickformat=".2f"),
                                          height=320, margin=dict(t=45, b=60, l=30, r=20),
                                          plot_bgcolor="white", paper_bgcolor="white",
                                          font=dict(size=11), xaxis=dict(tickangle=-20),
                                          showlegend=False)
                        st.plotly_chart(fig, width="stretch")
                        st.caption(f"Evaluator: {df['evaluator_model'].iloc[0]}  ·  "
                                   f"{len(df)} run(s) shown  ·  leaderboard at `{lb_path}`")
            except Exception as e:
                st.error(f"Could not load leaderboard: {e}")

    # ── DB INFO ───────────────────────────────────────────────────────────────
    with tab_info:
        if st.button("Refresh", key="info_refresh"):
            st.cache_resource.clear()
        try:
            import chromadb
            client      = chromadb.PersistentClient(path=db_path)
            collections = client.list_collections()
            total       = sum(client.get_collection(c.name).count() for c in collections)
            m1, m2, m3 = st.columns(3)
            m1.metric("Collections", len(collections))
            m2.metric("Total chunks", f"{total:,}")
            m3.metric("DB path", db_path)
            st.markdown("---")
            if collections:
                for col in sorted(collections, key=lambda c: c.name):
                    c = client.get_collection(col.name)
                    st.markdown(f"<div class='col-row'>"
                                f"<span class='col-name'>{col.name}</span>"
                                f"<span class='col-count'>{c.count():,} chunks</span></div>",
                                unsafe_allow_html=True)
            else:
                st.info(f"No collections found in '{db_path}'. Run ingest first.")
        except Exception as e:
            st.error(f"Could not read database: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# QNA GENERATION PIPELINE TABS
# ══════════════════════════════════════════════════════════════════════════════
else:
    tab_genqa, tab_review, tab_qna_info = st.tabs(["Generate QnA", "Review QnA", "QnA Info"])

    # ── GENERATE QNA ─────────────────────────────────────────────────────────
    with tab_genqa:
        st.caption("Build a generation pipeline from the controls below. "
                   "Outputs are DRAFTS — review them in the Review QnA tab.")

        from qna_generation.formats import VALID_FORMATS

        st.markdown("##### Sampling")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            genqa_sampling = st.selectbox(
                "Strategy", ["stratified", "boundary"], 
                index=0,
                key="genqa_sampling",
                help="stratified: random spread across documents.\n"
                     "boundary: one chunk per document section (heading-based).",
            )
        with col_g2:
            genqa_n_chunks = st.number_input("Chunks to sample", 1, 1000, 30, key="genqa_n_chunks")

        st.markdown("##### Question generation")
        col_g3, col_g4 = st.columns(2)
        with col_g3:
            genqa_formats = st.multiselect(
                "Question types", VALID_FORMATS, default=["open"], key="genqa_formats",
                help="open: free-text  ·  tf: true/false  ·  mc: multiple choice  ·  fill_blank: fill-in-the-blank",
            )
            genqa_n_questions = st.number_input("Total questions", 1, 1000, 20, key="genqa_n_questions")
        with col_g4:
            genqa_qpc = st.number_input(
                "Questions per chunk", 1, 10, 1, key="genqa_qpc",
                help="1 = new chunk per question.\n>1 = ask same chunk for multiple distinct questions.",
            )

        st.markdown("##### Filtering & Pre-validation Pass")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            use_competitor = st.checkbox("Enable Competitor Filter", value=False, help="Queries vector DB to ensure the question is specific to this chunk.")
            num_competitors = st.number_input("Number of competitor chunks to track", min_value=1, max_value=10, value=3, disabled=not use_competitor)
        with col_f2:
            validate_answer_mode = st.selectbox(
                "Validate Answerable Pass (validate_answer.py)",
                options=["Disabled", "Flag Failures", "Drop Failures"],
                index=0,
                help="Runs a secondary single-LLM pass to verify if the question is genuinely answerable."
            )

        st.markdown("##### Concurrency & summarization")
        col_gc1, col_gc2 = st.columns(2)
        with col_gc1:
            genqa_concurrency = st.number_input(
                "Max concurrent LLM calls", min_value=1, max_value=50, value=10,
                key="genqa_concurrency",
                help="How many LLM calls to run in parallel. Keep lower for rate-limited endpoints.",
            )
        with col_gc2:
            genqa_summarize = st.checkbox(
                "Summarize chunks",
                value=False, key="genqa_summarize",
                help="Adds a summarization step before question generation — chunk → summary → QnA.",
            )
        if genqa_summarize:
            genqa_summary_sentences = st.number_input(
                "Summary length (sentences)", min_value=2, max_value=10, value=5,
                key="genqa_summary_sentences",
            )
        else:
            genqa_summary_sentences = 5

        # ── MODULAR LLM JUDGES SELECTION SECTION ─────────────────────────────
        st.markdown("##### Active LLM Validation Judges")
        st.caption("Toggle the independent agent modules to include in the validation gauntlet:")
        
        col_j1, col_j2, col_j3 = st.columns(3)
        with col_j1:
            j_grounding = st.checkbox("Grounding Judge", value=False, help="Ensures reference answers derive strictly from the text.")
            j_domain = st.checkbox("Telecom Domain Relevance Judge", value=False, help="Filters out generic reading-comprehension prompts.")
        with col_j2:
            j_answerability = st.checkbox("Answerability Judge", value=False, help="Ensures the context has sufficient information to answer the question.")
            j_faithfulness = st.checkbox("Faithfulness Judge", value=False, help="Checks for extraneous or hallucinated specification references.")
        with col_j3:
            j_semantic = st.checkbox("Semantic Correctness Judge", value=False, help="Guarantees logical integrity and nuance preservation.")
            j_relevance = st.checkbox("Relevance Judge", value=False, help="Confirms questions target primary technical elements.")

        st.markdown("##### Paths, Deduplication & Output Workflow")
        col_g5, col_g6, col_g7 = st.columns(3)
        with col_g5:
            dedup_options = ["exact", "prefix", "none"]
            genqa_dedup = st.selectbox("Dedup mode", dedup_options, index=0, key="genqa_dedup")
        with col_g6:
            genqa_output_dir = st.text_input("Output directory", value="draft_questions", key="genqa_output_dir")
            skip_review = st.checkbox("Skip Manual Review (Auto-approve and merge directly)", value=False)
        with col_g7:
            genqa_set_name = st.text_input("QnA Set Name", value=f"{source_type}", key="genqa_set_name", help="Custom name for your question set (used for the output filename).")

        genqa_btn = st.button("Generate ▶", type="primary", key="genqa_btn")

        if genqa_btn:
            if not genqa_formats:
                st.warning("Select at least one question type.")
            else:
                log_box   = st.empty()
                log_lines = [f"Corpus: {source_type}  ·  source: {qna_source}", ""]

                def _log(msg):
                    log_lines.append(msg)
                    log_box.markdown("<div class='log-box'>" + "\n".join(log_lines[-25:]) + "</div>",
                                     unsafe_allow_html=True)

                _log("Starting Pipeline Core Engine…")
                try:
                    from qna_generation.sources import make_source
                    from qna_generation.flows import Flow
                    from qna_generation.io_utils import merge_with_existing, save_draft
                    from pathlib import Path as _Path
                    import json

                    src_kwargs = dict(source_type=qna_source, corpus=source_type)
                    if qna_source == "chroma":
                        src_kwargs.update(db_path=db_path, model_key=model_key,
                                          collection=qna_collection.strip() or None)
                    else:
                        src_kwargs.update(jsonl_path=qna_jsonl_path)

                    source = make_source(**src_kwargs)
                    _log(f"Source: {qna_source} ({source.count():,} chunks)")

                    # Assemble block array sequences sequentially
                    block_specs = [
                        {"type": "SampleChunksBlock",
                         "params": {"strategy": genqa_sampling, "n_chunks": genqa_n_chunks}},
                    ]
                    if genqa_summarize:
                        block_specs.append({
                            "type": "SummarizeChunkBlock",
                            "params": {"summary_sentences": genqa_summary_sentences,
                                       "max_concurrency": genqa_concurrency},
                        })

                    # Base generator sequence
                    block_specs.append({
                        "type": "GenerateQuestionBlock",
                        "params": {"formats": genqa_formats, "n": genqa_n_questions,
                                   "questions_per_chunk": genqa_qpc,
                                   "max_concurrency": genqa_concurrency},
                    })

                    # Inject Competitor Filter if checked
                    if use_competitor:
                        block_specs.append({
                            "type": "CompetitorFilterBlock",
                            "params": {"num_competitors": num_competitors}
                        })

                    # Inject Validate Answerable Block Mode]
                    if validate_answer_mode != "Disabled":
                        block_specs.append({
                            "type": "ValidateAnswerableBlock",
                            "params": {"drop_failures": (validate_answer_mode == "Drop Failures")}
                        })

                    # Dynamically inject active modular judges
                    if j_grounding:
                        block_specs.append({"type": "GroundingJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})
                    if j_domain:
                        block_specs.append({"type": "TelecomDomainRelevanceJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})
                    if j_answerability:
                        block_specs.append({"type": "AnswerabilityJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})
                    if j_faithfulness:
                        block_specs.append({"type": "FaithfulnessJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})
                    if j_semantic:
                        block_specs.append({"type": "SemanticCorrectnessJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})
                    if j_relevance:
                        block_specs.append({"type": "RelevanceJudgeBlock", "params": {"max_concurrency": genqa_concurrency}})

                    if genqa_dedup != "none":
                        block_specs.append({
                            "type": "DedupBlock", "params": {"mode": genqa_dedup},
                        })

                    flow = Flow.from_dict({"name": "gui_flow", "blocks": block_specs})
                    flow.set_model_config(generator="openrouter", model=qna_gen_model)
                    
                    # Bind validation credentials if any judge or validate pass is active
                    if any([j_grounding, j_domain, j_answerability, j_faithfulness, j_semantic, j_relevance]) or validate_answer_mode != "Disabled":
                        flow.set_validation_model_config(generator="openrouter", model=qna_val_model)

                    qa_pairs = flow.generate(source, source_type, log=_log)
                    _log(f"Generated {len(qa_pairs)} questions")

                    # Retrieve and clean the custom set name
                    safe_set_name = genqa_set_name.strip() or source_type
                    eval_path = _Path("configs/eval_questions") / f"{safe_set_name}.json"

                    # Check for Auto-Merge / Skip Review functionality
                    if skip_review:
                        _log("Skip review active: Auto-approving and merging directly into production eval set...")
                        for q in qa_pairs:
                            q["validated"] = True
                            q["generated"] = False  # Set to approved target layout status
                        
                        eval_path.parent.mkdir(parents=True, exist_ok=True)
                        existing = {"questions": []}
                        if eval_path.exists():
                            try:
                                existing = json.loads(eval_path.read_text(encoding="utf-8"))
                            except Exception:
                                existing = {"questions": []}
                        
                        existing["questions"].extend(qa_pairs)
                        eval_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
                        _log(f"Successfully auto-merged {len(qa_pairs)} items into {eval_path}")
                        st.success(f"Successfully auto-merged {len(qa_pairs)} questions directly into evaluation set! ✓")
                    else:
                        # Standard sequence: Save temporary review file layout via core utility
                        safe_set_name = genqa_set_name.strip() or source_type
                        eval_path = _Path("configs/eval_questions") / f"{safe_set_name}.json"
                        
                        merged = merge_with_existing(qa_pairs, eval_path)
                        
                        out_path = save_draft(
                            merged, 
                            _Path(genqa_output_dir), 
                            source_type,
                            n_questions=len(qa_pairs), 
                            model="automated_jury_flow",
                            set_name=safe_set_name,  
                        )
                        _log(f"Saved draft -> {out_path}")
                        st.success(f"Generated {len(qa_pairs)} draft questions -> {out_path}")
                        st.session_state["last_draft_path"] = str(out_path)

                except Exception as e:
                    _log(f"ERROR: {e}")
                    st.error(f"Generation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ── REVIEW QNA ───────────────────────────────────────────────────────────
    with tab_review:
        st.caption("Load a draft QnA file, review each question against its source chunk, "
                   "and approve / correct / reject before merging into configs/eval_questions/.")

        import json as _json2
        from pathlib import Path as _Path2

        # ── Source URL builder per SDO ────────────────────────────────────
        SDO_PORTAL_URLS = {
            "oran":    "https://orandownloadsweb.azurewebsites.net/specifications",
            "3gpp":    "https://www.3gpp.org/ftp/Specs/archive/",
            "etsi":    "https://www.etsi.org/deliver/",
            "itu":     "https://www.itu.int/rec/T-REC/en",
            "tmforum": "https://github.com/tmforum-rand/tmf-otel",
            "gsma":    "https://www.gsma.com/solutions-and-impact/gsma-open-gateway/",
            "camara":  "https://github.com/camaraproject",
        }

        def _build_doc_url(q: dict) -> tuple[str | None, str | None]:
            corpus       = q.get("corpus", "")
            source_doc   = q.get("source_document", "")
            canonical    = q.get("canonical_source", "")
            chunk_id     = q.get("chunk_id", "")
            portal       = SDO_PORTAL_URLS.get(corpus)

            page_num = None
            if "::" in chunk_id:
                parts = chunk_id.split("::")
                try:
                    page_num = int(parts[-1])
                except ValueError:
                    pass

            deep_link = None

            if corpus == "etsi" and source_doc:
                fname = source_doc.replace(".pdf", "").replace(".PDF", "")
                base_url = f"https://www.etsi.org/deliver/etsi_{fname}.pdf"
                deep_link = f"{base_url}#page={page_num}" if page_num else base_url
            elif corpus == "itu" and canonical:
                series = canonical.split(".")[0] if "." in canonical else canonical[:1]
                rec    = canonical.replace(".", "-")
                deep_link = f"https://www.itu.int/rec/T-REC-{rec}/en"
            elif corpus == "3gpp" and canonical:
                series = canonical.split(".")[0] if "." in canonical else ""
                if series:
                    deep_link = f"https://www.3gpp.org/ftp/Specs/archive/{series}_series/"
            elif corpus == "oran":
                deep_link = portal
            elif corpus == "camara" and canonical:
                deep_link = f"https://github.com/camaraproject/{canonical}"
            elif corpus == "tmforum" and canonical:
                deep_link = f"https://github.com/tmforum-rand/tmf-otel"

            return portal, deep_link

        def _fetch_chunk_text(chunk_id: str, corpus: str, db_path_str: str,
                               model_key_str: str) -> str | None:
            if not chunk_id:
                return None
            try:
                import chromadb
                client     = chromadb.PersistentClient(path=db_path_str)
                collection = client.get_collection(f"{corpus}_{model_key_str}")
                result     = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
                if result["documents"]:
                    return result["documents"][0]
            except Exception:
                pass
            return None

        # ── Dropdown / File picker for Drafts ──────────────────────────────
        draft_dir = _Path2("draft_questions")
        available_drafts = [str(p) for p in sorted(draft_dir.glob("*.json"))] if draft_dir.exists() else []
        default_draft = st.session_state.get("last_draft_path", "")

        if available_drafts:
            # Auto-select the most recently generated draft if available
            default_idx = available_drafts.index(default_draft) if default_draft in available_drafts else 0
            draft_options = available_drafts + ["Custom path..."]
            
            selected_draft = st.selectbox(
                "Select Draft QnA set (.json)",
                options=draft_options,
                index=default_idx,
                key="review_draft_select",
                help="Select an unapproved draft QnA file from draft_questions/"
            )
            if selected_draft == "Custom path...":
                review_path_str = st.text_input("Custom Draft QnA path", value=default_draft, key="review_custom_path")
            else:
                review_path_str = selected_draft
        else:
            review_path_str = st.text_input("Draft QnA file (.json)", value=default_draft,
                                            placeholder="draft_questions/draft_oran.json",
                                            key="review_path_fallback")

        col_rv1, col_rv2 = st.columns(2)
        with col_rv1:
            review_db_path = st.text_input("ChromaDB path (for chunk lookup)",
                                            value="chroma_db", key="review_db_path")
        with col_rv2:
            review_embedder = st.selectbox("Embedder (collection)",
                                            options=list(_load_embedder_keys().keys()),
                                            format_func=lambda k: _load_embedder_keys()[k],
                                            key="review_embedder")

        if review_path_str:
            review_path = _Path2(review_path_str)
            if not review_path.exists():
                st.error(f"File not found: {review_path}")
            else:
                data      = _json2.loads(review_path.read_text(encoding="utf-8"))
                questions = data.get("questions", [])
                st.info(f"{len(questions)} questions loaded from {review_path.name}")

                if ("review_state" not in st.session_state or
                        st.session_state.get("review_file") != str(review_path)):
                    st.session_state["review_state"] = {
                        i: {"status": "pending", "answer": q.get("reference_answer", "") or ""}
                        for i, q in enumerate(questions)
                    }
                    st.session_state["review_file"] = str(review_path)

                for i, q in enumerate(questions):
                    fmt        = q.get("answer_type", "open")
                    difficulty = q.get("difficulty", "unspecified")
                    val_passed = q.get("validation_passed")
                    val_note   = q.get("validation_note", "")
                    chunk_id   = q.get("chunk_id", "")
                    corpus_q   = q.get("corpus", source_type)
                    state      = st.session_state["review_state"][i]

                    val_badge = (" ✅ validated" if val_passed is True
                                 else " ⚠️ validation flagged" if val_passed is False else "")
                    header = f"Q{i+1} · [{fmt} / {difficulty}]{val_badge} · {q['question'][:60]}..."

                    with st.expander(header):
                        st.markdown(f"**Question:** {q['question']}")

                        # ── Source reference row ──────────────────────────
                        source_doc = q.get("source_document", "—")
                        canonical  = q.get("canonical_source", "—")
                        chunk_idx  = chunk_id.split("::")[-1] if "::" in chunk_id else "—"

                        portal_url, deep_link = _build_doc_url(q)

                        col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
                        with col_s1:
                            st.markdown(
                                f"<div style='font-size:0.78rem; color:#888; margin-bottom:4px;'>"
                                f"📄 <b>{source_doc}</b> · chunk {chunk_idx} · {canonical}"
                                f"</div>", unsafe_allow_html=True,
                            )
                        with col_s2:
                            if portal_url:
                                st.link_button("🌐 Open portal", portal_url,
                                               width="stretch")
                        with col_s3:
                            if deep_link and deep_link != portal_url:
                                st.link_button("📄 Open doc", deep_link,
                                               width="stretch")

                        # ── Source chunk text ─────────────────────────────
                        if chunk_id:
                            with st.expander("📖 View source chunk text", expanded=False):
                                chunk_text = _fetch_chunk_text(
                                    chunk_id, corpus_q, review_db_path, review_embedder,
                                )
                                if chunk_text:
                                    st.markdown(
                                        f"<div style='background:#f8f8f8; border:1px solid #e0e0e0; "
                                        f"border-left:3px solid #1a6bbf; border-radius:6px; "
                                        f"padding:0.75rem 1rem; font-size:0.82rem; "
                                        f"line-height:1.6; white-space:pre-wrap; "
                                        f"font-family:monospace;'>{chunk_text[:3000]}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(f"chunk_id: {chunk_id}")
                                else:
                                    st.caption("Chunk text not available.")

                        if val_passed is False:
                            st.warning(f"Validation flagged: {val_note}")
                        elif val_passed is True and val_note:
                            st.caption(f"Validation note: {val_note}")

                        if fmt == "mc":
                            st.markdown("**Options:**")
                            for letter, text in sorted(q.get("options", {}).items()):
                                marker = " ← draft answer" if letter == q.get("reference_answer") else ""
                                st.markdown(f"- {letter}: {text}{marker}")
                        elif fmt == "tf":
                            st.markdown(f"**Draft explanation:** {q.get('explanation', '')}")

                        edited_answer = st.text_area("Reference answer (edit if needed)",
                            value=state["answer"], key=f"answer_{i}", height=80)
                        state["answer"] = edited_answer

                        col_r1, col_r2, col_r3 = st.columns(3)
                        if col_r1.button("✅ Approve",        key=f"approve_{i}"): state["status"] = "approved"
                        if col_r2.button("✏️ Mark corrected", key=f"correct_{i}"): state["status"] = "corrected"
                        if col_r3.button("❌ Reject",          key=f"reject_{i}"):  state["status"] = "rejected"
                        st.caption(f"Status: **{state['status']}**")

                st.markdown("---")
                approved_count = sum(1 for s in st.session_state["review_state"].values()
                                     if s["status"] in ("approved", "corrected"))
                rejected_count = sum(1 for s in st.session_state["review_state"].values()
                                     if s["status"] == "rejected")
                pending_count  = sum(1 for s in st.session_state["review_state"].values()
                                     if s["status"] == "pending")
                st.markdown(f"**{approved_count} approved/corrected · "
                            f"{rejected_count} rejected · {pending_count} pending**")

                if st.button("💾 Merge approved into eval set", type="primary", key="merge_btn"):
                    draft_name = review_path.name
                    target_name = draft_name[6:] if draft_name.startswith("draft_") else draft_name
                    
                    eval_path = _Path2("configs/eval_questions") / target_name
                    eval_path.parent.mkdir(parents=True, exist_ok=True)
                    existing = {"questions": []}
                    if eval_path.exists():
                        existing = _json2.loads(eval_path.read_text(encoding="utf-8"))
                    to_add = []
                    for i, q in enumerate(questions):
                        state = st.session_state["review_state"][i]
                        if state["status"] in ("approved", "corrected"):
                            q2 = dict(q)
                            q2["reference_answer"] = state["answer"]
                            q2["validated"] = True
                            q2["generated"] = state["status"] == "corrected"
                            to_add.append(q2)
                    existing["questions"].extend(to_add)
                    eval_path.write_text(_json2.dumps(existing, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
                    st.success(f"Merged {len(to_add)} questions into {eval_path}")

    # ── QNA INFO ─────────────────────────────────────────────────────────────
    with tab_qna_info:
        st.caption("Overview of generated and validated QnA sets across directories.")
        if st.button("Refresh", key="qna_info_refresh"):
            st.rerun()

        import json as _json_info
        from pathlib import Path as _Path_info

        def _render_qna_directory(dir_path_str: str, title: str, is_draft: bool = False):
            dir_path = _Path_info(dir_path_str)
            st.markdown(f"#### {title}")
            
            if not dir_path.exists() or not dir_path.is_dir():
                st.info(f"Directory `{dir_path_str}` does not exist yet.")
                return

            files = list(dir_path.glob("*.json"))
            if not files:
                st.info(f"No QnA sets found in `{dir_path_str}`.")
                return

            total_q = 0
            for f in sorted(files):
                try:
                    data = _json_info.loads(f.read_text(encoding="utf-8"))
                    qs = data.get("questions", [])
                    num_qs = len(qs)
                    total_q += num_qs
                    val_qs = sum(1 for q in qs if q.get("validated") is True)

                    # Toggle styling based on set status
                    badge = "Draft" if is_draft else "Validated"
                    badge_style = "background:#fff8e1;color:#8a6200;border-color:#f0d080;" if is_draft else "background:#e8f5e9;color:#2e7d32;border-color:#a5d6a7;"
                    
                    st.markdown(
                        f"<div class='col-row'>"
                        f"<span class='col-name'>📄 {f.name} "
                        f"<span class='model-tag' style='{badge_style}'>{badge}</span></span>"
                        f"<span class='col-count'>{val_qs}/{num_qs} validated</span></div>",
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.markdown(
                        f"<div class='col-row'>"
                        f"<span class='col-name'>📄 {f.name}</span>"
                        f"<span class='col-count' style='color:#e74c3c;'>Error reading file</span></div>",
                        unsafe_allow_html=True
                    )
            st.caption(f"**Total questions across {len(files)} sets:** {total_q}")

        _render_qna_directory("configs/eval_questions", "Validated QnA Sets")
        st.markdown("---")
        
        # Fallback directory if genqa_output_dir is out of scope 
        target_draft_dir = genqa_output_dir if "genqa_output_dir" in locals() else "draft_questions"
        _render_qna_directory(target_draft_dir, "Draft QnA Sets", is_draft=True)