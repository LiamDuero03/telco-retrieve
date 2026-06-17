"""Streamlit web UI for TelcoLens."""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TelcoLens",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        background: #f8f8f8; border: 1px solid #e8e8e8;
        border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.5rem;
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
        background: #f5f5f5; border-radius: 8px; margin-bottom: 5px; font-size: 0.85rem;
    }
    .col-name { font-weight: 500; }
    .col-count { color: #888; }
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


@st.cache_data(show_spinner=False)
def _load_embedder_keys() -> dict:
    """Load embedder keys and descriptions from configs/embedding_models.json."""
    import json
    from pathlib import Path
    path = Path("configs/embedding_models.json")
    if not path.exists():
        return {"minilm": "minilm (default)"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: f"{k} — {v.get('description', v['model_name'])}" for k, v in data.items()}


def _get_store(source_type: str, model_key: str, db_path: str,
               collection: str | None = None):
    from open_telco_rag.stores.chroma import ChromaStore
    col = collection.strip() if collection and collection.strip() else f"{source_type}_{model_key}"
    return ChromaStore(path=db_path, collection_name=col)


def _similarity(distance: float) -> float:
    import math
    return round(math.exp(-distance), 3)


def _build_generator(llm_model: str):
    """All models go through OpenRouter."""
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

SOURCE_TYPES = ["oran", "3gpp", "camara", "tmforum", "gsma", "itu", "etsi"]
source_type = st.sidebar.selectbox("Corpus", SOURCE_TYPES)

_embedder_options = _load_embedder_keys()
model_key = st.sidebar.selectbox(
    "Embedder",
    options=list(_embedder_options.keys()),
    format_func=lambda k: _embedder_options[k],
)

LLM_OPTIONS = {
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Claude Sonnet 4":   "anthropic/claude-sonnet-4",
    "Claude Haiku 4.5":  "anthropic/claude-haiku-4-5",
    "GPT-4o":            "openai/gpt-4o",
    "GPT-4o mini":       "openai/gpt-4o-mini",
    "Llama 3.3 70B":     "meta-llama/llama-3.3-70b-instruct",
    "Llama 3.1 8B":      "meta-llama/llama-3.1-8b-instruct",
    "Qwen 2.5 7B":       "qwen/qwen-2.5-7b-instruct",
    "Gemma 3n E4B":      "google/gemma-3n-e4b-it",
    "Mistral Large":     "mistralai/mistral-large",
}
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

# ── Cookbook ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("**📚 Cookbook**")
st.sidebar.markdown(
    "[TelcoLens Cookbook](https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf)",
    unsafe_allow_html=True,
)

st.sidebar.markdown("**▶ Try it yourself**")

NOTEBOOKS = [
    ("O-RAN",     "01_oran_quickstart.ipynb"),
    ("3GPP",      "02_3gpp_quickstart.ipynb"),
    ("CAMARA",    "03_camara_quickstart.ipynb"),
    ("TM Forum",  "04_tmforum_quickstart.ipynb"),
    ("GSMA",      "05_gsma_quickstart.ipynb"),
    ("ITU-T",     "06_itu_quickstart.ipynb"),
    ("ETSI",      "07_etsi_quickstart.ipynb"),
]

NB_BASE = "https://github.com/Znbne/telcolens/blob/main/notebooks"

for name, fname in NOTEBOOKS:
    if fname:
        st.sidebar.markdown(f"[{name}]({NB_BASE}/{fname})", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<span style='color:#aaa'>{name}</span>", unsafe_allow_html=True)



# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_home, tab_ask, tab_search, tab_ingest, tab_eval, tab_lb, tab_info = st.tabs(
    ["Home", "Ask", "Search", "Ingest", "Evaluate", "Leaderboard", "DB info"]
)


# ══════════════════════════════════════════════════════════════════════════════
# HOME tab
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── Corpus stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Indexed corpora")

    CORPUS_STATS = [
        ("O-RAN",     "oran",    "📡", "O-RAN Alliance R003–R005",        169130),
        ("3GPP",      "3gpp",    "📶", "Release 8 through 19",             1642543),
        ("ETSI",      "etsi",    "🇪🇺", "ENI, ZSM, MEC, NFV, SAI, F5G",   984728),
        ("ITU-T",     "itu",     "🌐", "All 27 active series",             575762),
        ("TM Forum",  "tmforum", "🔧", "TMF Open APIs, ODA",              6905),
        ("CAMARA",    "camara",  "🔌", "Fall25 — 75+ API repos",          3919),
        ("GSMA",      "gsma",    "📱", "Open Gateway & security series",  121118),
    ]
    CARD_BG     = "#1a1a1a"
    CARD_BORDER = "#cc0000"

    total_chunks = sum(s[4] for s in CORPUS_STATS)

    # Top metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Standards bodies", "7")
    m2.metric("Total published chunks", "3,504,105")
    m3.metric("Embedding models", "6")

    st.markdown("")

    # Corpus cards — 2 rows
    cols1 = st.columns(4)
    cols2 = st.columns(4)
    all_cols = cols1 + cols2 + [None]  # 7 corpora + 1 empty
    for col, (name, key, icon, desc, chunks) in zip(all_cols, CORPUS_STATS):
        bg, border = CARD_BG, CARD_BORDER
        if col is None:
            continue
        with col:
            chunk_str = f"{chunks:,} chunks" if chunks > 0 else "🔜 coming soon"
            st.markdown(
                f"<div style='background:{bg}; border:1px solid {border}; border-radius:10px;"
                f"padding:14px 14px 12px; text-align:center; min-height:120px;'>"
                f"<div style='font-size:1.6rem;'>{icon}</div>"
                f"<div style='font-weight:600; font-size:0.9rem; margin-top:4px; color:#ffffff;'>{name}</div>"
                f"<div style='font-size:0.75rem; color:#cccccc; margin-top:2px; line-height:1.3;'>{desc}</div>"
                f"<div style='font-size:0.72rem; color:{border}; margin-top:6px; font-family:monospace;'>{chunk_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Quick start ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Get started")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div style='background:#f0f7ff;border:1px solid #cce0ff;border-radius:10px;padding:14px;'>"
            "<div style='font-weight:600; margin-bottom:6px;'>1. Select corpus & model</div>"
            "<div style='font-size:0.82rem;color:#555;'>Use the sidebar to choose a corpus, "
            "embedder, LLM, and DB path. Set the collection override if using a custom ingestion.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='background:#f0fff4;border:1px solid #b2f0c8;border-radius:10px;padding:14px;'>"
            "<div style='font-weight:600; margin-bottom:6px;'>2. Ask a question</div>"
            "<div style='font-size:0.82rem;color:#555;'>Go to the <b>Ask</b> tab and type any "
            "technical question. The pipeline retrieves relevant chunks and generates a grounded answer.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div style='background:#fff8f0;border:1px solid #ffdcb2;border-radius:10px;padding:14px;'>"
            "<div style='font-weight:600; margin-bottom:6px;'>3. Evaluate & compare</div>"
            "<div style='font-size:0.82rem;color:#555;'>Use the <b>Evaluate</b> tab to score RAG quality "
            "across 5 metrics. Results appear on the <b>Leaderboard</b> automatically.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Links ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.82rem; color:#888;'>"
        "📄 <a href='https://github.com/Znbne/telcolens-cookbook/blob/master/cookbook.pdf' target='_blank'>Cookbook</a> &nbsp;·&nbsp; "
        "💻 <a href='https://github.com/Znbne/telcolens' target='_blank'>GitHub</a> &nbsp;·&nbsp; 🤗 <a href='https://huggingface.co/datasets/GSMA/telcolens-chunks' target='_blank'>HuggingFace Dataset</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ASK tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.caption("Retrieval + generation — asks the LLM to answer using retrieved chunks.")
    question = st.text_input(
        "Question",
        placeholder="What are the xApp conflict mitigation strategies in O-RAN?",
        label_visibility="collapsed",
    )
    col_ask1, col_ask2 = st.columns(2)
    with col_ask1:
        top_k_ask = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=8)
    with col_ask2:
        threshold_ask = st.slider("Similarity threshold", min_value=0.0, max_value=1.0,
                                  value=0.0, step=0.05,
                                  help="Minimum similarity score (0 = no filtering)")
    ask_btn = st.button("Ask ✦", type="primary")

    if ask_btn and question.strip():
        with st.spinner("Retrieving and generating…"):
            try:
                from open_telco_rag.core.pipeline import generate, search

                embedder = _get_embedder(model_key)
                store    = _get_store(source_type, model_key, db_path, collection_override)
                results  = search(question, store=store, embedder=embedder,
                                  top_k=top_k_ask, threshold=1.0 - threshold_ask)
                answer   = generate(
                    question, results,
                    source_type = source_type,
                    generator   = _build_generator(llm_model),
                )

                # ── Confidence indicator ──────────────────────────────────
                import math as _math
                mean_sim = (
                    sum(_math.exp(-r.distance) for r in results) / len(results)
                    if results else 0.0
                )
                if mean_sim >= 0.65:
                    conf_color, conf_label = "#2ecc71", "High confidence"
                elif mean_sim >= 0.45:
                    conf_color, conf_label = "#f39c12", "Medium confidence"
                else:
                    conf_color, conf_label = "#e74c3c", "Low confidence"

                st.markdown(
                    f"<div style='font-size:0.75rem;color:#888;margin-bottom:6px;"
                    f"display:flex;align-items:center;gap:8px;'>"
                    f"<span style='width:8px;height:8px;border-radius:50%;"
                    f"background:{conf_color};display:inline-block;'></span>"
                    f"<span>{conf_label} · sim {mean_sim:.3f} · "
                    f"{answer['chunks_used']} chunks</span>"
                    f"<span class='model-tag'>{llm_label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Answer ───────────────────────────────────────────────
                st.markdown(
                    f"<div class='answer-box'>{answer['answer']}</div>",
                    unsafe_allow_html=True,
                )

                if answer.get("sources"):
                    chips = "".join(
                        f"<span class='source-chip'>📄 {s}</span>"
                        for s in answer["sources"]
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                if results:
                    with st.expander(f"Retrieved chunks ({len(results)})"):
                        for i, r in enumerate(results, 1):
                            source  = r.metadata.get("source", "—")
                            heading = r.metadata.get("heading", "")
                            sim     = _similarity(r.distance)
                            st.markdown(
                                f"**{i}. {source}**"
                                + (f" — {heading}" if heading else "")
                                + f"  `sim {sim}`"
                            )
                            st.text(r.text)
                            st.divider()

            except Exception as e:
                st.error(f"Error: {e}")

    elif ask_btn:
        st.warning("Please enter a question.")


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.caption("Pure vector retrieval — returns the most relevant chunks, no LLM involved.")
    query = st.text_input(
        "Query",
        placeholder="near-RT RIC E2 interface latency requirements",
        label_visibility="collapsed",
        key="search_query",
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        top_k_search = st.number_input("Top K", min_value=1, max_value=20, value=5, key="sk_topk")
    with col_s2:
        threshold_search = st.slider("Similarity threshold", min_value=0.0, max_value=1.0,
                                     value=0.0, step=0.05, key="sk_threshold",
                                     help="Minimum similarity score (0 = no filtering)")
    search_btn = st.button("Search", type="primary", key="search_btn")

    if search_btn and query.strip():
        with st.spinner("Searching…"):
            try:
                from open_telco_rag.core.pipeline import search

                embedder = _get_embedder(model_key)
                store    = _get_store(source_type, model_key, db_path, collection_override)
                results  = search(query, store=store, embedder=embedder,
                                  top_k=top_k_search, threshold=1.0 - threshold_search)

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
                            st.markdown(
                                f"**{i}. {source}**"
                                + (f" — {heading}" if heading else "")
                                + f"  `sim {sim}`"
                            )
                            st.text(r.text)
                            st.divider()

            except Exception as e:
                st.error(f"Error: {e}")

    elif search_btn:
        st.warning("Please enter a query.")


# ══════════════════════════════════════════════════════════════════════════════
# INGEST tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        data_dir   = st.text_input("Data directory", value=f"data/{source_type}")
        chunk_size = st.number_input("Chunk size", min_value=128, max_value=4096,
                                     value=1024, step=128)
    with col_i2:
        ingest_collection = st.text_input("Collection name",
                                          placeholder=f"{source_type}_{model_key}")
        overlap = st.number_input("Overlap", min_value=0, max_value=512,
                                  value=200, step=32)

    col_i3, col_i4 = st.columns(2)
    with col_i3:
        use_marker = st.checkbox("Use Marker API (better PDF quality)")
    with col_i4:
        no_resume  = st.checkbox("Disable resume (reprocess all files)")

    ingest_btn = st.button("Run ingest ▶", type="primary")

    if ingest_btn:
        col_name  = ingest_collection.strip() or f"{source_type}_{model_key}"
        log_lines = [
            f"Ingesting {source_type.upper()} documents",
            f"  Data dir:   {data_dir}",
            f"  Model:      {model_key}",
            f"  Collection: {col_name}",
            f"  PDF loader: {'Marker API' if use_marker else 'pypdf (default)'}",
            "",
        ]
        log_box = st.empty()
        log_box.markdown(
            "<div class='log-box'>" + "\n".join(log_lines) + "Running…</div>",
            unsafe_allow_html=True,
        )

        try:
            from open_telco_rag.core.pipeline import ingest
            from open_telco_rag.stores.chroma import ChromaStore

            src      = _get_source(source_type, use_marker=use_marker)
            embedder = _get_embedder(model_key)
            store    = ChromaStore(path=db_path, collection_name=col_name)

            stats = ingest(
                source=src, store=store, embedder=embedder,
                data_dir=data_dir, chunk_size=chunk_size,
                overlap=overlap, resume=not no_resume,
            )

            log_lines += [
                f"Files processed: {stats.files_processed}",
                f"Chunks stored:   {stats.chunks_stored}",
                f"Skipped:         {stats.skipped_files}",
                f"Errors:          {stats.error_files}",
            ]
            log_box.markdown(
                "<div class='log-box'>" + "\n".join(log_lines) + "</div>",
                unsafe_allow_html=True,
            )
            st.success(
                f"Done — {stats.chunks_stored:,} chunks stored, "
                f"{stats.skipped_files} skipped, {stats.error_files} errors."
            )

        except Exception as e:
            log_lines.append(f"ERROR: {e}")
            log_box.markdown(
                "<div class='log-box'>" + "\n".join(log_lines) + "</div>",
                unsafe_allow_html=True,
            )
            st.error(f"Ingest failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE tab
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.caption(f"RAG generation uses the sidebar LLM ({llm_label}). Metrics scoring uses the model selected below.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        questions_file = st.text_input(
            "Questions file (.json)",
            value="configs/eval_questions/3gpp_r19_qna.json",
            help="Path to eval questions JSON file",
        )
        top_k_eval = st.number_input("Top K", min_value=1, max_value=20, value=5,
                                     key="eval_topk")
        output_dir = st.text_input("Output directory", value="results/gui",
                                   help="Where to save the result JSON")

    with col_e2:
        METRICS_LLM_OPTIONS = {
            "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
            "Claude Sonnet 4":   "anthropic/claude-sonnet-4",
            "GPT-4o":            "openai/gpt-4o",
            "Llama 3.3 70B":     "meta-llama/llama-3.3-70b-instruct",
        }
        metrics_llm_label = st.selectbox("Metrics evaluation model",
                                         list(METRICS_LLM_OPTIONS.keys()))
        eval_delay = st.slider("Delay between questions (s)", min_value=0.0,
                               max_value=5.0, value=1.0, step=0.5)

    rag_llm_model     = llm_model   # use sidebar LLM for RAG generation
    metrics_llm_model = METRICS_LLM_OPTIONS[metrics_llm_label]

    include_baseline = st.checkbox(
        "Include baseline (no retrieval)",
        value=False,
        help="Runs a second pass with no context chunks — answers from LLM knowledge only. "
             "Baseline metrics are stored alongside RAG metrics in the leaderboard."
    )
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

                st.info(
                    f"Running evaluation on **{n} questions** · "
                    f"collection `{col_name}` · "
                    f"RAG: {llm_label} · "
                    f"Metrics: {metrics_llm_label}"
                )

                from open_telco_rag.evaluation.evaluate import evaluate_corpus
                from open_telco_rag.evaluation.questions import EvalQuestion
                from open_telco_rag.evaluation._claude import make_llm_call

                embedder     = _get_embedder(model_key)
                store        = _get_store(source_type, model_key, db_path, collection_override)
                questions    = [EvalQuestion.from_dict(q) for q in qs]
                llm_call     = make_llm_call("openrouter", metrics_llm_model)
                rag_llm_call = make_llm_call("openrouter", rag_llm_model)

                # ── Live progress ─────────────────────────────────────────────
                progress_bar = st.progress(0, text="Starting…")
                log_area     = st.empty()
                log_lines    = []

                # Monkey-patch evaluate_question to stream progress to UI
                import open_telco_rag.evaluation.evaluate as _eval_mod
                _orig_eq = _eval_mod.evaluate_question

                def _patched_eq(q, *args, **kwargs):
                    qr = _orig_eq(q, *args, **kwargs)
                    idx = questions.index(q) + 1
                    progress_bar.progress(idx / n, text=f"Question {idx}/{n}")
                    if qr.error:
                        log_lines.append(f"[{idx}/{n}] ERROR: {qr.error}")
                    else:
                        log_lines.append(
                            f"[{idx}/{n}] faithful={qr.faithfulness:.2f}  "
                            f"relevancy={qr.answer_relevancy:.2f}  "
                            f"sim={qr.mean_similarity:.3f}  "
                            f"recall={qr.context_recall:.2f}  "
                            f"correct={qr.answer_correctness:.2f}"
                        )
                    log_area.markdown(
                        "<div class='log-box'>" + "\n".join(log_lines[-20:]) + "</div>",
                        unsafe_allow_html=True,
                    )
                    return qr

                _eval_mod.evaluate_question = _patched_eq

                result, baseline_result = evaluate_corpus(
                    corpus            = source_type,
                    store             = store,
                    embedder          = embedder,
                    questions         = questions,
                    model             = model_key,
                    top_k             = top_k_eval,
                    delay             = eval_delay,
                    output_dir        = Path(output_dir),
                    llm_call          = llm_call,
                    rag_llm_call      = rag_llm_call,
                    include_baseline  = include_baseline,
                )

                _eval_mod.evaluate_question = _orig_eq  # restore
                progress_bar.progress(1.0, text="Done ✓")

                # ── Update leaderboard ────────────────────────────────────────
                scores = [
                    result.mean_faithfulness,
                    result.mean_answer_relevancy,
                    result.mean_similarity,
                    result.mean_context_recall,
                    result.mean_answer_correctness,
                ]
                valid = [s for s in scores if s >= 0]
                avg   = round(sum(valid) / len(valid), 3) if valid else -1.0
                col_name = collection_override.strip() or f"{source_type}_{model_key}"

                def _bl(v):
                    return round(max(0, v), 3) if v is not None and v >= 0 else None

                lb_row = {
                    "timestamp":             __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "rag_model":             rag_llm_model,
                    "evaluator_model":       metrics_llm_model,
                    "embedder":              model_key,
                    "corpus":               source_type,
                    "collection":            col_name,
                    "top_k":                 top_k_eval,
                    "n_questions":           result.n_questions,
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
                lb_path.write_text(
                    json.dumps(lb_rows, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # ── Summary table + charts ───────────────────────────────
                st.markdown("---")
                st.markdown(f"**Results — {result.corpus.upper()} · "
                            f"model={result.model} · top-k={result.top_k} · "
                            f"n={result.n_questions}**")

                rows = [
                    ("Faithfulness",          result.mean_faithfulness,       "Generation"),
                    ("Answer relevancy",       result.mean_answer_relevancy,   "Generation"),
                    ("Mean similarity",        result.mean_similarity,         "Retrieval"),
                    ("Source coverage (mean)", result.mean_source_coverage,    "Retrieval"),
                    ("Context recall",         result.mean_context_recall,     "Retrieval+GT"),
                    ("Answer correctness",     result.mean_answer_correctness, "Generation+GT"),
                ]

                # Summary table
                st.markdown(
                    "<div class='eval-row eval-row-header'>"
                    "<span class='eval-metric'>Metric</span>"
                    "<span class='eval-score'>Score</span>"
                    "<span class='eval-type'>Type</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                for label, value, kind in rows:
                    score_str = "N/A" if value < 0 else (
                        f"{value:.1f}" if "coverage" in label.lower() else f"{value:.3f}"
                    )
                    st.markdown(
                        f"<div class='eval-row eval-row-data'>"
                        f"<span class='eval-metric'>{label}</span>"
                        f"<span class='eval-score'>{score_str}</span>"
                        f"<span class='eval-type'>{kind}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Charts — one per metric ───────────────────────────────
                st.markdown("---")
                import plotly.graph_objects as go

                qrs     = result.question_results
                qlabels = [f"Q{i+1}" for i in range(len(qrs))]

                def _metric_chart(title, values, color, mean_val):
                    """Bar chart for one metric across all questions."""
                    clean = [max(0.0, v) if v >= 0 else 0.0 for v in values]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=qlabels, y=clean,
                        marker_color=color,
                        text=[f"{v:.2f}" for v in clean],
                        textposition="outside",
                        name=title,
                    ))
                    # Mean line
                    if mean_val >= 0:
                        fig.add_hline(
                            y=mean_val, line_dash="dash",
                            line_color="#555", line_width=1.5,
                            annotation_text=f"avg {mean_val:.3f}",
                            annotation_position="top right",
                            annotation_font_size=11,
                        )
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=13)),
                        yaxis=dict(range=[0, 1.15], gridcolor="#eee",
                                   tickformat=".2f"),
                        xaxis=dict(tickangle=-45 if len(qrs) > 20 else 0),
                        height=300,
                        margin=dict(t=45, b=50, l=30, r=20),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(size=11),
                        showlegend=False,
                    )
                    return fig

                metrics_to_plot = [
                    ("Faithfulness",       [q.faithfulness      for q in qrs], "#4C9BE8", result.mean_faithfulness),
                    ("Answer relevancy",   [q.answer_relevancy  for q in qrs], "#82C5A0", result.mean_answer_relevancy),
                    ("Mean similarity",    [q.mean_similarity   for q in qrs], "#F5A623", result.mean_similarity),
                    ("Context recall",     [q.context_recall    for q in qrs], "#9B59B6", result.mean_context_recall),
                    ("Answer correctness", [q.answer_correctness for q in qrs],"#E74C3C", result.mean_answer_correctness),
                ]

                # Layout: 3 on top row, 2 on bottom row
                r1c1, r1c2, r1c3 = st.columns(3)
                r2c1, r2c2, _    = st.columns([1, 1, 1])

                cols = [r1c1, r1c2, r1c3, r2c1, r2c2]
                for col, (title, values, color, mean_val) in zip(cols, metrics_to_plot):
                    with col:
                        st.plotly_chart(
                            _metric_chart(title, values, color, mean_val),
                            use_container_width=True,
                        )

                st.success(f"Results saved → {output_dir}/")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD tab
# ══════════════════════════════════════════════════════════════════════════════
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
        st.info("No leaderboard data yet. Run an evaluation with `open-telco-rag evaluate` "
                "or use the Evaluate tab — results will appear here automatically.")
    else:
        try:
            rows = _json.loads(lb_path.read_text(encoding="utf-8"))
            if not rows:
                st.info("Leaderboard file is empty. Run an evaluation to populate it.")
            else:
                import pandas as pd
                import pandas as _pd
                import plotly.graph_objects as go

                df = pd.DataFrame(rows)

                # ── Filters ───────────────────────────────────────────────────
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    collection_options = ["All"] + sorted(df["collection"].unique().tolist())
                    filter_collection  = st.selectbox("Collection", collection_options, key="lb_collection")
                with col_f2:
                    embedder_options = ["All"] + sorted(df["embedder"].unique().tolist())
                    filter_embedder  = st.selectbox("Embedder", embedder_options, key="lb_embedder")
                with col_f3:
                    model_options = ["All"] + sorted(df["rag_model"].unique().tolist())
                    filter_model  = st.selectbox("Generator", model_options, key="lb_model")

                if filter_collection != "All": df = df[df["collection"] == filter_collection]
                if filter_embedder   != "All": df = df[df["embedder"]   == filter_embedder]
                if filter_model      != "All": df = df[df["rag_model"]  == filter_model]

                if df.empty:
                    st.info("No results match the selected filters.")
                else:
                    # Sort by average descending
                    df = df.sort_values("average", ascending=False).reset_index(drop=True)
                    df.insert(0, "rank", range(1, len(df) + 1))

                    # ── Compute delta columns where baseline exists ────────────
                    has_baseline = "bl_average" in df.columns and df["bl_average"].notna().any()
                    if has_baseline:
                        df["Δ correctness"] = (df["answer_correctness"] - df["bl_answer_correctness"]).round(3)
                        df["Δ recall"]      = (df["context_recall"]     - df["bl_context_recall"]).round(3)
                        df["Δ avg"]         = (df["average"]            - df["bl_average"]).round(3)

                    # ── Table ─────────────────────────────────────────────────
                    display_cols = {
                        "rank":               "Rank",
                        "rag_model":          "Generator",
                        "evaluator_model":    "Evaluator",
                        "embedder":           "Embedder",
                        "reranker":           "Reranker",
                        "collection":         "Collection",
                        "faithfulness":       "Faithful",
                        "answer_relevancy":   "Relevancy",
                        "mean_similarity":    "Similarity",
                        "context_recall":     "Recall",
                        "answer_correctness": "Correctness",
                        "average":            "RAG Avg ↓",
                    }
                    if has_baseline:
                        display_cols.update({
                            "bl_answer_correctness": "BL Correctness",
                            "bl_context_recall":     "BL Recall",
                            "bl_average":            "BL Avg",
                            "Δ correctness":         "Δ Correctness",
                            "Δ recall":              "Δ Recall",
                            "Δ avg":                 "Δ Avg",
                        })

                    df_display = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
                    col_config = {
                        "RAG Avg ↓":      st.column_config.NumberColumn(format="%.3f"),
                        "Faithful":       st.column_config.NumberColumn(format="%.3f"),
                        "Relevancy":      st.column_config.NumberColumn(format="%.3f"),
                        "Similarity":     st.column_config.NumberColumn(format="%.3f"),
                        "Recall":         st.column_config.NumberColumn(format="%.3f"),
                        "Correctness":    st.column_config.NumberColumn(format="%.3f"),
                        "BL Correctness": st.column_config.NumberColumn(format="%.3f"),
                        "BL Recall":      st.column_config.NumberColumn(format="%.3f"),
                        "BL Avg":         st.column_config.NumberColumn(format="%.3f"),
                        "Δ Correctness":  st.column_config.NumberColumn(format="%+.3f"),
                        "Δ Recall":       st.column_config.NumberColumn(format="%+.3f"),
                        "Δ Avg":          st.column_config.NumberColumn(format="%+.3f"),
                    }
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config=col_config,
                    )

                    # ── Delete a row ──────────────────────────────────────────
                    with st.expander("🗑 Remove a row"):
                        # Build a readable label for each row using original (unfiltered) indices
                        all_rows = _json.loads(lb_path.read_text(encoding="utf-8"))
                        all_df   = _pd.DataFrame(all_rows)

                        # Match filtered df rows back to original indices
                        row_labels = {
                            i: (
                                f"{r.get('timestamp','?')}  ·  "
                                f"{r.get('rag_model','?')}  ·  "
                                f"{r.get('embedder','?')}  ·  "
                                f"{r.get('corpus','?')}  ·  "
                                f"avg={r.get('average','?')}"
                            )
                            for i, r in enumerate(all_rows)
                        }
                        selected_label = st.selectbox(
                            "Select row to remove",
                            options=list(row_labels.keys()),
                            format_func=lambda i: row_labels[i],
                            key="lb_delete_select",
                        )
                        if st.button("Remove selected row", type="primary", key="lb_delete_btn"):
                            all_rows.pop(selected_label)
                            lb_path.write_text(
                                _json.dumps(all_rows, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            st.success("Row removed.")
                            st.rerun()

                    # ── Bar chart — average by generator ─────────────────────
                    st.markdown("---")
                    avg_by_model = (
                        df.groupby("rag_model")["average"]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index()
                    )
                    fig = go.Figure(go.Bar(
                        x=avg_by_model["rag_model"],
                        y=avg_by_model["average"],
                        marker_color="#4C9BE8",
                        text=[f"{v:.3f}" for v in avg_by_model["average"]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Average score by generator model",
                        yaxis=dict(range=[0, 1.15], gridcolor="#eee", tickformat=".2f"),
                        height=320,
                        margin=dict(t=45, b=60, l=30, r=20),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(size=11),
                        xaxis=dict(tickangle=-20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        f"Evaluator: {df['evaluator_model'].iloc[0]}  ·  "
                        f"{len(df)} run(s) shown  ·  "
                        f"leaderboard at `{lb_path}`"
                    )

        except Exception as e:
            st.error(f"Could not load leaderboard: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# DB INFO tab
# ══════════════════════════════════════════════════════════════════════════════
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
                st.markdown(
                    f"<div class='col-row'>"
                    f"<span class='col-name'>{col.name}</span>"
                    f"<span class='col-count'>{c.count():,} chunks</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No collections found in '{db_path}'. Run ingest first.")

    except Exception as e:
        st.error(f"Could not read database: {e}")
