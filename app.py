"""Streamlit Web Application for Graph RAG."""

import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

from graph_rag.engine import GraphRAG
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.llm.mock import MockLLM, MockEmbedding

# Page configuration
st.set_page_config(
    page_title="Graph RAG Studio",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Storage paths for persistent cache
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR = DATA_DIR / "active_index"
ENV_FILE = Path(__file__).parent / ".env"


def persist_api_key(key: str) -> None:
    """Save GROQ_API_KEY to .env so it automatically persists across browser reloads."""
    cleaned = key.strip().strip("'\"")
    if not cleaned or not cleaned.startswith("gsk_"):
        return
    try:
        lines = []
        if ENV_FILE.exists():
            lines = ENV_FILE.read_text().splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith("GROQ_API_KEY="):
                new_lines.append(f"GROQ_API_KEY={cleaned}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"GROQ_API_KEY={cleaned}")
        ENV_FILE.write_text("\n".join(new_lines) + "\n")
        os.environ["GROQ_API_KEY"] = cleaned
    except Exception:
        pass


def auto_tune_parameters(total_chars: int, doc_count: int = 1) -> tuple:
    """Calculate optimal chunk size, overlap, graph hops, and batch size based on document volume."""
    if total_chars <= 0:
        return 800, 120, 1, 3, "Default settings (800 chars / 120 overlap / 1 hop / 3 chunks per call)"
    
    if total_chars < 15000:
        return (
            650,
            120,
            2,
            2,
            f"Targeted Profile ({total_chars:,} chars): Set 650 Chunk Size, 120 Overlap, 2 Hops, Batch Size 2. Denser extraction preserves specific entity interactions in concise documents."
        )
    elif total_chars < 75000:
        return (
            900,
            160,
            2,
            3,
            f"Balanced Profile ({total_chars:,} chars, {doc_count} docs): Set 900 Chunk Size, 160 Overlap, 2 Hops, Batch Size 3. Balances entity extraction density with relational context."
        )
    else:
        return (
            1300,
            220,
            1,
            4,
            f"High-Volume Profile ({total_chars:,} chars, {doc_count} docs): Set 1300 Chunk Size, 220 Overlap, 1 Hop, Batch Size 4. Packs 4 chunks per call for 4x faster indexing and cross-chunk relationship discovery."
        )


def init_session():
    """Initialize session state variables."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph_html" not in st.session_state:
        st.session_state.graph_html = None
    if "highlight_nodes" not in st.session_state:
        st.session_state.highlight_nodes = []
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 800
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 120
    if "max_hops" not in st.session_state:
        st.session_state.max_hops = 1
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 3
    if "auto_tuned_msg" not in st.session_state:
        st.session_state.auto_tuned_msg = ""


init_session()

# ==========================================
# SIDEBAR: Configuration & API Keys
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Inference Provider")
    
    env_key = os.getenv("GROQ_API_KEY", "")
    has_env_key = bool(env_key.strip())

    provider_mode = st.radio(
        "Select Provider Mode:",
        options=["Groq Cloud (Free Llama 3.3)", "Offline / Mock Mode"],
        index=0,
        help="Groq Cloud uses free high-speed Llama 3.3 models. Mock mode runs offline without an API key."
    )
    
    use_offline_mock = (provider_mode == "Offline / Mock Mode")
    
    groq_api_key = ""
    if not use_offline_mock:
        if has_env_key:
            st.success("API Key loaded from environment", icon="🔑")
            groq_api_key = env_key
            override = st.text_input("Override Groq API Key:", value=env_key, type="password")
            if override.strip():
                groq_api_key = override.strip()
                persist_api_key(groq_api_key)
        else:
            raw_input = st.text_input(
                "🔑 Enter Groq API Key:",
                type="password",
                placeholder="gsk_...",
                help="Get a 100% free key at console.groq.com/keys",
            )
            groq_api_key = raw_input.strip().strip("'\"")
            if groq_api_key:
                persist_api_key(groq_api_key)
                if not groq_api_key.startswith("gsk_"):
                    st.warning("⚠️ Groq API keys usually start with 'gsk_'. Please ensure you copied the full secret key from the Groq popup.")
                else:
                    st.success("API key saved! Will persist on reload.", icon="💾")
            st.caption("👉 Get a free API key at [console.groq.com/keys](https://console.groq.com/keys)")

        # Fetch accessible models dynamically or use robust defaults
        available_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        if groq_api_key.strip():
            try:
                fetched = GroqLLM.get_available_models(groq_api_key.strip())
                if fetched:
                    available_models = fetched
            except Exception:
                pass

        model_option = st.selectbox(
            "Groq LLM Model:",
            options=available_models,
            index=0,
            help="High-speed, free reasoning & extraction models supported on your Groq account."
        )
    else:
        st.info("ℹ️ Running in Offline Mock mode. No API key needed.")
        model_option = "mock"

    st.markdown("---")
    st.markdown("### 📐 Pipeline Parameters")

    if st.button("🪄 Auto-Set Parameters", use_container_width=True, help="Automatically inspects staged text/files and sets the optimal chunk size, overlap, and graph hops for highest graph fidelity."):
        # Check active engine chunks
        eng = st.session_state.rag_engine
        chars = sum(len(c.text) for c in eng.chunks_map.values()) if eng else 0
        docs = len(set(c.doc_id for c in eng.chunks_map.values())) if eng else 0
        if chars == 0 and (STORAGE_DIR / "chunks.json").exists():
            try:
                import json
                with open(STORAGE_DIR / "chunks.json", "r") as f:
                    c_data = json.load(f)
                    chars = sum(len(c.get("text", "")) for c in c_data)
                    docs = len(set(c.get("doc_id", "doc") for c in c_data))
            except Exception:
                pass
        c_size, c_overlap, hops, b_size, reason = auto_tune_parameters(chars, max(docs, 1))
        st.session_state.chunk_size = c_size
        st.session_state.chunk_overlap = c_overlap
        st.session_state.max_hops = hops
        st.session_state.batch_size = b_size
        st.session_state.auto_tuned_msg = reason
        st.rerun()

    if st.session_state.get("auto_tuned_msg"):
        st.info(st.session_state.auto_tuned_msg, icon="✨")

    chunk_size = st.slider(
        "Chunk Size (characters):",
        min_value=300,
        max_value=2000,
        value=st.session_state.chunk_size,
        step=50,
        help="Target character size for text chunks. Smaller = denser entity extraction. Larger = broader context."
    )
    st.session_state.chunk_size = chunk_size

    chunk_overlap = st.slider(
        "Chunk Overlap (characters):",
        min_value=50,
        max_value=400,
        value=st.session_state.chunk_overlap,
        step=10,
        help="Characters shared between consecutive chunks to avoid breaking relationship sentences."
    )
    st.session_state.chunk_overlap = chunk_overlap

    max_hops = st.slider(
        "Local Search Graph Hops:",
        min_value=1,
        max_value=3,
        value=st.session_state.max_hops,
        help="Number of edge steps traversed around query entities during Local Search."
    )
    st.session_state.max_hops = max_hops

    batch_size = st.slider(
        "⚡ Batch Size (chunks / LLM call):",
        min_value=1,
        max_value=5,
        value=st.session_state.batch_size,
        help="Analyzes multiple contiguous chunks in one LLM call. 3-4x faster for large documents and discovers cross-chunk connections."
    )
    st.session_state.batch_size = batch_size

    # Persistence Status & Reset Controls
    if (STORAGE_DIR / "graph.json").exists():
        st.markdown("---")
        st.markdown("### 💾 Saved Index on Disk")
        st.success("Knowledge Graph is saved. Automatically reloaded on page refresh.", icon="💾")
        if st.button("🗑️ Delete Saved Index from Disk", use_container_width=True):
            import shutil
            if STORAGE_DIR.exists():
                shutil.rmtree(STORAGE_DIR)
            st.session_state.rag_engine = None
            st.session_state.is_indexed = False
            st.session_state.graph_html = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset Graph & Clear Session", use_container_width=True):
        st.session_state.rag_engine = None
        st.session_state.is_indexed = False
        st.session_state.chat_history = []
        st.session_state.graph_html = None
        st.session_state.highlight_nodes = []
        st.rerun()

    st.caption("Graph RAG v0.1.0 • 100% Free & Open-Source")


@st.cache_resource
def get_embedding_model() -> FastEmbedEmbedding:
    return FastEmbedEmbedding()


# Instantiate or update the engine
def get_engine() -> GraphRAG:
    current_mode_key = (use_offline_mock, groq_api_key, model_option)
    
    if st.session_state.rag_engine is None or st.session_state.get("last_mode_key") != current_mode_key:
        st.session_state.last_mode_key = current_mode_key
        embedding = get_embedding_model()
        if use_offline_mock:
            llm = MockLLM()
        else:
            if not groq_api_key:
                st.error("⚠️ Please enter a Groq API key in the sidebar, or select 'Offline / Mock Mode'.")
                st.stop()
            try:
                llm = GroqLLM(api_key=groq_api_key, model=model_option)
            except Exception as e:
                st.error(f"⚠️ Could not connect to Groq: {e}. Please check your internet connection or API key.")
                st.stop()

        # Preserve existing chunks if re-configuring engine
        existing_chunks = st.session_state.rag_engine.chunks_map if st.session_state.rag_engine else {}

        st.session_state.rag_engine = GraphRAG(
            llm=llm,
            embedding_model=embedding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        st.session_state.rag_engine.set_chunks(existing_chunks)

        # Auto-load saved index from disk if present!
        if (STORAGE_DIR / "graph.json").exists() and len(st.session_state.rag_engine.chunks_map) == 0:
            try:
                st.session_state.rag_engine.load(str(STORAGE_DIR))
                st.session_state.is_indexed = True
                if not st.session_state.graph_html:
                    st.session_state.graph_html = st.session_state.rag_engine.get_graph_html()
            except Exception:
                pass

    if st.session_state.rag_engine:
        st.session_state.rag_engine.chunker.chunk_size = chunk_size
        st.session_state.rag_engine.chunker.chunk_overlap = chunk_overlap

    return st.session_state.rag_engine


# Header
st.markdown('<div class="main-title">🕸️ Graph RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-hop Knowledge Graph Extraction, Community Sensemaking, and Augmented Retrieval</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_ingest, tab_graph, tab_chat, tab_analytics = st.tabs([
    "📂 1. Document Ingestion",
    "🕸️ 2. Knowledge Graph View",
    "💬 3. Chat & Retrieval",
    "📊 4. Graph Analytics",
])

# ==========================================
# TAB 1: Document Ingestion
# ==========================================
with tab_ingest:
    st.markdown("#### Ingest Unstructured Documents")
    st.markdown("Upload files (PDF, TXT, MD) or paste text to extract entities, relationships, and communities.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("##### 📄 Upload Files")
        uploaded_files = st.file_uploader(
            "Select files:",
            type=["txt", "md", "markdown", "pdf"],
            accept_multiple_files=True,
        )

        st.markdown("##### ⚡ Quick Start with Sample Data")
        if st.button("📥 Load Built-in AI & Biotech Dataset (3 Docs)", use_container_width=True):
            engine = get_engine()
            sample_dir = Path(__file__).parent / "sample_data"
            loaded_count = 0
            if sample_dir.exists():
                for f in sample_dir.glob("*.txt"):
                    engine.add_file(f)
                    loaded_count += 1
            st.success(f"Loaded {loaded_count} sample documents into staging! Ready to build index.")

    with col2:
        with st.expander("✍️ Or Paste Text Directly", expanded=False):
            pasted_title = st.text_input("Document Identifier:", value="manual_notes")
            pasted_text = st.text_area("Paste text content here:", height=100)
            if st.button("Add Pasted Text to Staging", use_container_width=True):
                if pasted_text.strip():
                    engine = get_engine()
                    engine.add_text(pasted_text, doc_id=pasted_title)
                    st.success(f"Added '{pasted_title}' to staging!")
                else:
                    st.warning("Please paste some text first.")

    st.markdown("---")
    st.markdown("##### ⚡ Parameter Optimization")
    col_t1, col_t2 = st.columns([1, 2], gap="medium")
    with col_t1:
        if st.button("🪄 Auto-Set Parameters from Staged Text", use_container_width=True, help="Automatically inspects staged text/files and configures optimal chunking and graph hop parameters."):
            engine = get_engine()
            total_len = sum(len(c.text) for c in engine.chunks_map.values())
            doc_cnt = len(set(c.doc_id for c in engine.chunks_map.values()))
            if uploaded_files:
                for uf in uploaded_files:
                    total_len += len(uf.getvalue())
                    doc_cnt += 1
            if pasted_text.strip():
                total_len += len(pasted_text.strip())
                doc_cnt += 1
            c_size, c_overlap, hops, b_size, reason = auto_tune_parameters(total_len, max(doc_cnt, 1))
            st.session_state.chunk_size = c_size
            st.session_state.chunk_overlap = c_overlap
            st.session_state.max_hops = hops
            st.session_state.batch_size = b_size
            st.session_state.auto_tuned_msg = reason
            st.rerun()
    with col_t2:
        if st.session_state.get("auto_tuned_msg"):
            st.info(st.session_state.auto_tuned_msg, icon="✨")
        else:
            st.caption(f"Active Settings: **{chunk_size}** chars chunk size • **{chunk_overlap}** overlap • **{max_hops}** hops • **{st.session_state.batch_size}** chunks/batch. Click Auto-Set to compute optimal parameters for your documents.")

    st.markdown("---")

    # Ingestion Trigger
    engine = get_engine()
    staged_chunks = len(engine.chunks_map)
    st.info(f"**Staged Chunks ready for indexing:** {staged_chunks}")

    if st.button("🚀 Build Knowledge Graph Index", type="primary", use_container_width=True, disabled=(staged_chunks == 0 and not uploaded_files)):
        # Process uploaded files first
        if uploaded_files:
            for uf in uploaded_files:
                engine.add_bytes(uf.getvalue(), uf.name)

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, val: float):
            status_text.text(msg)
            progress_bar.progress(min(val, 1.0))

        with st.spinner("Extracting graph triples and building indices..."):
            try:
                results = engine.build_index(
                    batch_size=st.session_state.batch_size,
                    progress_callback=update_progress,
                )
                engine.save(str(STORAGE_DIR))
                st.session_state.is_indexed = True
                st.session_state.graph_html = engine.get_graph_html()
                if results.get("status") == "partial":
                    st.warning(f"⚠️ {results.get('message')}\n\nYour graph with {results.get('entities_count', 0)} entities is saved and ready to explore in Tabs 2 & 3! You can click 'Build Knowledge Graph Index' anytime to resume remaining batches.")
                else:
                    st.success("✅ Knowledge Graph indexed and auto-saved to disk! It will now persist across page reloads.")
            except Exception as e:
                st.error(f"Indexing error: {e}")

    # Display status metrics
    if st.session_state.is_indexed:
        stats = engine.get_stats()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed Chunks", len(engine.chunks_map))
        m2.metric("Extracted Entities", stats.node_count)
        m3.metric("Relationships", stats.edge_count)
        m4.metric("Communities", stats.community_count)


# ==========================================
# TAB 2: Interactive Knowledge Graph
# ==========================================
with tab_graph:
    st.markdown("#### Interactive Knowledge Graph Explorer")
    st.caption("Drag nodes, zoom with mousewheel, click any node to focus its connected neighborhood, or filter by community cluster.")

    if not st.session_state.is_indexed or not st.session_state.graph_html:
        st.info("Knowledge Graph is currently empty. Please index some documents in the Ingestion tab first.")
    else:
        engine = get_engine()
        stats = engine.get_stats()

        # Modern Legend Bar
        st.markdown(
            """
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; font-size: 12px;">
                <span style="background: rgba(56, 189, 248, 0.2); border: 1px solid #38bdf8; border-radius: 9999px; padding: 2px 10px; color: #38bdf8;">🏢 Organization</span>
                <span style="background: rgba(244, 63, 94, 0.2); border: 1px solid #f43f5e; border-radius: 9999px; padding: 2px 10px; color: #f43f5e;">👤 Person</span>
                <span style="background: rgba(6, 182, 212, 0.2); border: 1px solid #06b6d4; border-radius: 9999px; padding: 2px 10px; color: #06b6d4;">📄 Document</span>
                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 9999px; padding: 2px 10px; color: #10b981;">⚙️ Technology</span>
                <span style="background: rgba(168, 85, 247, 0.2); border: 1px solid #a855f7; border-radius: 9999px; padding: 2px 10px; color: #a855f7;">💡 Concept</span>
                <span style="background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; border-radius: 9999px; padding: 2px 10px; color: #f59e0b;">📍 Location</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.3, 1.2])
        with ctrl_col1:
            comm_options = ["All Communities"] + [f"Community {i}" for i in range(stats.community_count)]
            selected_comm_str = st.selectbox("Filter Community:", comm_options)
            filter_id = None
            if selected_comm_str != "All Communities":
                filter_id = int(selected_comm_str.split()[1])

        with ctrl_col2:
            view_mode = st.radio(
                "Graph Density Mode:",
                options=["Connected Core (Degree ≥ 1)", "All Entities (Include Unconnected)"],
                index=0,
                horizontal=True,
                help="Connected Core hides isolated singletons, creating a crisp, interconnected relationship graph."
            )
            min_deg = 1 if "Connected Core" in view_mode else 0

        with ctrl_col3:
            color_mode = st.radio(
                "Color Palette:",
                options=["By Entity Type", "By Community Cluster"],
                index=0,
                horizontal=True,
                help="Color nodes either by their semantic type or by Louvain modularity clusters."
            )
            color_by = "community" if "Community" in color_mode else "type"

        # Render HTML component
        html_content = engine.get_graph_html(
            highlight_nodes=st.session_state.highlight_nodes,
            filter_community_id=filter_id,
            min_degree=min_deg,
            color_by=color_by,
        )
        components.html(html_content, height=700, scrolling=False)


# ==========================================
# TAB 3: Chat & Retrieval
# ==========================================
with tab_chat:
    st.markdown("#### Ask Questions with Graph-Augmented Retrieval")

    if not st.session_state.is_indexed:
        st.warning("Please build the Knowledge Graph index in Tab 1 before querying.")
    else:
        engine = get_engine()

        # Query options
        q_col1, q_col2 = st.columns([1, 2])
        with q_col1:
            search_mode = st.radio(
                "Search Mode:",
                options=["Local Search", "Global Search"],
                help="Local: traverses entity subgraphs + text chunks for specific questions.\nGlobal: synthesizes corpus themes using community reports.",
            )

        with q_col2:
            st.markdown("##### 💡 Example Questions")
            ex_cols = st.columns(3)
            example_prompt = None
            if ex_cols[0].button("Alice Zhao & CureX", use_container_width=True):
                example_prompt = "How is Alice Zhao connected to CureX and what role did NexusLabs play?"
            if ex_cols[1].button("ApexCloud Role", use_container_width=True):
                example_prompt = "What infrastructure and agreements did ApexCloud provide for NexusLabs and CureX?"
            if ex_cols[2].button("Corpus Overview (Global)", use_container_width=True):
                example_prompt = "What are the major organizational partnerships and scientific breakthroughs in this dataset?"

        query_input = st.chat_input("Ask a question about the indexed knowledge...")
        effective_query = example_prompt or query_input

        if effective_query:
            mode_arg = "global" if search_mode == "Global Search" else "local"

            with st.spinner(f"Executing {search_mode}..."):
                try:
                    result = engine.query(
                        effective_query,
                        mode=mode_arg,
                        max_hops=max_hops,
                    )

                    # Highlight retrieved entities in graph tab
                    highlight_names = [e.name for e in result.cited_entities]
                    st.session_state.highlight_nodes = highlight_names

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "query": effective_query,
                        "result": result,
                        "mode": search_mode,
                    })
                except Exception as e:
                    st.error(f"Error executing query: {e}")

        # Render conversation history
        for item in reversed(st.session_state.chat_history):
            res = item["result"]
            st.chat_message("user").write(item["query"])

            with st.chat_message("assistant"):
                st.markdown(f"**[{item['mode']}]**\n\n{res.answer}")

                # Expandable details
                with st.expander("🔍 Knowledge Graph Context & Citations"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Cited Entities:**")
                        if res.cited_entities:
                            for ent in res.cited_entities:
                                st.markdown(f"- **{ent.name}** (`{ent.type}`): {ent.description}")
                        else:
                            st.write("None")

                    with c2:
                        st.markdown("**Cited Relationships (Triples):**")
                        if res.cited_relations:
                            for rel in res.cited_relations:
                                st.markdown(f"- `{rel.source}` ➔ **{rel.relation_type}** ➔ `{rel.target}`")
                        else:
                            st.write("None")

                    if res.cited_chunks:
                        st.markdown("---")
                        st.markdown("**Supporting Source Text Chunks:**")
                        for c in res.cited_chunks:
                            st.caption(f"Chunk ID: `{c.id}` (from `{c.doc_id}`)")
                            st.text(c.text[:400] + ("..." if len(c.text) > 400 else ""))


# ==========================================
# TAB 4: Graph Analytics
# ==========================================
with tab_analytics:
    st.markdown("#### Knowledge Graph Topology & Community Reports")

    if not st.session_state.is_indexed:
        st.info("No indexed graph data available. Please build the index in Tab 1.")
    else:
        engine = get_engine()
        stats = engine.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Node Count", stats.node_count)
        col2.metric("Edge Count", stats.edge_count)
        col3.metric("Network Density", f"{stats.density:.4f}")
        col4.metric("Total Communities", stats.community_count)

        st.markdown("---")
        c_left, c_right = st.columns([1, 1], gap="large")

        with c_left:
            st.markdown("##### 🏆 Top Entities by Degree Centrality")
            if stats.top_entities:
                for ent_info in stats.top_entities:
                    st.markdown(f"- **{ent_info['name']}** ({ent_info['type']}) — Degree: `{ent_info['degree']}`")
            else:
                st.write("No entities available.")

        with c_right:
            st.markdown("##### 🏘️ Detected Communities & Summaries")
            communities = engine.community_store.get_all_communities()
            if communities:
                for comm in communities:
                    with st.expander(f"Community {comm.id}: {comm.title}"):
                        st.markdown(f"**Members:** {', '.join(comm.entity_names)}")
                        st.markdown(f"**Summary:** {comm.summary}")
                        if comm.findings:
                            st.markdown("**Key Findings:**")
                            for f in comm.findings:
                                st.markdown(f"- {f}")
            else:
                st.write("No communities detected.")
