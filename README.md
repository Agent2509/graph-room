# 🕸️ Graph RAG: 100% Free Multiplatform Knowledge Graph RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Groq Powered](https://img.shields.io/badge/Groq-Free%20Tier-orange.svg)](https://groq.com/)
[![Embeddings: FastEmbed](https://img.shields.io/badge/FastEmbed-ONNX%20CPU-green.svg)](https://github.com/qdrant/fastembed)

A complete, production-grade **Graph RAG (Graph Retrieval-Augmented Generation)** framework built completely from scratch in Python. Designed for **100% free-of-cost, multiplatform deployment** across **Hugging Face Spaces**, **Streamlit Community Cloud**, **Docker**, and local environments.

---

## 🚀 Key Highlights

* **100% Free Inference**:
  * **LLM**: Powered by **Groq Cloud Free Tier** (default `openai/gpt-oss-20b`, with automatic failover to `openai/gpt-oss-120b` and Llama models) with ultra-fast generation and high throughput.
  * **Embeddings**: Powered by **FastEmbed** (`bge-small-en-v1.5`) running locally on CPU in milliseconds via ONNX Runtime — **0 API cost, 0 rate limits, minimal RAM (~100MB)**.
* **Embedded Storage**:
  * Graph Engine: Built on **NetworkX** with multi-relational attributes, $k$-hop ego-network traversals, and JSON persistence — **zero external database setup or hosting fees**.
  * Vector Store: In-memory **NumPy cosine similarity** index with instant JSON serialization.
* **Dual Search Modes**:
  * **Local Search**: Answers targeted, entity-grounded questions by traversing multi-hop subgraphs, extracting connected triples, and pairing them with source text chunks.
  * **Global Search**: Answers broad, corpus-wide questions by clustering the graph into **Louvain Communities** and synthesizing pre-compiled community analytical reports.
* **Interactive Physics Visualization**:
  * Embedded **PyVis / Vis.js** network graph with clickable nodes, color-coded community clusters, edge relationship inspection, and real-time subgraph highlighting.
* **Multiplatform Deployment**:
  * **Streamlit Web Studio**: Full-featured web app with chat interface, document uploader, and graph inspector.
  * **FastAPI Backend**: Headless REST API with auto-generated OpenAPI / Swagger docs.
  * **Dockerized**: 1-command startup with `docker compose up`.

---

## 🏛️ System Architecture

```
                                 [ User / Query ]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
      [ Local Search ]                                      [ Global Search ]
(Entity Vector Search + Subgraph Traversal)           (Semantic Community Report Search)
             │                                                     │
             ▼                                                     ▼
 [ Subgraph Triples + Raw Chunks ]                       [ Community Summaries ]
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                             [ Groq Llama 3.3 LLM ]
                                        │
                                        ▼
                   [ Grounded Answer with Graph Citations ]
```

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install dependencies using `uv` (recommended) or standard Python `venv`:

```bash
# Using uv (ultra-fast)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your free Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_free_key_here
```
*(You can obtain a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)).*

> **Note**: You can also run in **Offline / Mock Mode** without an API key!

### 3. Run Interactive Demo

```bash
.venv/bin/python examples/demo.py
```

### 4. Launch Streamlit Web App

```bash
.venv/bin/streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Launch FastAPI REST Service

```bash
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

Run both the Streamlit Studio and FastAPI backend with Docker Compose:

```bash
# Provide key or leave blank for offline mock mode
export GROQ_API_KEY=gsk_...

docker compose up --build
```
* **Web Studio**: `http://localhost:8501`
* **REST API**: `http://localhost:8000`

---

## ☁️ Free Cloud Deployment Guides

### Deploy to Hugging Face Spaces (100% Free)
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK and **CPU Basic (Free 16 GB RAM)**.
3. Push this repository's files to your HF Space.
4. Under **Settings > Variables and secrets**, add your `GROQ_API_KEY`.
5. Your interactive Graph RAG app is live on the internet for free!

### Deploy to Streamlit Community Cloud (100% Free)
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select your repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings > Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
5. Click **Deploy**!

---

## 📖 Python Library Usage

```python
from graph_rag import GraphRAG
from graph_rag.llm import GroqLLM, FastEmbedEmbedding

# Initialize engine
rag = GraphRAG(
    llm=GroqLLM(api_key="gsk_..."),
    embedding_model=FastEmbedEmbedding(),
    chunk_size=800,
    chunk_overlap=120,
)

# 1. Ingest files or raw text
rag.add_file("sample_data/nexuslabs.txt")
rag.add_file("sample_data/deepbio.txt")
rag.add_file("sample_data/curex_trials.txt")

# 2. Build Knowledge Graph & vector index
rag.build_index()

# 3. Local Search (multi-hop entity traversal)
res_local = rag.query("How is Alice Zhao connected to CureX?", mode="local")
print(res_local.answer)

# 4. Global Search (corpus community sensemaking)
res_global = rag.query("What are the key themes and breakthroughs?", mode="global")
print(res_global.answer)

# 5. Persist and reload index
rag.save("./saved_graph_index")

new_rag = GraphRAG(...)
new_rag.load("./saved_graph_index")
```

---

## 🧪 Testing

Run the automated test suite:

```bash
.venv/bin/pytest tests/ -v
```

---

## 📜 License

MIT License. Free for personal, academic, and commercial use.
