"""FastAPI REST API for Graph RAG."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from graph_rag.engine import GraphRAG
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.llm.mock import MockLLM, MockEmbedding

app = FastAPI(
    title="Graph RAG REST API",
    description="100% Free Multiplatform Knowledge Graph RAG API powered by Groq & FastEmbed",
    version="0.1.0",
)

# Enable CORS for cross-origin frontend clients
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501,http://localhost:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
_engine: Optional[GraphRAG] = None


def get_engine() -> GraphRAG:
    global _engine
    if _engine is None:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            llm = GroqLLM(api_key=api_key)
            embed = FastEmbedEmbedding()
        else:
            # Fallback to Mock LLM if no key configured
            llm = MockLLM()
            embed = FastEmbedEmbedding()
        _engine = GraphRAG(llm=llm, embedding_model=embed)
    return _engine


class TextIndexRequest(BaseModel):
    text: str
    doc_id: str = "document"


class QueryRequest(BaseModel):
    question: str
    mode: str = "auto"
    max_hops: int = 1


class QueryResponse(BaseModel):
    query: str
    answer: str
    search_mode: str
    cited_entities: List[Dict[str, Any]]
    cited_relations: List[Dict[str, Any]]
    cited_chunks: List[Dict[str, Any]]
    graph_context: str


@app.get("/")
def read_root():
    return {
        "service": "Graph RAG API",
        "version": "0.1.0",
        "status": "healthy",
        "docs_url": "/docs",
    }


@app.post("/api/index/text")
def index_text(req: TextIndexRequest):
    engine = get_engine()
    engine.add_text(req.text, doc_id=req.doc_id)
    results = engine.build_index()
    return {"status": "success", "results": results}


@app.post("/api/index/file")
async def index_file(file: UploadFile = File(...)):
    engine = get_engine()
    contents = await file.read()
    engine.add_bytes(contents, file.filename or "uploaded_doc")
    results = engine.build_index()
    return {"status": "success", "filename": file.filename, "results": results}


@app.post("/api/index/sample")
def index_sample_dataset():
    engine = get_engine()
    sample_dir = Path(__file__).parent / "sample_data"
    count = 0
    if sample_dir.exists():
        for f in sample_dir.glob("*.txt"):
            engine.add_file(f)
            count += 1
    results = engine.build_index()
    return {"status": "success", "sample_files_loaded": count, "results": results}


@app.post("/api/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    engine = get_engine()
    if len(engine.chunks_map) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Index documents first.")

    res = engine.query(req.question, mode=req.mode, max_hops=req.max_hops)
    return QueryResponse(
        query=res.query,
        answer=res.answer,
        search_mode=res.search_mode,
        cited_entities=[e.model_dump() for e in res.cited_entities],
        cited_relations=[r.model_dump() for r in res.cited_relations],
        cited_chunks=[c.model_dump() for c in res.cited_chunks],
        graph_context=res.graph_context_text,
    )


@app.get("/api/graph")
def get_graph():
    engine = get_engine()
    return engine.graph_store.to_dict()


@app.get("/api/graph/html", response_class=HTMLResponse)
def get_graph_visualization():
    engine = get_engine()
    return HTMLResponse(content=engine.get_graph_html())


@app.get("/api/stats")
def get_stats():
    engine = get_engine()
    return engine.get_stats().model_dump()
