"""FastAPI application factory."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from graph_rag import __version__
from graph_rag import config as config_module
from graph_rag.api.routes import documents, graph, health, jobs, query, stats
from graph_rag.config import Settings
from graph_rag.db import Database
from graph_rag.embeddings import Embedder
from graph_rag.jobs import JobRunner
from graph_rag.llm import GroqLLM, MockLLM

logger = logging.getLogger("graph_rag")


def build_llm(config: Settings):
    if config.groq_api_key:
        return GroqLLM(config.groq_api_key, config.groq_model)
    logger.warning("GROQ_API_KEY is not set: using the offline mock LLM")
    return MockLLM()


def create_app(config: Settings | None = None, llm=None, embed=None) -> FastAPI:
    config = config or config_module.settings
    logging.basicConfig(
        level=config.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    app = FastAPI(
        title="Graph RAG 2.0",
        version=__version__,
        description="Knowledge graph RAG in one SQLite file. Streamed answers, cached communities.",
    )
    app.state.settings = config
    app.state.db = Database(config.db_path)
    app.state.llm = llm or build_llm(config)
    app.state.embed = embed or Embedder()
    app.state.runner = JobRunner(app.state.db)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (health, documents, jobs, query, graph, stats):
        app.include_router(module.router)

    if config.static_dir.exists():
        app.mount("/", StaticFiles(directory=config.static_dir, html=True), name="static")

    return app
