"""Health and feature flags. Never requires a token."""

from fastapi import APIRouter, Request

from graph_rag import __version__

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(request: Request) -> dict:
    state = request.app.state
    conn = state.db.conn()
    vector_version = conn.execute("SELECT vec_version() AS v").fetchone()["v"]
    return {
        "status": "ok",
        "version": __version__,
        "counts": {
            "documents": conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"],
            "chunks": conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"],
            "entities": conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"],
        },
        "features": {
            "llm": "injected" if state.llm is not None else "user-supplied",
            "requires_llm_key": state.llm is None,
            "embeddings": state.embed.model_name,
            "vectors": f"sqlite-vec {vector_version}",
            "auth_required": bool(state.settings.api_token),
        },
    }
