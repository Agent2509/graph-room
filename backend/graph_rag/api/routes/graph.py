"""Graph payload for the canvas."""

from fastapi import APIRouter, Depends, Query, Request

from graph_rag.api.deps import require_token
from graph_rag.pipeline.graph import graph_payload

router = APIRouter(prefix="/api/graph", tags=["graph"], dependencies=[Depends(require_token)])


@router.get("")
def get_graph(
    request: Request,
    community: int | None = None,
    min_degree: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict:
    return graph_payload(request.app.state.db.conn(), community, min_degree, limit)
