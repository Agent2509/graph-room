"""Counts, density and top entities."""

import json

from fastapi import APIRouter, Depends, Request

from graph_rag.api.deps import require_token

router = APIRouter(prefix="/api/stats", tags=["stats"], dependencies=[Depends(require_token)])


@router.get("")
def get_stats(request: Request) -> dict:
    conn = request.app.state.db.conn()
    counts = {
        "documents": conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"],
        "chunks": conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"],
        "entities": conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"],
        "edges": conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"],
        "communities": conn.execute("SELECT COUNT(*) AS c FROM communities").fetchone()["c"],
        "cooccurrences": conn.execute(
            "SELECT COUNT(*) AS c FROM (SELECT src, dst FROM chunk_pairs GROUP BY src, dst)"
        ).fetchone()["c"],
    }
    top_entities = [
        dict(row)
        for row in conn.execute(
            "SELECT e.name, e.type, COUNT(*) AS degree "
            "FROM (SELECT src AS name FROM edges UNION ALL SELECT dst FROM edges) d "
            "JOIN entities e ON e.name = d.name "
            "GROUP BY e.name ORDER BY degree DESC LIMIT 10"
        )
    ]
    communities = [
        {
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "findings": json.loads(row["findings"]),
            "rating": row["rating"],
        }
        for row in conn.execute("SELECT * FROM communities ORDER BY id")
    ]
    nodes, edges = counts["entities"], counts["edges"]
    density = round(edges / (nodes * (nodes - 1)), 4) if nodes > 1 else 0.0
    return {
        **counts,
        "density": density,
        "top_entities": top_entities,
        "community_reports": communities,
    }
