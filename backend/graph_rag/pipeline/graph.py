"""Graph persistence. Aggregates are always derived from mention rows, so writes
are idempotent and a document delete is exactly revertible.
"""

import json
import sqlite3
from collections import defaultdict
from itertools import combinations

from graph_rag.models import ExtractedEdge, ExtractedEntity
from graph_rag.pipeline import vectors
from graph_rag.pipeline.resolver import merge_descriptions


def write_extraction(
    conn: sqlite3.Connection,
    entities: dict[str, ExtractedEntity],
    edges: list[ExtractedEdge],
) -> None:
    for entity in entities.values():
        row = conn.execute(
            "SELECT type FROM entities WHERE name=?", (entity.name,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO entities(name, type, description) VALUES (?, ?, '')",
                (entity.name, entity.type),
            )
        elif row["type"] == "CONCEPT" and entity.type != "CONCEPT":
            conn.execute("UPDATE entities SET type=? WHERE name=?", (entity.type, entity.name))

        for chunk_id in set(entity.chunk_ids):
            conn.execute(
                "INSERT OR REPLACE INTO entity_mentions(name, chunk_id, description) "
                "VALUES (?, ?, ?)",
                (entity.name, chunk_id, entity.description[:600]),
            )
        _recompute_entity(conn, entity.name)

    for edge in edges:
        for endpoint in (edge.src, edge.dst):
            conn.execute(
                "INSERT OR IGNORE INTO entities(name, type) VALUES (?, 'CONCEPT')", (endpoint,)
            )
        conn.execute(
            "INSERT OR IGNORE INTO edges(src, dst, rel) VALUES (?, ?, ?)",
            (edge.src, edge.dst, edge.rel),
        )
        for chunk_id in set(edge.chunk_ids):
            conn.execute(
                "INSERT OR REPLACE INTO edge_mentions(src, dst, rel, chunk_id, description, weight) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (edge.src, edge.dst, edge.rel, chunk_id, edge.description[:400], edge.weight),
            )
        _recompute_edge(conn, edge.src, edge.dst, edge.rel)


def write_cooccurrence(
    conn: sqlite3.Connection, document_id: str, max_entities: int = 25
) -> None:
    """Remember which entities share a chunk. Bounded: at most 25 per chunk."""
    rows = conn.execute(
        "SELECT c.id AS chunk_id, m.name AS name FROM chunks c "
        "JOIN entity_mentions m ON m.chunk_id = c.id "
        "WHERE c.document_id=? ORDER BY c.id, m.name",
        (document_id,),
    ).fetchall()
    by_chunk: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_chunk[row["chunk_id"]].append(row["name"])
    for chunk_id, names in by_chunk.items():
        unique = sorted(set(names))[:max_entities]
        for left, right in combinations(unique, 2):
            conn.execute(
                "INSERT OR IGNORE INTO chunk_pairs(chunk_id, src, dst) VALUES (?, ?, ?)",
                (chunk_id, left, right),
            )


def cooccurrence_pairs(
    conn: sqlite3.Connection, per_node: int = 3
) -> list[tuple[str, str, float]]:
    """Strongest co-occurrence partners per entity, capped so the map stays readable."""
    rows = conn.execute(
        "SELECT src, dst, COUNT(*) AS weight FROM chunk_pairs GROUP BY src, dst"
    ).fetchall()
    weights = {(row["src"], row["dst"]): float(row["weight"]) for row in rows}

    by_node: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (left, right), weight in weights.items():
        by_node[left].append((weight, right))
        by_node[right].append((weight, left))

    kept: set[tuple[str, str]] = set()
    degree: dict[str, int] = defaultdict(int)
    for node in sorted(by_node):
        for _, other in sorted(by_node[node], key=lambda pair: (-pair[0], pair[1])):
            if degree[node] >= per_node:
                break
            if degree[other] >= per_node:
                continue
            pair = (node, other) if node < other else (other, node)
            kept.add(pair)
            degree[node] += 1
            degree[other] += 1
    return [(left, right, weights[(left, right)]) for left, right in sorted(kept)]


def delete_document(db, document_id: str) -> dict[str, int]:
    conn = db.conn()
    affected_entities = [
        row["name"]
        for row in conn.execute(
            "SELECT DISTINCT name FROM entity_mentions "
            "WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",
            (document_id,),
        )
    ]
    affected_edges = [
        (row["src"], row["dst"], row["rel"])
        for row in conn.execute(
            "SELECT DISTINCT src, dst, rel FROM edge_mentions "
            "WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=?)",
            (document_id,),
        )
    ]
    chunk_ids = [
        row["id"]
        for row in conn.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))
    ]

    with db.tx() as tx:
        tx.execute("DELETE FROM documents WHERE id=?", (document_id,))
        for chunk_id in chunk_ids:
            vectors.delete(tx, "chunk", chunk_id)
        for name in affected_entities:
            _recompute_entity(tx, name)
            vectors.delete(tx, "entity", name)
        for src, dst, rel in affected_edges:
            _recompute_edge(tx, src, dst, rel)
        tx.execute(
            "DELETE FROM entities WHERE name NOT IN (SELECT name FROM entity_mentions)"
        )
        tx.execute(
            "DELETE FROM edges WHERE NOT EXISTS ("
            "SELECT 1 FROM edge_mentions m "
            "WHERE m.src=edges.src AND m.dst=edges.dst AND m.rel=edges.rel)"
        )

    return {
        "chunks_removed": len(chunk_ids),
        "entities_affected": len(affected_entities),
        "edges_affected": len(affected_edges),
    }


def _recompute_entity(conn: sqlite3.Connection, name: str) -> None:
    rows = conn.execute(
        "SELECT description FROM entity_mentions WHERE name=?", (name,)
    ).fetchall()
    if not rows:
        return
    description = ""
    for row in rows:
        description = merge_descriptions(description, row["description"])
    conn.execute("UPDATE entities SET description=? WHERE name=?", (description, name))


def _recompute_edge(conn: sqlite3.Connection, src: str, dst: str, rel: str) -> None:
    rows = conn.execute(
        "SELECT description, weight FROM edge_mentions WHERE src=? AND dst=? AND rel=?",
        (src, dst, rel),
    ).fetchall()
    if not rows:
        return
    description = ""
    total = 0.0
    for row in rows:
        description = merge_descriptions(description, row["description"], max_chars=400)
        total += float(row["weight"])
    conn.execute(
        "UPDATE edges SET description=?, weight=? WHERE src=? AND dst=? AND rel=?",
        (description, total, src, dst, rel),
    )


def subgraph(
    conn: sqlite3.Connection, seeds: list[str], max_hops: int = 2, max_nodes: int = 40
) -> tuple[list[dict], list[dict]]:
    """Bounded neighbourhood via a recursive CTE over the adjacency view."""
    if not seeds:
        return [], []
    rows = conn.execute(
        """
        WITH RECURSIVE walk(name, depth) AS (
            SELECT value, 0 FROM json_each(?)
            WHERE value IN (SELECT name FROM entities)
            UNION
            SELECT adjacency.v, walk.depth + 1
            FROM adjacency JOIN walk ON adjacency.u = walk.name
            WHERE walk.depth < ?
        )
        SELECT name FROM walk LIMIT ?
        """,
        (json.dumps(seeds), max_hops, max_nodes),
    ).fetchall()
    names = [row["name"] for row in rows]
    if not names:
        return [], []

    placeholders = ",".join("?" for _ in names)
    entities = [
        dict(row)
        for row in conn.execute(
            f"SELECT name, type, description FROM entities WHERE name IN ({placeholders})",
            names,
        )
    ]
    edges = [
        dict(row)
        for row in conn.execute(
            f"SELECT src, dst, rel, description, weight FROM edges "
            f"WHERE src IN ({placeholders}) AND dst IN ({placeholders})",
            [*names, *names],
        )
    ]
    return entities, edges


def graph_payload(
    conn: sqlite3.Connection,
    community: int | None = None,
    min_degree: int = 0,
    limit: int = 500,
) -> dict:
    members = None
    if community is not None:
        members = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM community_members WHERE community_id=?", (community,)
            )
        ]
        if not members:
            return {"nodes": [], "edges": []}

    degrees = {
        row["name"]: row["degree"]
        for row in conn.execute(
            "SELECT name, COUNT(*) AS degree FROM ("
            "  SELECT src AS name FROM edges UNION ALL SELECT dst FROM edges"
            ") GROUP BY name"
        )
    }
    communities = {
        row["name"]: row["community_id"]
        for row in conn.execute("SELECT name, community_id FROM community_members")
    }
    rows = conn.execute("SELECT name, type, description FROM entities").fetchall()
    nodes = [
        {
            "id": row["name"],
            "type": row["type"],
            "description": row["description"],
            "degree": degrees.get(row["name"], 0),
            "community": communities.get(row["name"]),
        }
        for row in rows
        if degrees.get(row["name"], 0) >= min_degree
        and (members is None or row["name"] in members)
    ]
    nodes.sort(key=lambda node: node["degree"], reverse=True)
    nodes = nodes[:limit]
    keep = {node["id"] for node in nodes}

    edges = [
        {**dict(row), "kind": "relation"}
        for row in conn.execute("SELECT src, dst, rel, description, weight FROM edges")
        if row["src"] in keep and row["dst"] in keep
    ]

    # Structural edges: every document points at the entities it mentions. Without
    # these a document with no prose relations produced a blank canvas.
    documents: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT d.id AS document_id, d.filename, m.name AS entity, COUNT(*) AS weight "
        "FROM entity_mentions m "
        "JOIN chunks c ON c.id = m.chunk_id "
        "JOIN documents d ON d.id = c.document_id "
        "GROUP BY d.id, m.name"
    ):
        if row["entity"] not in keep:
            continue
        node_id = f"doc:{row['document_id']}"
        if node_id not in documents:
            documents[node_id] = {
                "id": node_id, "type": "DOCUMENT", "name": row["filename"],
                "description": "", "degree": 0, "mentions": 0,
            }
        documents[node_id]["mentions"] += row["weight"]
        edges.append(
            {
                "src": node_id, "dst": row["entity"], "rel": "MENTIONED_IN",
                "description": "", "weight": row["weight"], "kind": "mention",
            }
        )

    relation_pairs = {(edge["src"], edge["dst"]) for edge in edges}
    for left, right, weight in cooccurrence_pairs(conn):
        if left not in keep or right not in keep:
            continue
        if (left, right) in relation_pairs or (right, left) in relation_pairs:
            continue
        edges.append(
            {
                "src": left, "dst": right, "rel": "CO_OCCURS",
                "description": "", "weight": weight, "kind": "cooccur",
            }
        )

    nodes.extend(documents.values())
    return {"nodes": nodes, "edges": edges}
