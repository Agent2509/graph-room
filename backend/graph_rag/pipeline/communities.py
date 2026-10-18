"""Community detection (Louvain) and cached summaries.

Summaries are reused when the member set is unchanged, so rebuilding an index does
not re-pay for LLM calls. All slow work (LLM, embeddings) happens before the write
transaction opens: a long write lock would block every other writer.
"""

import hashlib
import json
import sqlite3

import networkx as nx

from graph_rag.db import Database
from graph_rag.models import Community
from graph_rag.pipeline import vectors
from graph_rag.pipeline.graph import cooccurrence_pairs

SUMMARY_SYSTEM = """You are an analyst writing a short report about a cluster of related
entities from a knowledge graph.

Return JSON: {"title": "...", "summary": "2-3 sentences", "findings": ["..."], "rating": 7.0}
Ground everything in the supplied entities and relationships."""

SUMMARY_USER = """Entities:
{entities}

Relationships:
{relations}"""


def detect(conn: sqlite3.Connection) -> list[list[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(row["name"] for row in conn.execute("SELECT name FROM entities"))
    graph.add_edges_from(
        (row["src"], row["dst"]) for row in conn.execute("SELECT src, dst FROM edges")
    )
    graph.add_edges_from(
        (left, right) for left, right, _ in cooccurrence_pairs(conn)
    )
    if graph.number_of_nodes() == 0:
        return []
    if graph.number_of_edges() == 0:
        return []  # isolated entities are not a community, and 100 of them are noise
    try:
        parts = nx.community.louvain_communities(graph, seed=42)
    except Exception:  # noqa: BLE001 - fall back to components on any solver issue
        parts = nx.connected_components(graph)
    return [sorted(part) for part in parts if len(part) >= 2]


def rebuild(db: Database, llm=None, embed=None, max_llm: int = 12) -> int:
    """Recompute communities, reusing summaries whose members did not change."""
    conn = db.conn()
    memberships = detect(conn)
    rows: list[tuple[int, list[str], str, str, str, float, list[float] | None, str]] = []

    for index, members in enumerate(memberships):
        digest = hashlib.sha1("\x1f".join(members).encode()).hexdigest()
        cached = conn.execute(
            "SELECT title, summary, findings, rating FROM communities "
            "WHERE members_hash=? AND source='llm' LIMIT 1",
            (digest,),
        ).fetchone()
        if cached:
            title, summary, findings, rating = (
                cached["title"], cached["summary"], cached["findings"], cached["rating"],
            )
            source = "llm"
        else:
            use_llm = llm if llm and index < max_llm and len(members) >= 2 else None
            community = _summarize(use_llm, index, members, conn)
            title, summary, findings, rating = (
                community.title, community.summary,
                json.dumps(community.findings), community.rating,
            )
            source = "llm" if use_llm is not None else "heuristic"
        vector = embed.one(f"{title}\n{summary}") if embed is not None else None
        rows.append((index, members, title, summary, findings, rating, vector, source))

    with db.tx() as tx:
        tx.execute("DELETE FROM communities")
        tx.execute("DELETE FROM vec_items WHERE kind='community'")
        for index, members, title, summary, findings, rating, vector, source in rows:
            digest = hashlib.sha1("\x1f".join(members).encode()).hexdigest()
            tx.execute(
                "INSERT INTO communities(id, title, summary, findings, rating, members_hash, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (index, title, summary, findings, rating, digest, source),
            )
            tx.executemany(
                "INSERT INTO community_members(community_id, name) VALUES (?, ?)",
                [(index, name) for name in members],
            )
            if vector is not None:
                vectors.upsert(tx, "community", str(index), vector)

    return len(rows)


def _summarize(llm, index: int, members: list[str], conn: sqlite3.Connection) -> Community:
    title = f"Cluster {index + 1}: " + ", ".join(members[:3])
    summary = f"{len(members)} related entities centered on {', '.join(members[:3])}."
    findings = [f"Covers {', '.join(members[:3])}."]
    rating = 5.0

    if llm is not None:
        placeholders = ",".join("?" for _ in members)
        entities = "\n".join(
            f"- {row['name']} ({row['type']}): {row['description']}"
            for row in conn.execute(
                f"SELECT name, type, description FROM entities WHERE name IN ({placeholders})",
                members,
            )
        )
        relations = "\n".join(
            f"- {row['src']} -[{row['rel']}]-> {row['dst']}: {row['description']}"
            for row in conn.execute(
                f"SELECT src, dst, rel, description FROM edges "
                f"WHERE src IN ({placeholders}) AND dst IN ({placeholders})",
                [*members, *members],
            )
        ) or "None"
        data = llm.generate_json(
            SUMMARY_USER.format(entities=entities, relations=relations[:4000]),
            SUMMARY_SYSTEM,
        )
        if isinstance(data, dict):
            title = str(data.get("title") or title)[:120]
            summary = str(data.get("summary") or summary)
            raw_findings = data.get("findings")
            if isinstance(raw_findings, list):
                findings = [str(item) for item in raw_findings if str(item).strip()][:8] or findings
            try:
                rating = float(data.get("rating", rating))
            except (TypeError, ValueError):
                pass

    return Community(
        id=index, title=title, summary=summary, findings=findings, rating=rating, members=members
    )
