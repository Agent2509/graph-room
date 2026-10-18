"""Global search: rank community reports by similarity, then synthesize."""

import json
import re
import sqlite3

from graph_rag.pipeline import vectors

SYSTEM = """You are an analyst answering broad, thematic questions from pre-written
community reports. Structure the answer with clear themes. Cite the communities you use
as [Community N: title]. If the reports do not cover the question, say so.
Write plain prose: no markdown bullets, asterisks, or headings."""

USER = """Question: {question}

--- COMMUNITY REPORTS ---
{reports}
--- END ---

Synthesize an answer grounded in these reports:"""


def prepare(conn: sqlite3.Connection, embed, question: str, top_k_communities: int = 5) -> dict:
    ranked = [
        ref for ref, _ in vectors.knn(conn, "community", embed.one(question), top_k_communities)
    ]
    communities = []
    for ref in ranked:
        row = conn.execute(
            "SELECT id, title, summary, findings FROM communities WHERE id=?", (int(ref),)
        ).fetchone()
        if row:
            communities.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "findings": json.loads(row["findings"]),
                }
            )

    if not communities:
        return {
            "mode": "global",
            "system": SYSTEM,
            "prompt": "",
            "entities": [],
            "edges": [],
            "chunks": [],
            "context": "No community reports are available yet. Index a document first.",
        }

    def clean_title(title: str) -> str:
        # heuristic summaries are titled "Cluster 4: ..."; the citation already
        # prefixes the community id.
        return re.sub(r"^Cluster\s+\d+:\s*", "", title)

    reports = "\n\n".join(
        f"[Community {c['id']}: {clean_title(c['title'])}]\n"
        f"Summary: {c['summary']}\n"
        f"Findings:\n" + "\n".join(f"  * {finding}" for finding in c["findings"])
        for c in communities
    )
    return {
        "mode": "global",
        "system": SYSTEM,
        "prompt": USER.format(question=question, reports=reports),
        "entities": [],
        "edges": [],
        "chunks": [],
        "context": reports,
    }
