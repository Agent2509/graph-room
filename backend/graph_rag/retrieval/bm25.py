"""Sparse retrieval with SQLite FTS5's built-in BM25. No custom scoring code."""

import re
import sqlite3


def search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[str]:
    """Return chunk ids ranked by BM25, best first."""
    tokens = [token for token in re.findall(r"\w+", query.lower()) if len(token) > 1]
    if not tokens:
        return []
    # Quote every token: FTS5 treats input as a query language otherwise.
    match = " OR ".join(f'"{token}"' for token in tokens)
    rows = conn.execute(
        "SELECT chunks.id FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, limit),
    )
    return [row["id"] for row in rows]
