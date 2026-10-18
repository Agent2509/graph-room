"""Dense vectors in the same SQLite file (sqlite-vec).

Vectors are L2-normalized on write and read, so vec0's L2 distance ranks exactly
like cosine similarity.

shortcut: vec0 is brute force, ~40ms per 100k chunks on one core. Past ~1M chunks
(roughly 400ms, 1.6GB) swap this module for LanceDB. Keep the four functions.
"""

import sqlite3

import numpy as np
from sqlite_vec import serialize_float32


def _blob(vector: list[float]) -> bytes:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm:
        array = array / norm
    return serialize_float32(array.tolist())


def upsert(conn: sqlite3.Connection, kind: str, ref: str, vector: list[float]) -> None:
    conn.execute("DELETE FROM vec_items WHERE kind=? AND ref=?", (kind, ref))
    conn.execute(
        "INSERT INTO vec_items(kind, ref, embedding) VALUES (?, ?, ?)",
        (kind, ref, _blob(vector)),
    )


def delete(conn: sqlite3.Connection, kind: str, ref: str) -> None:
    conn.execute("DELETE FROM vec_items WHERE kind=? AND ref=?", (kind, ref))


def knn(
    conn: sqlite3.Connection, kind: str, vector: list[float], k: int = 10
) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT ref, distance FROM vec_items "
        "WHERE embedding MATCH ? AND k = ? AND kind = ? ORDER BY distance",
        (_blob(vector), k, kind),
    )
    return [(row["ref"], float(row["distance"])) for row in rows]
