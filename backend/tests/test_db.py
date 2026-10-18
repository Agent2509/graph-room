"""Storage: FTS triggers, cascade deletes, vector KNN filters."""

from graph_rag.pipeline import vectors
from graph_rag.pipeline.indexer import add_document


def _ingest(db, text="Alice Zhao founded NexusLabs in Boston.", name="a.txt"):
    return add_document(db, name, text, 200, 20)


def test_chunks_are_content_addressed(db):
    first, count = _ingest(db)
    second, again = _ingest(db)
    assert first == second
    assert count == 1
    assert again == 0


def test_fts_follows_chunk_lifecycle(db):
    add_document(db, "a.txt", "NexusLabs builds CortexGraph in Boston.", 200, 20)
    conn = db.conn()
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_fts WHERE chunks_fts MATCH 'nexuslabs'"
    ).fetchone()["c"] == 1

    document_id = conn.execute("SELECT id FROM documents").fetchone()["id"]
    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
    assert conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_fts WHERE chunks_fts MATCH 'nexuslabs'"
    ).fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"] == 0


def test_vector_rows_removed_with_document(db):
    from graph_rag.pipeline.graph import delete_document

    document_id, _ = _ingest(db)
    conn = db.conn()
    chunk_id = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    vectors.upsert(conn, "chunk", chunk_id, [1.0] * 384)

    delete_document(db, document_id)
    assert conn.execute("SELECT COUNT(*) AS c FROM vec_items").fetchone()["c"] == 0


def test_knn_respects_kind_and_orders_by_distance(db):
    conn = db.conn()
    vectors.upsert(conn, "chunk", "near", [1.0, 0.0, *([0.0] * 382)])
    vectors.upsert(conn, "chunk", "far", [0.0, 1.0, *([0.0] * 382)])
    vectors.upsert(conn, "entity", "ignored", [1.0, 0.0, *([0.0] * 382)])

    hits = vectors.knn(conn, "chunk", [1.0, 0.0, *([0.0] * 382)], k=5)
    assert [ref for ref, _ in hits] == ["near", "far"]

    vectors.delete(conn, "chunk", "near")
    assert [ref for ref, _ in vectors.knn(conn, "chunk", [1.0, 0.0] + [0.0] * 382, 5)] == ["far"]
