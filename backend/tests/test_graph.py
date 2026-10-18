"""Graph writes are idempotent; deletes are exact; traversal is bounded."""

from graph_rag.models import ExtractedEdge, ExtractedEntity
from graph_rag.pipeline import graph as graph_pipeline
from graph_rag.pipeline import vectors
from graph_rag.pipeline.indexer import add_document


def _seed(db, filename, text):
    document_id, _ = add_document(db, filename, text, 500, 50)
    return document_id, db.conn().execute(
        "SELECT id FROM chunks WHERE document_id=? ORDER BY idx LIMIT 1", (document_id,)
    ).fetchone()["id"]


def _write(conn, chunk_id, entities, edges):
    resolved = {
        entity.name.lower(): ExtractedEntity(
            name=entity.name,
            type=entity.type,
            description=entity.description,
            chunk_ids=[chunk_id],
        )
        for entity in entities
    }
    graph_pipeline.write_extraction(
        conn,
        resolved,
        [
            ExtractedEdge(
                src=edge.src,
                dst=edge.dst,
                rel=edge.rel,
                description=edge.description,
                weight=edge.weight,
                chunk_ids=[chunk_id],
            )
            for edge in edges
        ],
    )


def test_write_is_idempotent(db):
    doc, chunk = _seed(db, "a.txt", "Alice founded NexusLabs.")
    entities = [ExtractedEntity(name="Alice"), ExtractedEntity(name="NexusLabs")]
    edges = [ExtractedEdge(src="Alice", dst="NexusLabs", rel="FOUNDED", weight=1.0)]
    _write(db.conn(), chunk, entities, edges)
    _write(db.conn(), chunk, entities, edges)
    assert db.conn().execute("SELECT weight FROM edges").fetchone()["weight"] == 1.0
    nodes = graph_pipeline.graph_payload(db.conn())["nodes"]
    assert len([node for node in nodes if node["type"] != "DOCUMENT"]) == 2


def test_delete_recomputes_entity_and_drops_orphans(db):
    doc_a, chunk_a = _seed(db, "a.txt", "Alice founded NexusLabs. Alice works in Boston.")
    doc_b, chunk_b = _seed(db, "b.txt", "Alice advises DeepBio in Zurich.")

    _write(
        db.conn(),
        chunk_a,
        [
            ExtractedEntity(name="Alice", description="Founder of NexusLabs."),
            ExtractedEntity(name="NexusLabs", description="A lab."),
        ],
        [ExtractedEdge(src="Alice", dst="NexusLabs", rel="FOUNDED", description="Founded it.")],
    )
    _write(
        db.conn(),
        chunk_b,
        [
            ExtractedEntity(name="Alice", description="Advisor to DeepBio."),
            ExtractedEntity(name="DeepBio", description="A biotech."),
        ],
        [ExtractedEdge(src="Alice", dst="DeepBio", rel="ADVISES", description="Advises it.")],
    )

    removed = graph_pipeline.delete_document(db, doc_a)
    assert removed["chunks_removed"] == 1

    names = {row["name"] for row in db.conn().execute("SELECT name FROM entities")}
    assert names == {"Alice", "DeepBio"}
    alice = db.conn().execute("SELECT description FROM entities WHERE name='Alice'").fetchone()
    assert "Advisor" in alice["description"]
    assert "Founder of NexusLabs" not in alice["description"]
    assert db.conn().execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"] == 1
    assert db.conn().execute("SELECT COUNT(*) AS c FROM vec_items").fetchone()["c"] == 0


def test_subgraph_is_bounded(db):
    doc, chunk = _seed(db, "a.txt", "A B C D")
    entities = [ExtractedEntity(name=name) for name in ["A", "B", "C", "D"]]
    edges = [
        ExtractedEdge(src="A", dst="B", rel="LINKS"),
        ExtractedEdge(src="B", dst="C", rel="LINKS"),
        ExtractedEdge(src="C", dst="D", rel="LINKS"),
    ]
    _write(db.conn(), chunk, entities, edges)

    one_hop, _ = graph_pipeline.subgraph(db.conn(), ["A"], max_hops=1)
    assert {node["name"] for node in one_hop} == {"A", "B"}

    two_hops, _ = graph_pipeline.subgraph(db.conn(), ["A"], max_hops=2)
    assert {node["name"] for node in two_hops} == {"A", "B", "C"}

    capped, _ = graph_pipeline.subgraph(db.conn(), ["A"], max_hops=3, max_nodes=2)
    assert len(capped) == 2


def test_vectors_are_removed_on_delete(db):
    doc, chunk = _seed(db, "a.txt", "Alice founded NexusLabs.")
    vectors.upsert(db.conn(), "chunk", chunk, [1.0] * 384)
    _write(db.conn(), chunk, [ExtractedEntity(name="Alice")], [])
    graph_pipeline.delete_document(db, doc)
    assert db.conn().execute("SELECT COUNT(*) AS c FROM vec_items").fetchone()["c"] == 0


def test_payload_links_documents_to_entities(db):
    from graph_rag.pipeline.indexer import add_document

    document_id, _ = add_document(db, "a.txt", "Alice Zhao founded NexusLabs.", 500, 50)
    chunk_id = db.conn().execute("SELECT id FROM chunks").fetchone()["id"]
    _write(db.conn(), chunk_id, [ExtractedEntity(name="Alice Zhao")], [])

    payload = graph_pipeline.graph_payload(db.conn())
    doc_nodes = [node for node in payload["nodes"] if node["type"] == "DOCUMENT"]
    mention_edges = [edge for edge in payload["edges"] if edge["kind"] == "mention"]
    assert doc_nodes and doc_nodes[0]["id"] == f"doc:{document_id}"
    assert doc_nodes[0]["mentions"] == 1
    assert mention_edges[0]["src"] == f"doc:{document_id}"
    assert mention_edges[0]["dst"] == "Alice Zhao"
