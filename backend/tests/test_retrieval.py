"""BM25, RRF, and the two retrieval modes."""

from graph_rag.models import ExtractedEdge, ExtractedEntity
from graph_rag.pipeline import vectors
from graph_rag.pipeline.graph import write_extraction
from graph_rag.pipeline.indexer import add_document
from graph_rag.retrieval import bm25, global_, local
from graph_rag.retrieval.fusion import rrf


def _ingest(db, embed, text, name="a.txt"):
    document_id, _ = add_document(db, name, text, 500, 50)
    conn = db.conn()
    chunk = conn.execute(
        "SELECT id, text FROM chunks WHERE document_id=?", (document_id,)
    ).fetchone()
    vectors.upsert(conn, "chunk", chunk["id"], embed.one(chunk["text"]))
    return document_id, chunk


def test_rrf_prefers_items_ranked_by_both_lists():
    assert rrf([["a", "b", "c"], ["b", "a"]])[0] in {"a", "b"}
    assert rrf([["x"]]) == ["x"]
    assert rrf([]) == []


def test_bm25_finds_keyword_and_survives_punctuation(db, embed):
    _ingest(db, embed, "CureX showed response rates in the oncology trial.")
    assert bm25.search(db.conn(), "curex") == [
        db.conn().execute("SELECT id FROM chunks").fetchone()["id"]
    ]
    assert bm25.search(db.conn(), "!!! ???") == []


def test_local_context_contains_seeded_entity(db, embed):
    document_id, chunk = _ingest(db, embed, "Alice Zhao founded NexusLabs in Boston.")
    conn = db.conn()
    write_extraction(
        conn,
        {
            "alice zhao": ExtractedEntity(
                name="Alice Zhao", type="PERSON", description="Founder.", chunk_ids=[chunk["id"]]
            )
        },
        [
            ExtractedEdge(
                src="Alice Zhao",
                dst="NexusLabs",
                rel="FOUNDED",
                description="Founded NexusLabs.",
                chunk_ids=[chunk["id"]],
            )
        ],
    )
    conn.execute("INSERT OR IGNORE INTO entities(name, type) VALUES ('NexusLabs', 'ORGANIZATION')")
    vectors.upsert(conn, "entity", "Alice Zhao", embed.one("Alice Zhao: Founder."))

    context = local.prepare(conn, embed, "Who is Alice Zhao?", max_hops=1)
    assert context["mode"] == "local"
    assert "Alice Zhao" in context["prompt"]
    assert context["chunks"]


def test_global_without_communities_is_explicit(db, embed):
    context = global_.prepare(db.conn(), embed, "What are the main themes?")
    assert context["prompt"] == ""
    assert "No community reports" in context["context"]


def test_empty_database_returns_no_evidence_without_calling_the_llm(db, embed):
    from graph_rag.retrieval import service

    class Explodes:
        model = "explodes"

        def generate(self, *args, **kwargs):
            raise AssertionError("the LLM must not be called without evidence")

        def generate_json(self, *args, **kwargs):
            raise AssertionError("the LLM must not be called without evidence")

        async def stream(self, *args, **kwargs):
            raise AssertionError("the LLM must not be called without evidence")
            yield ""  # pragma: no cover

    context = local.prepare(db.conn(), embed, "anything")
    assert context["entities"] == []
    assert context["prompt"] == ""
    assert "No evidence" in context["context"]

    result = service.answer(db, Explodes(), embed, "anything about unicorns?", mode="local")
    assert "No evidence" in result.answer


def test_document_scope_limits_chunks_to_that_document(db, embed):
    document_a, _ = _ingest(db, embed, "Alice founded NexusLabs in Boston.", "a.txt")
    _ingest(db, embed, "CureX treats tumors in Zurich.", "b.txt")

    scoped = local.prepare(db.conn(), embed, "Alice NexusLabs CureX Zurich", document_id=document_a)
    assert scoped["chunks"]
    assert all(chunk["id"].startswith(document_a) for chunk in scoped["chunks"])

    empty = local.prepare(db.conn(), embed, "anything", document_id="missing")
    assert empty["prompt"] == ""
    assert "No evidence" in empty["context"]
