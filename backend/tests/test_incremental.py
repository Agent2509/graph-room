"""Per-document CRUD must leave no ghosts: entities, edges, vectors, communities."""

from graph_rag.llm.mock import MockLLM
from graph_rag.pipeline import indexer
from graph_rag.pipeline.graph import delete_document
from graph_rag.retrieval import service

ALICE_DOC = "Alice Zhao founded NexusLabs in Boston. NexusLabs builds CortexGraph."
CURE_DOC = "DeepBio develops CureX for oncology. DeepBio is based in Zurich."


def test_delete_removes_only_that_documents_entities(db, embed):
    llm = MockLLM()
    first, _ = indexer.add_document(db, "a.txt", ALICE_DOC, 300, 30)
    second, _ = indexer.add_document(db, "b.txt", CURE_DOC, 300, 30)
    indexer.index_document(db, llm, embed, "mock", first, batch_size=1)
    indexer.index_document(db, llm, embed, "mock", second, batch_size=1)

    before = {row["name"] for row in db.conn().execute("SELECT name FROM entities")}
    assert "NexusLabs" in before and "DeepBio" in before

    delete_document(db, first)
    indexer.reindex_communities(db, llm=None, embed=embed)

    names = {row["name"] for row in db.conn().execute("SELECT name FROM entities")}
    assert "DeepBio" in names
    assert "NexusLabs" not in names

    stale = [
        row["name"]
        for row in db.conn().execute(
            "SELECT name FROM community_members WHERE name='NexusLabs'"
        )
    ]
    assert stale == []
    assert db.conn().execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"] > 0


def test_query_after_delete_never_cites_removed_entity(db, embed):
    llm = MockLLM()
    first, _ = indexer.add_document(db, "a.txt", ALICE_DOC, 300, 30)
    second, _ = indexer.add_document(db, "b.txt", CURE_DOC, 300, 30)
    indexer.index_document(db, llm, embed, "mock", first, batch_size=1)
    indexer.index_document(db, llm, embed, "mock", second, batch_size=1)
    delete_document(db, first)
    indexer.reindex_communities(db, llm=None, embed=embed)

    result = service.answer(db, llm, embed, "What is NexusLabs?", mode="local", max_hops=2)
    cited = {entity["name"] for entity in result.entities}
    assert "NexusLabs" not in cited
