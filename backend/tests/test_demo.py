"""The demo seed replays the recording: no API calls, no stale prompts."""

from graph_rag.demo import seed_demo


def test_seed_builds_a_queryable_graph_without_llm_calls(db, embed):
    seeded = seed_demo(db, embed)
    assert seeded == 3
    counts = {
        table: db.conn().execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        for table in ("documents", "chunks", "entities", "edges", "communities")
    }
    assert counts["documents"] == 3
    assert counts["entities"] > 5
    assert counts["edges"] > 5
    assert counts["communities"] >= 1


def test_seed_is_a_noop_when_documents_exist(db, embed):
    assert seed_demo(db, embed) == 3
    assert seed_demo(db, embed) == 0
