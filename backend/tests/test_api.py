"""HTTP surface: auth, size cap, honest status, streaming."""

import time

from fastapi.testclient import TestClient

from graph_rag.api.app import create_app
from graph_rag.config import Settings
from graph_rag.llm.mock import MockLLM


def _wait(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_needs_no_token_and_reports_features(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["features"]["vectors"].startswith("sqlite-vec")
    assert body["features"]["answer_model"] != body["features"]["extraction_model"]


def test_paste_index_and_query(client):
    created = client.post(
        "/api/documents",
        data={"text": "Alice Zhao founded NexusLabs in Boston.", "filename": "note"},
    ).json()
    assert created["chunks"] == 1
    assert _wait(client, created["job_id"])["status"] == "done"

    duplicate = client.post(
        "/api/documents",
        data={"text": "Alice Zhao founded NexusLabs in Boston.", "filename": "note"},
    ).json()
    assert duplicate["duplicate"] is True and duplicate["job_id"] is None

    graph = client.get("/api/graph").json()
    assert graph["nodes"]

    result = client.post("/api/query", json={"question": "Who founded NexusLabs?"}).json()
    assert result["mode"] in {"local", "global"}
    assert result["answer"]

    with client.stream(
        "POST", "/api/query?stream=true", json={"question": "Who founded NexusLabs?"}
    ) as response:
        body = "".join(response.iter_text())
    assert "event: meta" in body and "event: done" in body


def test_upload_cap_returns_413(tmp_path, embed):
    app = create_app(
        Settings(data_dir=tmp_path / "d", static_dir=tmp_path / "s", max_upload_bytes=10, seed_demo=False),
        llm=MockLLM(),
        embed=embed,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/documents", files={"file": ("big.txt", b"x" * 64, "text/plain")}
        )
    assert response.status_code == 413


def test_token_auth(tmp_path, embed):
    app = create_app(
        Settings(data_dir=tmp_path / "d", static_dir=tmp_path / "s", api_token="s3cret", seed_demo=False),
        llm=MockLLM(),
        embed=embed,
    )
    with TestClient(app) as client:
        assert client.get("/api/stats").status_code == 401
        assert client.get("/api/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get(
            "/api/stats", headers={"Authorization": "Bearer s3cret"}
        ).status_code == 200
        assert client.get("/api/health").status_code == 200


def test_unknown_document_is_404(client):
    assert client.delete("/api/documents/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_failed_indexing_can_be_retried(client):
    """A document whose job failed (no mentions) must be indexable again."""
    body = {"text": "Alice founded NexusLabs.", "filename": "note"}
    first = client.post("/api/documents", data=body).json()
    assert _wait(client, first["job_id"])["status"] == "done"

    conn = client.app.state.db.conn()  # type: ignore[attr-defined]
    conn.executescript("DELETE FROM entity_mentions; DELETE FROM entities;")

    retry = client.post("/api/documents", data=body).json()
    assert retry["duplicate"] is False
    assert retry["job_id"]
    assert _wait(client, retry["job_id"])["status"] == "done"


def test_fast_mode_ingests_without_a_key_but_queries_do_not(tmp_path, embed):
    app = create_app(
        Settings(data_dir=tmp_path / "d", static_dir=tmp_path / "s", seed_demo=False), embed=embed
    )  # no injected llm, no server key, fast extraction (default)
    with TestClient(app) as client:
        created = client.post(
            "/api/documents", data={"text": "Alice Zhao founded NexusLabs.", "filename": "n"}
        )
        assert created.status_code == 200
        assert created.json()["job_id"]
        assert _wait(client, created.json()["job_id"])["status"] == "done"

        # answering still needs the visitor's key
        response = client.post("/api/query", json={"question": "who?"})
        assert response.status_code == 400
        assert "Groq API key" in response.json()["detail"]


def test_llm_extraction_mode_requires_a_key(tmp_path, embed):
    app = create_app(
        Settings(
            data_dir=tmp_path / "d2", static_dir=tmp_path / "s2",
            seed_demo=False, extraction_mode="llm",
        ),
        embed=embed,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/documents", data={"text": "Alice founded NexusLabs.", "filename": "n"}
        )
        assert response.status_code == 400
        assert "Groq API key" in response.json()["detail"]


def test_delete_document_while_a_job_is_running(client):
    first = client.post(
        "/api/documents", data={"text": "Alice founded NexusLabs.", "filename": "a"}
    ).json()
    assert _wait(client, first["job_id"])["status"] == "done"
    client.post("/api/documents", data={"text": "DeepBio develops CureX.", "filename": "b"})

    response = client.delete(f"/api/documents/{first['document_id']}")
    assert response.status_code == 200
    assert response.json()["chunks_removed"] == 1
    remaining = [row["filename"] for row in client.get("/api/documents").json()]
    assert remaining == ["b"]


def test_deleting_the_last_document_empties_every_count(client):
    created = client.post(
        "/api/documents", data={"text": "Alice Zhao founded NexusLabs.", "filename": "only"}
    ).json()
    assert _wait(client, created["job_id"])["status"] == "done"
    before = client.get("/api/stats").json()
    assert before["documents"] == 1 and before["entities"] > 0

    assert client.delete(f"/api/documents/{created['document_id']}").status_code == 200

    after = client.get("/api/stats").json()
    for key in ("documents", "chunks", "entities", "edges", "communities"):
        assert after[key] == 0, (key, after[key])
    assert client.get("/api/graph").json() == {"nodes": [], "edges": []}
