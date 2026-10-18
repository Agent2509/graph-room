"""Job runner: success, failure, and resume-from-cache behavior."""

import asyncio

from graph_rag.jobs import runner as jobs
from graph_rag.jobs.runner import JobRunner
from graph_rag.llm.mock import MockLLM


async def _wait(db, job_id, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = jobs.get(db, job_id)
        if job["status"] in {"done", "failed"}:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {jobs.get(db, job_id)}")


async def test_job_completes(db):
    runner = JobRunner(db)
    job_id = runner.submit("index", lambda job_id: "done")
    assert (await _wait(db, job_id))["status"] == "done"
    await runner.shutdown()


async def test_job_failure_is_recorded(db):
    runner = JobRunner(db)

    def explode(job_id):
        raise RuntimeError("llm quota exceeded")

    job_id = runner.submit("index", explode)
    job = await _wait(db, job_id)
    assert job["status"] == "failed"
    assert "quota exceeded" in job["error"]
    await runner.shutdown()


async def test_progress_updates_reach_the_row(db):
    runner = JobRunner(db)

    def work(job_id):
        jobs.update(db, job_id, phase="extracting", done=2, total=5, message="extracting 2/5")
        return "done"

    job_id = runner.submit("index", work)
    job = await _wait(db, job_id)
    assert job["done"] == 2 and job["total"] == 5
    await runner.shutdown()


def test_failed_extraction_is_not_cached_and_retry_succeeds(db, embed):
    """A batch that raises must leave no cache row, so the retry re-does work."""
    from graph_rag.models import TextChunk
    from graph_rag.pipeline import extractor
    from graph_rag.pipeline.indexer import add_document

    document_id, _ = add_document(db, "a.txt", "Alice founded NexusLabs.", 500, 50)
    chunk = TextChunk(id="x#0000", document_id=document_id, idx=0,
                      text="Alice founded NexusLabs.", token_count=3)

    class Flaky(MockLLM):
        calls = 0

        def generate_json(self, prompt, system=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("rate limited")
            return super().generate_json(prompt, system)

    import pytest

    with pytest.raises(RuntimeError):
        extractor.extract(db, Flaky(), "flaky", [chunk], batch_size=1)
    entities, _ = extractor.extract(db, Flaky(), "flaky", [chunk], batch_size=1)
    assert entities


def test_embedding_never_runs_inside_a_write_transaction(db):
    """A slow embed must not hold BEGIN IMMEDIATE: that froze every other writer."""
    import numpy as np

    from graph_rag.pipeline.indexer import add_document, index_document

    class LockProbe:
        model_name = "probe"

        def __call__(self, texts):
            conn = db.conn()
            conn.execute("BEGIN IMMEDIATE")  # raises if a write tx is already open
            conn.execute("ROLLBACK")
            out = []
            for _ in texts:
                vector = np.ones(384, dtype=np.float32) / np.sqrt(384)
                out.append(vector.tolist())
            return out

        def one(self, text):
            return self([text])[0]

    document_id, _ = add_document(db, "a.txt", "Alice founded NexusLabs.", 500, 50)
    index_document(db, MockLLM(), LockProbe(), "mock", document_id, batch_size=1,
                   max_llm_summaries=0)
    assert db.conn().execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"] > 0


def test_index_job_does_not_spend_llm_calls_on_summaries(db, embed):
    """Indexing must finish after extraction; summaries are a separate job."""
    from graph_rag.pipeline.indexer import add_document, index_document

    class Counting(MockLLM):
        calls = 0

        def generate_json(self, prompt, system=None):
            Counting.calls += 1
            return super().generate_json(prompt, system)

    document_id, _ = add_document(db, "a.txt", "Alice founded NexusLabs.", 500, 50)
    index_document(db, Counting(), embed, "mock", document_id, batch_size=3)
    assert Counting.calls == 1
