"""Background job runner: asyncio task, SQLite state, one writer at a time."""

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from graph_rag.db import Database

FIELDS = {"status", "phase", "done", "total", "message", "error"}


def update(db: Database, job_id: str, **fields) -> None:
    values = {key: value for key, value in fields.items() if key in FIELDS}
    if not values:
        return
    values["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    assignments = ", ".join(f"{key}=?" for key in values)
    db.conn().execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), job_id))


def get(db: Database, job_id: str) -> dict | None:
    row = db.conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


class JobRunner:
    def __init__(self, db: Database, max_concurrent: int = 1):
        self.db = db
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task] = set()

    def submit(self, kind: str, work: Callable[[str], str]) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.db.conn().execute(
            "INSERT INTO jobs(id, kind, status) VALUES (?, ?, 'queued')", (job_id, kind)
        )
        task = asyncio.create_task(self._run(job_id, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def _run(self, job_id: str, work: Callable[[str], str]) -> None:
        async with self._semaphore:
            update(self.db, job_id, status="running")
            try:
                status = await asyncio.to_thread(work, job_id)
                update(self.db, job_id, status=status or "done")
            except Exception as error:  # noqa: BLE001 - job errors belong in the row
                update(self.db, job_id, status="failed", error=str(error)[:800])

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
