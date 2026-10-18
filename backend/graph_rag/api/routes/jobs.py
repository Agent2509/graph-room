"""Job status polling."""

from fastapi import APIRouter, Depends, HTTPException, Request

from graph_rag.api.deps import require_token
from graph_rag.jobs import runner as jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(require_token)])


@router.get("")
def list_jobs(request: Request, limit: int = 8) -> list[dict]:
    rows = request.app.state.db.conn().execute(
        "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in rows]


@router.get("/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    job = jobs.get(request.app.state.db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
