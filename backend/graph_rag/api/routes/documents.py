"""Document upload, listing and deletion."""

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from graph_rag.api.deps import check_upload_size, get_extraction_llm, require_token
from graph_rag.jobs import runner as jobs
from graph_rag.pipeline import graph as graph_pipeline
from graph_rag.pipeline import indexer
from graph_rag.pipeline.parsers import text_from_bytes

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(require_token)])


def _index_work(request: Request, document_id: str, llm):
    state = request.app.state
    config = state.settings

    def work(job_id: str) -> str:
        def progress(phase: str, done: int, total: int) -> None:
            jobs.update(
                state.db, job_id, phase=phase, done=done, total=total,
                message=f"{phase} {done}/{total}",
            )

        return indexer.index_document(
            state.db,
            llm,
            state.embed,
            config.extraction_model,
            document_id,
            batch_size=config.extraction_batch_size,
            max_llm_summaries=config.max_llm_summaries,
            progress=progress,
        )

    return work


@router.get("")
def list_documents(request: Request) -> list[dict]:
    rows = request.app.state.db.conn().execute(
        "SELECT d.id, d.filename, d.char_count, d.created_at, COUNT(c.id) AS chunks "
        "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
        "GROUP BY d.id ORDER BY d.created_at DESC, d.id"
    )
    return [dict(row) for row in rows]


@router.post("")
async def create_document(
    request: Request,
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    llm=Depends(get_extraction_llm),
) -> dict:
    state = request.app.state
    content: str
    name: str

    if file is not None:
        cap = state.settings.max_upload_bytes
        raw = await file.read(cap + 1)
        check_upload_size(len(raw), cap)
        name = file.filename or "upload"
        content = text_from_bytes(raw, name)
    elif text and text.strip():
        name = filename or "pasted"
        content = text
    else:
        raise HTTPException(status_code=400, detail="Send a file or non-empty text")

    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the upload")

    document_id, chunks = await asyncio.to_thread(
        indexer.add_document,
        state.db,
        name,
        content,
        state.settings.chunk_size,
        state.settings.chunk_overlap,
    )
    if chunks == 0:
        already = state.db.conn().execute(
            "SELECT 1 FROM entity_mentions "
            "JOIN chunks ON chunks.id = entity_mentions.chunk_id "
            "WHERE chunks.document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone()
        if already:
            return {"document_id": document_id, "chunks": 0, "job_id": None,
                    "duplicate": True}
        # Content is stored but extraction never finished (failed job): retry it.

    job_id = state.runner.submit("index", _index_work(request, document_id, llm))
    state.runner.submit("communities", _summarize_work(request, llm))
    return {"document_id": document_id, "chunks": chunks, "job_id": job_id,
            "duplicate": False}


def _summarize_work(request: Request, llm):
    state = request.app.state

    def work(job_id: str) -> str:
        indexer.reindex_communities(
            state.db, llm=llm, embed=state.embed,
            max_llm_summaries=state.settings.max_llm_summaries,
        )
        return "done"

    return work


@router.delete("/{document_id}")
async def delete_document(request: Request, document_id: str, llm=Depends(get_extraction_llm)) -> dict:
    state = request.app.state
    exists = state.db.conn().execute(
        "SELECT 1 FROM documents WHERE id=?", (document_id,)
    ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Document not found")

    removed = await asyncio.to_thread(graph_pipeline.delete_document, state.db, document_id)

    # Communities are cheap to recompute and expensive to leave stale. Rebuild the
    # memberships now (heuristic summaries) and re-summarize in the background.
    await asyncio.to_thread(indexer.reindex_communities, state.db, None, state.embed)

    job_id = state.runner.submit("communities", _summarize_work(request, llm))
    return {**removed, "job_id": job_id}
