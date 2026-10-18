"""Query endpoint, streaming and not."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from graph_rag.api.deps import get_answer_llm, require_token
from graph_rag.retrieval import service

router = APIRouter(prefix="/api", tags=["query"], dependencies=[Depends(require_token)])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["auto", "local", "global"] = "auto"
    max_hops: int = Field(default=2, ge=1, le=3)
    document_id: str | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream(state, body: QueryRequest, llm) -> AsyncIterator[str]:
    try:
        async for event, data in service.stream_answer(
            state.db, llm, state.embed, body.question, body.mode, body.max_hops
        ):
            yield _sse(event, data)
    except Exception as error:  # noqa: BLE001 - the browser must see a real message
        yield _sse("error", {"message": str(error)[:300]})
        yield _sse("done", {})


@router.post("/query")
async def run_query(
    request: Request, body: QueryRequest, stream: bool = False, llm=Depends(get_answer_llm)
):
    state = request.app.state
    if stream:
        return StreamingResponse(
            _stream(state, body, llm), media_type="text/event-stream", headers=SSE_HEADERS
        )
    result = await asyncio.to_thread(
        service.answer,
        state.db,
        llm,
        state.embed,
        body.question,
        body.mode,
        body.max_hops,
        body.document_id,
    )
    return result.model_dump()
