"""One entry point for answering, streamed or not."""

import asyncio
from collections.abc import AsyncIterator

from graph_rag.db import Database
from graph_rag.llm import BaseLLM
from graph_rag.models import RetrievalResult
from graph_rag.retrieval import global_, local, router


def prepare(
    db: Database,
    llm: BaseLLM | None,
    embed,
    question: str,
    mode: str = "auto",
    max_hops: int = 2,
    document_id: str | None = None,
    use_graph: bool = True,
) -> dict:
    conn = db.conn()
    if document_id and mode != "global":
        mode = "local"  # a scoped question is always about one document
    if mode not in {"local", "global"}:
        has_communities = conn.execute("SELECT COUNT(*) AS c FROM communities").fetchone()["c"] > 0
        mode = router.decide(llm, question, has_communities)
    if mode == "global":
        return global_.prepare(conn, embed, question)
    return local.prepare(
        conn, embed, question, max_hops=max_hops, document_id=document_id, use_graph=use_graph
    )


def answer(
    db: Database,
    llm: BaseLLM,
    embed,
    question: str,
    mode: str = "auto",
    max_hops: int = 2,
    document_id: str | None = None,
) -> RetrievalResult:
    context = prepare(db, llm, embed, question, mode, max_hops, document_id)
    text = llm.generate(context["prompt"], context["system"]) if context["prompt"] else ""
    return RetrievalResult(
        query=question,
        mode=context["mode"],
        answer=text or context["context"],
        entities=context["entities"],
        edges=context["edges"],
        chunks=context["chunks"],
        context=context["context"],
    )


async def stream_answer(
    db: Database,
    llm: BaseLLM,
    embed,
    question: str,
    mode: str = "auto",
    max_hops: int = 2,
    document_id: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    context = await asyncio.to_thread(
        prepare, db, llm, embed, question, mode, max_hops, document_id
    )
    yield (
        "meta",
        {
            "mode": context["mode"],
            "entities": context["entities"],
            "edges": context["edges"],
            "chunks": [{"id": c["id"], "document_id": c["document_id"]} for c in context["chunks"]],
        },
    )
    if not context["prompt"]:
        yield "delta", {"text": context["context"]}
    else:
        async for delta in llm.stream(context["prompt"], context["system"]):
            yield "delta", {"text": delta}
    yield "done", {}
