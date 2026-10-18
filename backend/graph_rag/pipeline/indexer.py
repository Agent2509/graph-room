"""Ingestion and indexing.

Chunking happens at upload time (fast, synchronous); extraction, embedding and
community summaries happen in a background job. Every step is idempotent, so a
failed job is resumed by simply running it again.

Embeddings and LLM calls are computed before a write transaction opens. Doing slow
work under BEGIN IMMEDIATE would freeze every other writer in the app.
"""

import hashlib
from collections.abc import Callable

from graph_rag.db import Database
from graph_rag.llm import BaseLLM
from graph_rag.models import TextChunk

from . import communities, extractor, fast_extract, graph, resolver, vectors

Progress = Callable[[str, int, int], None]


def add_document(
    db: Database, filename: str, text: str, chunk_size: int, chunk_overlap: int
) -> tuple[str, int]:
    """Store a document and its chunks. Content-addressed: the same text is a no-op."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    document_id = digest[:16]
    conn = db.conn()
    existing = conn.execute(
        "SELECT id FROM documents WHERE content_sha=?", (digest,)
    ).fetchone()
    if existing:
        return existing["id"], 0

    from .chunker import chunk_text

    chunks = chunk_text(text, document_id, chunk_size, chunk_overlap)
    with db.tx() as tx:
        tx.execute(
            "INSERT INTO documents(id, filename, content_sha, char_count) VALUES (?, ?, ?, ?)",
            (document_id, filename, digest, len(text)),
        )
        tx.executemany(
            "INSERT INTO chunks(id, document_id, idx, text, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            [(c.id, c.document_id, c.idx, c.text, c.token_count) for c in chunks],
        )
    return document_id, len(chunks)


def index_document(
    db: Database,
    llm: BaseLLM,
    embed,
    model: str,
    document_id: str,
    batch_size: int = 3,
    max_llm_summaries: int = 12,
    progress: Progress | None = None,
    mode: str = "llm",
    relation_threshold: float = 0.62,
) -> str:
    def report(phase: str, done: int, total: int) -> None:
        if progress:
            progress(phase, done, total)

    conn = db.conn()
    chunks = [
        TextChunk(
            id=row["id"],
            document_id=row["document_id"],
            idx=row["idx"],
            text=row["text"],
            token_count=row["token_count"],
        )
        for row in conn.execute(
            "SELECT id, document_id, idx, text, token_count FROM chunks "
            "WHERE document_id=? ORDER BY idx",
            (document_id,),
        )
    ]
    if not chunks:
        return "done"

    report("extracting", 0, max(1, len(chunks) // batch_size))
    if mode == "fast":
        # Deterministic, no LLM, no cache: the work is milliseconds.
        raw_entities, raw_edges = fast_extract.extract(
            chunks, embed, threshold=relation_threshold
        )
        report("extracting", 1, 1)
    else:
        raw_entities, raw_edges = extractor.extract(
            db, llm, model, chunks, batch_size,
            progress=lambda done, total: report("extracting", done, total),
        )
    if not raw_entities:
        raise RuntimeError("Extraction produced no entities; check the LLM response")

    resolved = resolver.resolve_entities(raw_entities)
    resolved_edges = resolver.resolve_edges(raw_edges, resolved)
    with db.tx() as tx:
        graph.write_extraction(tx, resolved, resolved_edges)
        graph.write_cooccurrence(tx, document_id)

    total_vectors = len(chunks) + len(resolved)
    done = 0
    for start in range(0, len(chunks), 64):
        window = chunks[start : start + 64]
        batch_vectors = embed([chunk.text for chunk in window])
        with db.tx() as tx:
            for chunk, vector in zip(window, batch_vectors, strict=False):
                vectors.upsert(tx, "chunk", chunk.id, vector)
        done += len(window)
        report("embedding", done, total_vectors)

    entity_names = [entity.name for entity in resolved.values()]
    for start in range(0, len(entity_names), 64):
        window = entity_names[start : start + 64]
        rows = [
            row
            for row in (
                conn.execute("SELECT name, description FROM entities WHERE name=?", (name,)).fetchone()
                for name in window
            )
            if row
        ]
        batch_vectors = embed([f"{row['name']}: {row['description']}" for row in rows])
        with db.tx() as tx:
            for row, vector in zip(rows, batch_vectors, strict=False):
                vectors.upsert(tx, "entity", row["name"], vector)
        done += len(window)
        report("embedding", min(done, total_vectors), total_vectors)

    report("communities", 0, 1)
    # Heuristic summaries only: the job finishes as soon as the graph is queryable.
    # A follow-up "communities" job upgrades them with the LLM.
    communities.rebuild(db, llm=None, embed=embed)
    report("communities", 1, 1)
    return "done"


def reindex_communities(db: Database, llm, embed, max_llm_summaries: int = 12) -> int:
    return communities.rebuild(db, llm=llm, embed=embed, max_llm=max_llm_summaries)
