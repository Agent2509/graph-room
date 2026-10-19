"""Local search: entity seeds -> bounded subgraph -> fused chunk context."""

import sqlite3

from graph_rag.pipeline import vectors
from graph_rag.pipeline.graph import subgraph
from graph_rag.retrieval import bm25, fusion

SYSTEM = """You answer questions using ONLY the supplied knowledge graph context and source text.

Cite evidence inline:
- entities as [Entity: Name]
- graph edges as [Relation: Source -> REL -> Target]
- source chunks as [Chunk: chunk_id]

Describe the intermediate hops when explaining a connection. If the context does not
support an answer, say what is missing instead of guessing.
Write plain prose: no markdown bullets, asterisks, or headings."""

USER = """Question: {question}

--- ENTITIES ---
{entities}

--- RELATIONSHIPS ---
{relations}

--- SOURCE CHUNKS ---
{chunks}
--- END ---

Answer with inline citations:"""


def _no_evidence() -> dict:
    return {
        "mode": "local",
        "system": SYSTEM,
        "prompt": "",
        "entities": [],
        "edges": [],
        "chunks": [],
        "context": "No evidence in the corpus for that question. "
        "Try another wording, another document, or add one.",
    }


def prepare(
    conn: sqlite3.Connection,
    embed,
    question: str,
    top_k_entities: int = 5,
    max_hops: int = 2,
    top_k_chunks: int = 5,
    max_nodes: int = 40,
    document_id: str | None = None,
    use_graph: bool = True,
) -> dict:
    allowed: set[str] | None = None
    if document_id:
        allowed = {
            row["id"]
            for row in conn.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))
        }
        if not allowed:
            return _no_evidence()

    question_vector = embed.one(question)
    entities: list[dict] = []
    edges: list[dict] = []
    mention_ids: list[str] = []

    if use_graph:
        seeds = [ref for ref, _ in vectors.knn(conn, "entity", question_vector, top_k_entities)]

        lowered = question.lower()
        for row in conn.execute("SELECT name FROM entities"):
            if row["name"].lower() in lowered and row["name"] not in seeds:
                seeds.append(row["name"])

        entities, edges = subgraph(conn, seeds[:8], max_hops=max_hops, max_nodes=max_nodes)
        names = [entity["name"] for entity in entities]
        if names:
            placeholders = ",".join("?" for _ in names)
            mention_ids = [
                row["chunk_id"]
                for row in conn.execute(
                    f"SELECT DISTINCT chunk_id FROM entity_mentions WHERE name IN ({placeholders})",
                    names,
                )
            ]

    dense_ids = [
        ref for ref, _ in vectors.knn(conn, "chunk", question_vector, max(top_k_chunks * 6, 60))
    ]
    bm25_ids = bm25.search(conn, question, max(top_k_chunks * 6, 60))
    if allowed is not None:
        mention_ids = [cid for cid in mention_ids if cid in allowed]
        bm25_ids = [cid for cid in bm25_ids if cid in allowed]
        dense_ids = [cid for cid in dense_ids if cid in allowed]
    fused = fusion.rrf([mention_ids, bm25_ids, dense_ids])
    chosen = fused[:top_k_chunks]

    chunks = []
    if chosen:
        placeholders = ",".join("?" for _ in chosen)
        by_id = {
            row["id"]: dict(row)
            for row in conn.execute(
                f"SELECT id, document_id, text FROM chunks WHERE id IN ({placeholders})", chosen
            )
        }
        chunks = [by_id[chunk_id] for chunk_id in chosen if chunk_id in by_id]

    if not entities and not chunks:
        return _no_evidence()

    entities_text = (
        "\n".join(f"- {e['name']} ({e['type']}): {e['description']}" for e in entities)
        or "None matched."
    )
    relations_text = (
        "\n".join(
            f"- ({e['src']}) --[{e['rel']}]--> ({e['dst']}): {e['description']}" for e in edges
        )
        or "None."
    )
    chunks_text = (
        "\n\n".join(f"[Chunk {c['id']}]\n{c['text']}" for c in chunks)
        or "No source chunks available."
    )

    return {
        "mode": "local",
        "system": SYSTEM,
        "prompt": USER.format(
            question=question,
            entities=entities_text,
            relations=relations_text,
            chunks=chunks_text,
        ),
        "entities": entities,
        "edges": edges,
        "chunks": chunks,
        "context": f"Entities:\n{entities_text}\n\nRelationships:\n{relations_text}",
    }
