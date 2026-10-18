"""LLM extraction: batched, cached, and tolerant of malformed items.

A bad relationship is skipped, never allowed to kill a batch. Cache keys include
the model name, so switching models re-extracts instead of reusing stale triples.
"""

import hashlib
import json
from collections.abc import Callable

from pydantic import ValidationError

from graph_rag.db import Database
from graph_rag.llm import BaseLLM
from graph_rag.models import ExtractedEdge, ExtractedEntity, RawEdge, RawEntity, TextChunk

SYSTEM = """You are a knowledge graph extraction engine.
Return JSON: {"entities":[{"name","type","description"}],
"relationships":[{"source","target","relation_type","description","weight"}]}

Rules:
- Entity names are short and canonical ("Apple", not "the tech company Apple").
- type is one of PERSON, ORGANIZATION, TECHNOLOGY, PRODUCT, CONCEPT, LOCATION, EVENT.
- relationship endpoints must be entity names from your own list.
- relation_type is an uppercase verb phrase, e.g. FOUNDED, ACQUIRED, PARTNERS_WITH.
- weight is a number between 0.5 and 1.0.
- Only extract what the text supports. Do not invent entities."""

USER = """Extract entities and relationships from this text:

--- TEXT ---
{text}
--- END TEXT ---"""


def batch_key(model: str, texts: list[str]) -> str:
    digest = hashlib.sha256(model.encode())
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode())
    return digest.hexdigest()


def extract(
    db: Database,
    llm: BaseLLM,
    model: str,
    chunks: list[TextChunk],
    batch_size: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    entities: list[ExtractedEntity] = []
    edges: list[ExtractedEdge] = []
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    conn = db.conn()

    for index, batch in enumerate(batches, start=1):
        key = batch_key(model, [chunk.text for chunk in batch])
        cached = conn.execute("SELECT payload FROM extraction_cache WHERE key=?", (key,)).fetchone()
        if cached:
            payload = json.loads(cached["payload"])
        else:
            text = "\n\n".join(f"[Chunk {chunk.id}]\n{chunk.text}" for chunk in batch)
            payload = llm.generate_json(USER.format(text=text), SYSTEM)
            conn.execute(
                "INSERT OR REPLACE INTO extraction_cache(key, payload) VALUES (?, ?)",
                (key, json.dumps(payload)),
            )

        batch_entities, batch_edges = parse(payload, batch)
        entities.extend(batch_entities)
        edges.extend(batch_edges)
        if progress:
            progress(index, len(batches))

    return entities, edges


def parse(
    payload: object, batch: list[TextChunk]
) -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    """Validate item by item. One malformed relationship cannot lose a batch."""
    if not isinstance(payload, dict):
        return [], []

    texts = [(chunk.id, chunk.text.lower()) for chunk in batch]
    all_ids = [chunk.id for chunk in batch]
    entities: list[ExtractedEntity] = []
    edges: list[ExtractedEdge] = []

    for item in _as_list(payload.get("entities")):
        try:
            raw = RawEntity.model_validate(item)
        except ValidationError:
            continue
        chunk_ids = [cid for cid, text in texts if raw.name.lower() in text] or all_ids
        entities.append(
            ExtractedEntity(
                name=raw.name,
                type=raw.type.upper() or "CONCEPT",
                description=raw.description,
                chunk_ids=chunk_ids,
            )
        )

    for item in _as_list(payload.get("relationships")):
        try:
            raw = RawEdge.model_validate(item)
        except ValidationError:
            continue
        if raw.source.lower() == raw.target.lower():
            continue
        chunk_ids = [
            cid
            for cid, text in texts
            if raw.source.lower() in text or raw.target.lower() in text
        ] or all_ids
        edges.append(
            ExtractedEdge(
                src=raw.source,
                dst=raw.target,
                rel=raw.relation_type,
                description=raw.description,
                weight=raw.weight,
                chunk_ids=chunk_ids,
            )
        )

    return entities, edges


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []
