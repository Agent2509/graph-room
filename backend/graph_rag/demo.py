"""Seed a browsable demo index from the recorded LLM responses.

Zero API calls: the extraction recording in fixtures/ is replayed through the real
pipeline. Runs only when the database has no documents, so a real corpus is never
touched. Chunking is pinned to the values the recording was made with.
"""

import logging
from pathlib import Path

from graph_rag.db import Database
from graph_rag.llm.mock import RecordedLLM

logger = logging.getLogger("graph_rag")

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "sample_data"
RECORDING = ROOT / "fixtures" / "llm_responses.json"

# Must match evals/record.py, or the recording misses and the seed falls back to mock.
SEED_CHUNK_SIZE = 800
SEED_CHUNK_OVERLAP = 120
SEED_BATCH_SIZE = 3


def seed_demo(db: Database, embed) -> int:
    """Index the bundled sample corpus with replayed LLM output. Returns docs seeded."""
    conn = db.conn()
    if conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"] > 0:
        return 0
    if not SAMPLE_DIR.exists() or not RECORDING.exists():
        return 0

    from graph_rag.pipeline import indexer

    llm = RecordedLLM(RECORDING)
    seeded = 0
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        document_id, chunks = indexer.add_document(
            db,
            path.name,
            path.read_text(encoding="utf-8"),
            SEED_CHUNK_SIZE,
            SEED_CHUNK_OVERLAP,
        )
        if not chunks:
            continue
        indexer.index_document(
            db,
            llm,
            embed,
            llm.model,
            document_id,
            batch_size=SEED_BATCH_SIZE,
            max_llm_summaries=0,
        )
        seeded += 1

    if seeded and llm.misses:
        logger.warning(
            "Demo seed missed %d recorded prompts; the fixture is stale "
            "(run backend/evals/record.py)",
            llm.misses,
        )
    return seeded
