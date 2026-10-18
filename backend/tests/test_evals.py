"""The recorded LLM fixture must cover every prompt the current pipeline sends.

This is the drift guard: change the chunker, the extraction prompt, or the batch
size and this test fails before CI silently falls back to the mock LLM.
"""

import json
from pathlib import Path

from graph_rag.llm.mock import record_key
from graph_rag.pipeline import indexer
from graph_rag.pipeline.extractor import SYSTEM, USER

ROOT = Path(__file__).resolve().parents[2]


def test_recording_covers_current_prompts(db):
    data = json.loads((ROOT / "fixtures" / "llm_responses.json").read_text(encoding="utf-8"))
    checked = 0
    for path in sorted((ROOT / "sample_data").glob("*.txt")):
        document_id, _ = indexer.add_document(
            db, path.name, path.read_text(encoding="utf-8"), 800, 120
        )
        chunks = list(
            db.conn().execute(
                "SELECT id, text FROM chunks WHERE document_id=? ORDER BY idx", (document_id,)
            )
        )
        for start in range(0, len(chunks), 3):
            batch = chunks[start : start + 3]
            prompt = USER.format(
                text="\n\n".join(f"[Chunk {row['id']}]\n{row['text']}" for row in batch)
            )
            assert record_key(SYSTEM, prompt) in data, (
                f"fixtures/llm_responses.json is stale for {path.name}; "
                "run: set -a; . ./.env; set +a; python backend/evals/record.py"
            )
            checked += 1
    assert checked == 3
