"""Record real extraction responses so CI and evals can run the full pipeline offline.

Run this only when the extraction prompt or the sample data changes:

    set -a; source .env; set +a
    python backend/evals/record.py

Chunks are produced through the real ingestion path, so chunk ids (and therefore
prompts) are identical to what the app sends at runtime.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_rag.config import settings
from graph_rag.db import Database
from graph_rag.llm import GroqLLM
from graph_rag.llm.mock import record_key
from graph_rag.models import TextChunk
from graph_rag.pipeline import indexer
from graph_rag.pipeline.extractor import SYSTEM, USER

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set. Nothing to record.")
        return 1
    llm = GroqLLM(settings.groq_api_key, settings.groq_model)
    payloads: dict[str, object] = {"_model": llm.model}

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "record.db")
        for path in sorted((ROOT / "sample_data").glob("*.txt")):
            document_id, _ = indexer.add_document(
                db, path.name, path.read_text(encoding="utf-8"),
                settings.chunk_size, settings.chunk_overlap,
            )
            chunks = [
                TextChunk(
                    id=row["id"], document_id=row["document_id"], idx=row["idx"],
                    text=row["text"], token_count=row["token_count"],
                )
                for row in db.conn().execute(
                    "SELECT id, document_id, idx, text, token_count FROM chunks "
                    "WHERE document_id=? ORDER BY idx",
                    (document_id,),
                )
            ]
            for start in range(0, len(chunks), settings.extraction_batch_size):
                batch = chunks[start : start + settings.extraction_batch_size]
                prompt = USER.format(
                    text="\n\n".join(f"[Chunk {chunk.id}]\n{chunk.text}" for chunk in batch)
                )
                payloads[record_key(SYSTEM, prompt)] = llm.generate_json(prompt, SYSTEM)
                print(f"recorded {path.name} batch {start // settings.extraction_batch_size + 1}")
        db.close()

    target = ROOT / "fixtures" / "llm_responses.json"
    target.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    print(f"wrote {target} with {len(payloads) - 1} responses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
