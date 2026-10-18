"""Compare rule-based extraction with the recorded LLM run. No API calls.

    python backend/evals/compare_extractors.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_rag.db import Database
from graph_rag.embeddings import Embedder
from graph_rag.llm import RecordedLLM
from graph_rag.pipeline import indexer

ROOT = Path(__file__).resolve().parents[2]


def build(mode: str, embed, llm=None) -> dict[str, set]:
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "compare.db")
        started = time.perf_counter()
        for path in sorted((ROOT / "sample_data").glob("*.txt")):
            document_id, _ = indexer.add_document(
                db, path.name, path.read_text(encoding="utf-8"), 800, 120
            )
            indexer.index_document(
                db, llm, embed, getattr(llm, "model", "fast"), document_id,
                batch_size=3, max_llm_summaries=0, mode=mode,
            )
        elapsed = time.perf_counter() - started
        conn = db.conn()
        entities = {row["name"].lower() for row in conn.execute("SELECT name FROM entities")}
        pairs = {(row["src"].lower(), row["dst"].lower()) for row in conn.execute("SELECT src, dst FROM edges")}
        triples = {
            (row["src"].lower(), row["rel"], row["dst"].lower())
            for row in conn.execute("SELECT src, dst, rel FROM edges")
        }
        db.close()
    return {"entities": entities, "pairs": pairs, "triples": triples, "seconds": elapsed}


def score(reference: set, candidate: set) -> tuple[float, float]:
    if not reference or not candidate:
        return 0.0, 0.0
    overlap = len(reference & candidate)
    return overlap / len(reference), overlap / len(candidate)


def main() -> int:
    embed = Embedder()
    recorded = RecordedLLM(ROOT / "fixtures" / "llm_responses.json")
    llm = build("llm", embed, recorded)
    fast = build("fast", embed)

    rows = [
        ("entities", llm["entities"], fast["entities"]),
        ("relations (src,dst)", llm["pairs"], fast["pairs"]),
        ("triples (src,rel,dst)", llm["triples"], fast["triples"]),
    ]
    print(f"{'':<22}{'llm':>6}{'fast':>6}{'recall':>8}{'precision':>11}")
    for name, reference, candidate in rows:
        recall, precision = score(reference, candidate)
        print(f"{name:<22}{len(reference):>6}{len(candidate):>6}{recall:>8.2f}{precision:>11.2f}")
    print(f"\nindex time: llm {llm['seconds']:.1f}s | fast {fast['seconds']:.1f}s")

    print("\nLLM entities the fast extractor missed:")
    print("  " + ", ".join(sorted(llm["entities"] - fast["entities"])) or "  none")
    print("Fast entities the LLM missed:")
    print("  " + ", ".join(sorted(fast["entities"] - llm["entities"])) or "  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
