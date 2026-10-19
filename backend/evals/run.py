"""Retrieval evals: deterministic by default, live LLM with --live.

Scores what retrieval returned (entities, chunks, routed mode), not the prose the
LLM wrote, so the result is stable in CI. Exits non-zero below threshold.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_rag.db import Database
from graph_rag.embeddings import Embedder
from graph_rag.llm import GroqLLM, RecordedLLM
from graph_rag.pipeline import indexer
from graph_rag.retrieval import service

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {"mode_accuracy": 0.7, "entity_recall": 0.6, "chunk_recall": 0.6}


def load_golden() -> list[dict]:
    path = Path(__file__).with_name("golden.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_index(db: Database, llm, embed, model: str, extractor: str = "recorded") -> None:
    for path in sorted((ROOT / "sample_data").glob("*.txt")):
        document_id, _ = indexer.add_document(
            db, path.name, path.read_text(encoding="utf-8"), 800, 120
        )
        indexer.index_document(
            db,
            llm,
            embed,
            model,
            document_id,
            batch_size=3,
            max_llm_summaries=0,
            mode="fast" if extractor == "fast" else "llm",
        )


def score(db: Database, llm, embed, use_graph: bool = True) -> dict:
    rows = []
    for case in load_golden():
        context = service.prepare(db, llm, embed, case["question"], "auto", use_graph=use_graph)
        entity_names = " | ".join(item["name"].lower() for item in context["entities"])
        chunk_text = " ".join(item["text"].lower() for item in context["chunks"])
        entity_text = entity_names + " " + chunk_text
        expected_entities = [name.lower() for name in case["entities"]]
        expected_terms = [term.lower() for term in case["terms"]]
        rows.append(
            {
                "question": case["question"],
                "expected_mode": case["mode"],
                "mode": context["mode"],
                "mode_ok": context["mode"] == case["mode"],
                "entities_ok": (
                    any(name in entity_names for name in expected_entities)
                    if expected_entities
                    else None
                ),
                "chunks_ok": (
                    any(term in chunk_text for term in expected_terms) if expected_terms else None
                ),
                # same question, both retrievers: is the named entity visible at all?
                "evidence_ok": (
                    any(name in entity_text for name in expected_entities)
                    if expected_entities
                    else None
                ),
            }
        )

    def mean(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 1.0

    return {
        "rows": rows,
        "mode_accuracy": round(mean([row["mode_ok"] for row in rows]), 3),
        # Chunks-only retrieval has no entity nodes by design.
        "entity_recall": (
            round(mean([row["entities_ok"] for row in rows if row["entities_ok"] is not None]), 3)
            if use_graph
            else None
        ),
        "chunk_recall": round(
            mean([row["chunks_ok"] for row in rows if row["chunks_ok"] is not None]), 3
        ),
        "evidence_recall": round(
            mean([row["evidence_ok"] for row in rows if row["evidence_ok"] is not None]), 3
        ),
    }


def print_table(report: dict) -> None:
    print(f"{'mode':<8} {'entities':<9} {'chunks':<7} question")
    for row in report["rows"]:
        print(
            f"{'ok' if row['mode_ok'] else 'MISS':<8} "
            f"{'-' if row['entities_ok'] is None else ('ok' if row['entities_ok'] else 'MISS'):<9} "
            f"{'-' if row['chunks_ok'] is None else ('ok' if row['chunks_ok'] else 'MISS'):<7} "
            f"{row['question']}"
        )
    print(
        f"\nmode_accuracy={report['mode_accuracy']} "
        f"entity_recall={report['entity_recall']} "
        f"chunk_recall={report['chunk_recall']} "
        f"evidence_recall={report['evidence_recall']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use Groq instead of the recording")
    parser.add_argument(
        "--extractor",
        choices=["recorded", "fast"],
        default="recorded",
        help="recorded: replayed LLM extraction; fast: rule-based, no LLM",
    )
    parser.add_argument(
        "--retriever",
        choices=["graph", "chunks"],
        default="graph",
        help="graph: entity seeds + subgraph + mentions; chunks: BM25 + dense only",
    )
    parser.add_argument("--out", default=str(Path(__file__).with_name("report.json")))
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "eval.db")
        embed = Embedder()
        if arguments.live:
            from graph_rag.config import settings

            if not settings.groq_api_key:
                print("GROQ_API_KEY is not set; cannot run live evals")
                return 1
            llm, model = GroqLLM(settings.groq_api_key, settings.groq_model), settings.groq_model
        else:
            recorded = RecordedLLM(ROOT / "fixtures" / "llm_responses.json")
            llm, model = recorded, recorded.model

        build_index(db, llm, embed, model, arguments.extractor)
        report = score(db, llm, embed, use_graph=arguments.retriever == "graph")
        if not arguments.live:
            report["recording_misses"] = getattr(llm, "misses", 0)
        print_table(report)
        Path(arguments.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        db.close()

    failed = [
        name
        for name, minimum in THRESHOLDS.items()
        if report[name] is not None and report[name] < minimum
    ]
    if failed:
        print(f"FAILED thresholds: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
