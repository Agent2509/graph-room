"""Chunker must reproduce the behavior frozen in fixtures/chunker_golden.json."""

import json
from pathlib import Path

import pytest

from graph_rag.pipeline.chunker import chunk_text

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "fixtures" / "chunker_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename", sorted(GOLDEN["docs"]))
def test_matches_frozen_golden(filename):
    text = (ROOT / "sample_data" / filename).read_text(encoding="utf-8")
    for key, expected in GOLDEN["docs"][filename].items():
        size, overlap = (int(part) for part in key.split("_"))
        chunks = chunk_text(text, "doc", size, overlap)
        assert [chunk.text for chunk in chunks] == [item["text"] for item in expected]
        assert [chunk.idx for chunk in chunks] == [item["idx"] for item in expected]


def test_small_text_is_one_chunk():
    chunks = chunk_text("Short text.", "doc", 500, 50)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].id == "doc#0000"


def test_empty_text_is_no_chunks():
    assert chunk_text("   \n\n  ", "doc") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("x", "doc", 100, 100)


def test_long_word_run_splits_without_infinite_loop():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, "doc", 100, 20)
    assert chunks
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_ids_are_unique_and_content_addressed():
    chunks = chunk_text("A. " * 200, "abc123", 120, 30)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.id.startswith("abc123#") for chunk in chunks)
