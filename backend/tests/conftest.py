"""Shared fixtures. FakeEmbedder keeps tests offline and deterministic."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_rag.config import Settings  # noqa: E402
from graph_rag.db import Database  # noqa: E402


class FakeEmbedder:
    model_name = "fake"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    def one(self, text: str) -> list[float]:
        return self([text])[0]


SEMANTIC_KEYWORDS = {
    "FOUNDED": ("founded", "co-founded", "established", "started", "founding"),
    "ACQUIRED": ("acquired", "bought", "purchased", "acquisition"),
    "DEVELOPED": ("developed", "built", "created", "engineered", "designed"),
    "PARTNERS_WITH": (
        "partnership",
        "partnered",
        "collaborated",
        "collaboration",
        "teamed",
        "agreement",
    ),
    "PROVIDES": ("provides", "supplies", "offers", "hosts"),
    "BASED_IN": ("headquartered", "based in", "located in", "operates from"),
    "WORKS_FOR": ("appointed", "promoted", "works for", "joined", "employed"),
    "GRANTED": ("granted", "approved", "authorized", "priority review"),
    "USES": ("uses", "deployed", "adopted", "implemented"),
    "PART_OF": ("subsidiary", "part of", "division"),
}


class SemanticEmbedder:
    """Keyword → one-hot vectors. Tests the pipeline logic, not model quality."""

    model_name = "semantic-stub"
    labels = list(SEMANTIC_KEYWORDS)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(384, dtype=np.float32)
            for index, (_, words) in enumerate(SEMANTIC_KEYWORDS.items()):
                if any(word in lowered for word in words):
                    vector[index] = 1.0
            vectors.append(vector.tolist())
        return vectors

    def one(self, text: str) -> list[float]:
        return self([text])[0]


@pytest.fixture
def embed() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def semantic_embed() -> SemanticEmbedder:
    return SemanticEmbedder()


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "app.db")
    yield database
    database.close()


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "data",
        "static_dir": tmp_path / "static",
        "groq_api_key": None,
        "api_token": None,
        "seed_demo": False,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(tmp_path, embed):
    from fastapi.testclient import TestClient

    from graph_rag.api.app import create_app
    from graph_rag.llm.mock import MockLLM

    app = create_app(make_settings(tmp_path), llm=MockLLM(), embed=embed)
    with TestClient(app) as test_client:
        yield test_client
