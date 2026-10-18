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


@pytest.fixture
def embed() -> FakeEmbedder:
    return FakeEmbedder()


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
