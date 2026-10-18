"""Deterministic providers: offline mock and a replayed-recording provider.

RecordedLLM is how CI runs the real pipeline with zero API calls: the extraction
responses are captured once by evals/record.py and replayed by prompt hash.
"""

import hashlib
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .base import BaseLLM

STOPWORDS = {
    "The", "This", "That", "In", "On", "At", "For", "With", "By", "From", "And",
    "Or", "An", "A", "JSON", "Extract", "Return", "You", "Your", "TEXT", "END",
}


def record_key(system: str, prompt: str) -> str:
    return hashlib.sha256(f"{system}\x00{prompt}".encode()).hexdigest()


class MockLLM(BaseLLM):
    model = "mock"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return "Mock answer grounded in the provided context."

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        words = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", prompt)
        names = [w for w in dict.fromkeys(words) if w not in STOPWORDS][:8]
        entities = [
            {"name": name, "type": "CONCEPT", "description": f"{name} appears in the text."}
            for name in names
        ]
        edges = [
            {
                "source": a["name"],
                "target": b["name"],
                "relation_type": "RELATED_TO",
                "description": f"{a['name']} relates to {b['name']}.",
                "weight": 1.0,
            }
            for a, b in zip(entities, entities[1:], strict=False)
        ]
        return {"entities": entities, "relationships": edges}

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        for word in self.generate(prompt, system).split(" "):
            yield word + " "


class RecordedLLM(MockLLM):
    """Replays recorded responses; falls back to MockLLM when a prompt is new."""

    def __init__(self, path: Path):
        self._data: dict[str, Any] = {}
        self.misses = 0
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))
        self.model = self._data.get("_model", "recorded")

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        payload = self._data.get(record_key(system or "", prompt))
        if payload is None:
            self.misses += 1
            return super().generate_json(prompt, system)
        return payload
