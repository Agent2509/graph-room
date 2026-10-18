"""LLM interface: text, JSON, and async token stream."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseLLM(ABC):
    model: str = "mock"

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    @abstractmethod
    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]: ...
