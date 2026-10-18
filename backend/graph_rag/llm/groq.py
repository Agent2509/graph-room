"""Groq client. Sync for extraction, async for streamed answers.

Retries are explicit: 3 attempts with backoff, model failover on 404, and no retry
on auth or validation errors.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from groq import AsyncGroq, Groq

from .base import BaseLLM

FALLBACKS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
JSON_HINT = "\nReturn only valid JSON. No prose, no markdown fences."


def _retry_after(error: Exception) -> float:
    """Honour the server's wait hint on 429, else fall back to exponential backoff."""
    response = getattr(error, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return min(float(header.get("retry-after", 0)), 30.0)
    except (TypeError, ValueError):
        return 0.0


def _is_fatal(error: Exception) -> bool:
    code = getattr(error, "status_code", None)
    return code is not None and 400 <= int(code) < 500 and int(code) != 429


def _is_missing_model(error: Exception) -> bool:
    text = str(error).lower()
    return getattr(error, "status_code", None) == 404 or "model_not_found" in text


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = FALLBACKS[0]):
        if not api_key:
            raise ValueError("Groq API key is required")
        self.model = model
        self._models = [model] + [m for m in FALLBACKS if m != model]
        self._sync = Groq(api_key=api_key, max_retries=0)
        self._async = AsyncGroq(api_key=api_key, max_retries=0)

    def _call(self, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(4):
            try:
                return self._sync.chat.completions.create(**kwargs)
            except Exception as error:  # noqa: BLE001 - retry policy is the point
                if _is_fatal(error):
                    raise
                last = error
                if _is_missing_model(error) and len(self._models) > 1:
                    self._models.pop(0)
                    self.model = self._models[0]
                    kwargs["model"] = self.model
                    continue
                time.sleep(max(2**attempt, _retry_after(error)))
        raise RuntimeError(f"Groq call failed after retries: {last}") from last

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        response = self._call(
            model=self.model, messages=messages, temperature=0.1, max_tokens=2048
        )
        return response.choices[0].message.content or ""

    def generate_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        try:
            response = self._call(
                model=self.model,
                messages=[
                    {"role": "system", "content": (system or "") + JSON_HINT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return _loads(response.choices[0].message.content or "{}")
        except Exception as error:  # noqa: BLE001
            # Groq's JSON mode rejects the occasional malformed generation
            # (400 json_validate_failed). Plain mode + tolerant parsing beats losing a batch.
            if "json_validate_failed" not in str(error):
                raise
            return _loads(self.generate(prompt, (system or "") + JSON_HINT))

    async def stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        last: Exception | None = None
        result = None
        # Retry only before the first token: a retry mid-stream would duplicate text.
        for attempt in range(4):
            try:
                result = await self._async.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.2,
                    max_tokens=2048, stream=True,
                )
                break
            except Exception as error:  # noqa: BLE001 - retry policy is the point
                if _is_fatal(error):
                    raise
                last = error
                await asyncio.sleep(max(2**attempt, _retry_after(error)))
        if result is None:
            raise RuntimeError(f"Groq stream failed after retries: {last}") from last
        async for chunk in result:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def _loads(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating fences and surrounding prose."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text) or re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            data = json.loads(match.group(1))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}
