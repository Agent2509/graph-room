"""Shared API dependencies: bearer token and upload cap."""

import hashlib
import hmac
import os

from fastapi import Header, HTTPException, Request

from graph_rag.llm import BaseLLM, GroqLLM

MAX_CACHED_CLIENTS = 8


def require_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    """No API_TOKEN set means open (local dev). Otherwise require it on every route."""
    token = request.app.state.settings.api_token
    if not token:
        return
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def check_upload_size(size: int, cap: int) -> None:
    if size > cap:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is larger than the {cap // (1024 * 1024)} MB limit",
        )


def env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _client(request: Request, key: str, model: str) -> BaseLLM:
    cache: dict[str, BaseLLM] = request.app.state.llm_cache
    fingerprint = hashlib.sha256(f"{model}:{key}".encode()).hexdigest()
    client = cache.get(fingerprint)
    if client is None:
        client = GroqLLM(key, model)
        if len(cache) >= MAX_CACHED_CLIENTS:
            cache.pop(next(iter(cache)))
        cache[fingerprint] = client
    return client


def _require_key(request: Request, x_llm_key: str | None) -> str:
    key = (x_llm_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Missing Groq API key. Add yours in the sidebar; it stays in your browser.",
        )
    return key


def get_answer_llm(
    request: Request, x_llm_key: str | None = Header(default=None, alias="X-LLM-Key")
) -> BaseLLM:
    """The good model, used only for answering questions."""
    injected = request.app.state.llm
    if injected is not None:  # tests and CLI injection
        return injected
    return _client(request, _require_key(request, x_llm_key), request.app.state.settings.groq_model)


def get_extraction_llm(
    request: Request, x_llm_key: str | None = Header(default=None, alias="X-LLM-Key")
) -> BaseLLM | None:
    """The cheap model for graph work; None in fast mode (no key needed to ingest)."""
    injected = request.app.state.llm
    if injected is not None:
        return injected
    if request.app.state.settings.extraction_mode == "fast":
        return None
    return _client(
        request,
        _require_key(request, x_llm_key),
        request.app.state.settings.extraction_model,
    )
