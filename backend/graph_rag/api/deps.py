"""Shared API dependencies: bearer token and upload cap."""

import hmac
import os

from fastapi import Header, HTTPException, Request


def require_token(
    request: Request, authorization: str | None = Header(default=None)
) -> None:
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
