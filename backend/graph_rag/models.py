"""Pydantic models. The LLM is treated as untrusted input: every field coerces or is skipped."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TextChunk(BaseModel):
    id: str
    document_id: str
    idx: int
    text: str
    token_count: int


class RawEntity(BaseModel):
    name: str = Field(min_length=1)
    type: str = "CONCEPT"
    description: str = ""

    @field_validator("name", "description", "type", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> str:
        return str(value or "").strip()


class RawEdge(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation_type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0

    @field_validator("source", "target", "description", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("relation_type", mode="before")
    @classmethod
    def _relation(cls, value: Any) -> str:
        return str(value or "RELATED_TO").strip().upper().replace(" ", "_") or "RELATED_TO"

    # The model sometimes returns "0.9 (high)" or null. Never fail the batch for it.
    @field_validator("weight", mode="before")
    @classmethod
    def _weight(cls, value: Any) -> float:
        try:
            return min(max(float(value), 0.5), 1.0)
        except (TypeError, ValueError):
            return 1.0


class ExtractedEntity(BaseModel):
    name: str
    type: str = "CONCEPT"
    description: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class ExtractedEdge(BaseModel):
    src: str
    dst: str
    rel: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    chunk_ids: list[str] = Field(default_factory=list)


class Community(BaseModel):
    id: int
    title: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    rating: float = 5.0
    members: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    mode: str
    answer: str = ""
    entities: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    context: str = ""
