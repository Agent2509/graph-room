"""Environment-backed settings. One object, no config framework."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# bge-small-en-v1.5. Hardcoded because the vec0 table stores a fixed width.
EMBEDDING_DIM = 384


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path("DATA_DIR", "data"))
    static_dir: Path = field(default_factory=lambda: _path("STATIC_DIR", "static"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    extraction_model: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_MODEL", "openai/gpt-oss-20b")
    )
    extraction_mode: str = field(
        default_factory=lambda: os.getenv("EXTRACTION_MODE", "fast").strip().lower()
    )
    relation_threshold: float = field(
        default_factory=lambda: float(os.getenv("RELATION_THRESHOLD", "0.62"))
    )
    seed_demo: bool = field(
        default_factory=lambda: os.getenv("SEED_DEMO", "1").lower() not in {"0", "false", "no"}
    )
    api_token: str | None = field(default_factory=lambda: os.getenv("API_TOKEN") or None)
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        )
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    extraction_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EXTRACTION_BATCH_SIZE", "3"))
    )
    max_llm_summaries: int = field(default_factory=lambda: int(os.getenv("MAX_LLM_SUMMARIES", "6")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"


settings = Settings()
