from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.config_utils import find_env_file, normalize_legacy_path

_ENV_FILE = find_env_file(Path(__file__).resolve().parent)

_LEGACY_PATH_MAP = {
    "data/raw": "data_pipeline/data/raw",
    "data/cleaned": "data_pipeline/data/cleaned",
    "data/processed": "data_pipeline/data/processed",
    "data/indexes/bm25_index.pkl": "data_pipeline/data/indexes/bm25_index.pkl",
    "data/indexes/bm25_docs.pkl": "data_pipeline/data/indexes/bm25_docs.pkl",
    "data/indexes/qdrant_db": "data_pipeline/data/indexes/qdrant_db",
}


class DataPipelineSettings(BaseSettings):
    APP_NAME: str = "Vietnamese Legal Chatbot - Data Pipeline"
    DEBUG: bool = True

    EMBEDDING_MODEL_NAME: str = "huyydangg/DEk21_hcmute_embedding"

    QDRANT_URL: str | None = None
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "legal_documents"
    QDRANT_PATH: str | None = None

    RAW_DATA_DIR: str = "data_pipeline/data/raw"
    CLEANED_DATA_DIR: str = "data_pipeline/data/cleaned"
    PROCESSED_DATA_DIR: str = "data_pipeline/data/processed"

    BM25_INDEX_PATH: str = "data_pipeline/data/indexes/bm25_index.pkl"
    BM25_DOCS_PATH: str = "data_pipeline/data/indexes/bm25_docs.pkl"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return True

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return True

    @field_validator(
        "RAW_DATA_DIR",
        "CLEANED_DATA_DIR",
        "PROCESSED_DATA_DIR",
        "BM25_INDEX_PATH",
        "BM25_DOCS_PATH",
        "QDRANT_PATH",
        mode="before",
    )
    @classmethod
    def map_legacy_paths(cls, value):
        return normalize_legacy_path(value, _LEGACY_PATH_MAP)


@lru_cache
def get_settings() -> DataPipelineSettings:
    return DataPipelineSettings()
