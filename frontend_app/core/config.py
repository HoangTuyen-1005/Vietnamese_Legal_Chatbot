from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.config_utils import find_env_file

_ENV_FILE = find_env_file(Path(__file__).resolve().parent)


class FrontendSettings(BaseSettings):
    APP_NAME: str = "Vietnamese Legal Chatbot - Frontend"
    DEBUG: bool = True

    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = 7860
    FRONTEND_SHARE: bool = False

    RAG_ENGINE_BASE_URL: str = "http://127.0.0.1:8000"
    REQUEST_TIMEOUT_SECONDS: int = 120

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DEBUG", "FRONTEND_SHARE", mode="before")
    @classmethod
    def parse_bool_values(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False

        normalized = str(value).strip().lower()
        return normalized in {"1", "true", "yes", "on"}

    @field_validator("REQUEST_TIMEOUT_SECONDS", mode="before")
    @classmethod
    def parse_timeout(cls, value):
        if value is None:
            return 120
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be > 0.")
        return parsed

    @field_validator("FRONTEND_PORT", mode="before")
    @classmethod
    def parse_frontend_port(cls, value):
        if value is None:
            return 7860
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("FRONTEND_PORT must be > 0.")
        return parsed


@lru_cache
def get_settings() -> FrontendSettings:
    return FrontendSettings()
