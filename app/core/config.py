from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Vietnamese Legal Chatbot"
    DEBUG: bool = True

    EMBEDDING_MODEL_NAME: str = "huyydangg/DEk21_hcmute_embedding"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    GENERATOR_MODEL_NAME: str = "gemini-2.0-flash"
    GEMINI_API_KEY: str | None = None
    GEMINI_TEMPERATURE: float = 0.0

    QDRANT_URL: str | None = None
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "legal_documents"
    QDRANT_PATH: str | None = None

    TOP_K_BM25: int = 10
    TOP_K_VECTOR: int = 10
    TOP_K_RERANK: int = 5
    RRF_K: int = 60

    RAW_DATA_DIR: str = "data/raw"
    CLEANED_DATA_DIR: str = "data/cleaned"
    PROCESSED_DATA_DIR: str = "data/processed"

    BM25_INDEX_PATH: str = "data/indexes/bm25_index.pkl"
    BM25_DOCS_PATH: str = "data/indexes/bm25_docs.pkl"

    EVAL_QUESTIONS_PATH: str = "eval/eval_questions.json"
    ENABLE_GENERATION: bool = True

    MAX_NEW_TOKENS: int = 512
    RETRY_INCOMPLETE_ANSWER: bool = True
    MIN_COMPLETE_ANSWER_CHARS: int = 120
    FOLLOW_REFERENCED_DIEU: bool = True
    MAX_REFERENCED_DIEU_PER_CHUNK: int = 3
    MAX_REFERENCED_DIEU_TOTAL: int = 12

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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

    @field_validator("MAX_NEW_TOKENS", mode="before")
    @classmethod
    def validate_max_new_tokens(cls, value):
        if value is None:
            return 512
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError("MAX_NEW_TOKENS must be an integer.")
        if parsed <= 0:
            raise ValueError("MAX_NEW_TOKENS must be > 0.")
        return parsed

    @field_validator("MIN_COMPLETE_ANSWER_CHARS", mode="before")
    @classmethod
    def validate_min_complete_answer_chars(cls, value):
        if value is None:
            return 120
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError("MIN_COMPLETE_ANSWER_CHARS must be an integer.")
        if parsed < 0:
            raise ValueError("MIN_COMPLETE_ANSWER_CHARS must be >= 0.")
        return parsed

    @field_validator("MAX_REFERENCED_DIEU_PER_CHUNK", "MAX_REFERENCED_DIEU_TOTAL", mode="before")
    @classmethod
    def validate_non_negative_int(cls, value):
        if value is None:
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError("Referenced dieu limits must be integers.")
        if parsed < 0:
            raise ValueError("Referenced dieu limits must be >= 0.")
        return parsed


@lru_cache
def get_settings() -> Settings:
    return Settings()
