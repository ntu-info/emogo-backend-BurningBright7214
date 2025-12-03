from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration loaded from environment variables."""

    api_prefix: str = "/api/v1"
    project_name: str = "EmoGo Backend"
    mongodb_uri: str
    database_name: str = "emogo"
    export_token: str | None = None
    cors_allow_origins: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("mongodb_uri")
    @classmethod
    def validate_mongo_uri(cls, value: str) -> str:
        if not value:
            msg = "MONGODB_URI is required. Please set it in your environment or .env file."
            raise ValueError(msg)
        return value

    @property
    def cors_origins(self) -> List[str]:
        if not self.cors_allow_origins:
            return []
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()


settings = get_settings()

