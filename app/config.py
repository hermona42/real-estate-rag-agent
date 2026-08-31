# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Central configuration class reading from environment variables."""
    
    APP_NAME: str = "Real Estate RAG Agent"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    
    # Qdrant Vector DB Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate settings instance
settings = Settings()