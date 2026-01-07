"""Configuration management for Document Intelligence Pipeline."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "docint"
    postgres_password: str = "localdev"
    postgres_db: str = "docint"

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # Processing
    render_dpi: int = 300
    max_workers: int = 8

    # Chunking
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 64

    # Logging
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """Construct async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Construct sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
