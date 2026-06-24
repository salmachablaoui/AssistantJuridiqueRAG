from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "gestion-dossiers-juridique"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "0000"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "legal_documents"
    QDRANT_VECTOR_SIZE: int = 1024

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 32

    OLLAMA_URL: str = "http://localhost:11434"
    CHAT_MODEL: str = "mistral"
    LLM_TEMPERATURE: float = 0.3
    LLM_TIMEOUT: int = 120

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_CHARS: int = 100

    ROUTER_MODEL: str = "mistral"
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.7

    TOP_K_RESULTS: int = 5
    HYBRID_SQL_LIMIT: int = 10

    API_SECRET_KEY: str = "change-me"
    LARAVEL_API_KEY: str = "laravel-internal-key"
    DEBUG: bool = False

    PDF_STORAGE_PATH: str = "C:/Users/salma/gestion-dossier-juridique/public/storage/documents"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()