"""Configuration centrale de l'application."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres chargés depuis les variables d'environnement."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_embedding_model: str = "embeddinggemma"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    documents_path: str = "documents"
    vector_db_path: str = "vector_db"

    rag_collection_name: str = "backbone_documents"
    rag_top_k: int = 5
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    rag_max_distance: float = 0.90

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",     
    )


@lru_cache
def get_settings() -> Settings:
    """Retourne une instance unique de la configuration."""

    return Settings()