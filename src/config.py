"""
Global configuration — driven by environment variables via Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM: DeepSeek (OpenAI-compatible) ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ── Storage ──
    sqlite_db_path: str = "data/pipeline.db"
    chromadb_path: str = "data/chromadb"

    # ── RAG ──
    rag_top_k: int = 3
    rag_embedding_model: str = "all-MiniLM-L6-v2"

    # ── Pipeline ──
    max_retries: int = 3
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
