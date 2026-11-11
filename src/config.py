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
    raw_store_path: str = "data/raw"  # bronze layer: <sha256>.txt per source text

    # ── RAG ──
    # Off by default: with RAG on, a batch run feeds earlier documents' results
    # to later ones as few-shot examples, so results depend on processing order
    # and concurrency. Keep it off while running evaluations.
    rag_enabled: bool = False
    rag_top_k: int = 3
    rag_embedding_model: str = "all-MiniLM-L6-v2"

    # ── Classifier ──
    classifier_input_strategy: str = (
        "head"  # head (V0) | structured (V1) | keyword (V2)
    )
    classifier_input_tokens: int = 3500  # budget for structured / keyword
    classify_temperature: float = 0.05  # no effect while thinking is enabled
    # DeepSeek thinking mode for the classifier: "disabled" | "enabled" | ""
    # ("" = do not send the parameter, provider default).  Thinking spends
    # hidden reasoning tokens inside max_tokens, which truncated 26 of 233
    # replies at 1,024 tokens in the first E1 run, and it ignores temperature.
    classify_thinking: str = "disabled"
    classify_max_tokens: int = 1024

    # ── Pipeline ──
    max_retries: int = 3  # extraction self-correction budget
    classify_max_retries: int = 2  # classification self-correction budget
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
