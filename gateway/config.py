"""Gateway settings — everything comes from the environment (.env in dev)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # host port 5433 (Argus owns 5432 locally); inside compose the gateway
    # overrides these with postgres:5432
    database_url: str = "postgresql+asyncpg://relay:relay@localhost:5433/relay"
    database_url_sync: str = "postgresql+psycopg2://relay:relay@localhost:5433/relay"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"
    ollama_base_url: str = "http://localhost:11434"
    # host ports 8200/8201 (Argus owns 8100); compose overrides internally
    mock_provider_url: str = "http://localhost:8200"
    mock_provider_b_url: str = "http://localhost:8201"

    max_daily_spend_usd: float = 5.00
    slack_webhook_url: str = ""

    admin_key: str = "relay-admin-dev-key"
    verifier_metrics_port: int = 8090
    relay_host: str = "0.0.0.0"
    relay_port: int = 8080
    routing_config_path: str = "config/routing.yaml"
    log_level: str = "INFO"

    # Embedding model for the semantic cache (ADR-0001).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384


settings = Settings()
