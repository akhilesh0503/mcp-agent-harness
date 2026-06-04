from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "harness_db"
    postgres_user: str = "harness"
    postgres_password: str = "harness_secret"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # MCP Server
    mcp_server_url: str = "http://localhost:8001"
    mcp_server_port: int = 8001

    # Budget limits per session
    session_budget_tokens: int = 50000
    session_budget_calls: int = 100

    # Harness
    harness_host: str = "0.0.0.0"
    harness_port: int = 8000
    metrics_port: int = 8080
    log_level: str = "INFO"

    # Human-in-the-loop
    hitl_approval_timeout_seconds: int = 30
    hitl_webhook_url: str = ""

    # Circuit breaker (per tool)
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_seconds: int = 30

    # File read sandbox
    file_read_base_dir: str = "/tmp/agent_files"

    # ── Derived connection strings ────────────────────────────────────────────

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton — import this everywhere
settings = get_settings()
