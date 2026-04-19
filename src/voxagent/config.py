from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VOXAGENT_",
        populate_by_name=True,
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: Literal["anthropic", "openai", "gemini"] = "anthropic"

    # Anthropic — standard env var names, no VOXAGENT_ prefix
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_generator_model: str = "claude-sonnet-4-6"
    anthropic_judge_model: str = "claude-haiku-4-5-20251001"

    # OpenAI — stub until provider is implemented
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_generator_model: str = "gpt-4o"
    openai_judge_model: str = "gpt-4o-mini"

    # Gemini — stub until provider is implemented
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_generator_model: str = "gemini-2.5-pro"
    gemini_judge_model: str = "gemini-2.5-flash"

    # Tuning (provider-agnostic)
    generator_temperature: float = 0.3
    judge_temperature: float = 0.0
    max_retries: int = 1
    generator_max_tokens: int = 1024

    # Infrastructure — standard env var names, no VOXAGENT_ prefix
    postgres_dsn: str = Field(alias="POSTGRES_DSN")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


settings = Settings()
