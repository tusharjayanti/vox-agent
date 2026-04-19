from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from voxagent.llm import LLMProvider


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


def get_provider(settings: Settings) -> LLMProvider:
    """Factory: returns the correct LLMProvider based on settings.llm_provider.

    In v1, only the "anthropic" provider returns a working instance. The
    other two raise NotImplementedError from their methods — this factory
    still constructs them so the error comes from the method call, not
    from an ImportError at startup.
    """
    match settings.llm_provider:
        case "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when "
                    "VOXAGENT_LLM_PROVIDER=anthropic"
                )
            from voxagent.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key=settings.anthropic_api_key)
        case "openai":
            from voxagent.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(api_key=settings.openai_api_key)
        case "gemini":
            from voxagent.providers.gemini_provider import GeminiProvider
            return GeminiProvider(api_key=settings.gemini_api_key)
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_generator_model(settings: Settings) -> str:
    """Return the generator model name for the active provider."""
    match settings.llm_provider:
        case "anthropic":
            return settings.anthropic_generator_model
        case "openai":
            return settings.openai_generator_model
        case "gemini":
            return settings.gemini_generator_model
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_judge_model(settings: Settings) -> str:
    """Return the judge model name for the active provider."""
    match settings.llm_provider:
        case "anthropic":
            return settings.anthropic_judge_model
        case "openai":
            return settings.openai_judge_model
        case "gemini":
            return settings.gemini_judge_model
        case _:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
