from pydantic_settings import BaseSettings, SettingsConfigDict

from voxagent.llm import LLMProvider


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    postgres_dsn: str = ""

    voxagent_llm_provider: str = "anthropic"

    voxagent_generator_model: str = "claude-sonnet-4-6"
    voxagent_judge_model: str = "claude-haiku-4-5-20251001"

    voxagent_max_retries: int = 1
    voxagent_judge_temperature: float = 0.0
    voxagent_generator_temperature: float = 0.3


def build_provider(settings: Settings) -> LLMProvider:
    match settings.voxagent_llm_provider.lower():
        case "anthropic":
            from voxagent.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(
                api_key=settings.anthropic_api_key,
                generator_model=settings.voxagent_generator_model,
                judge_model=settings.voxagent_judge_model,
            )
        case "openai":
            from voxagent.providers.openai_provider import OpenAIProvider
            return OpenAIProvider()
        case "gemini":
            from voxagent.providers.gemini_provider import GeminiProvider
            return GeminiProvider()
        case other:
            raise ValueError(f"Unknown LLM provider: {other!r}. Valid options: anthropic, openai, gemini")


settings = Settings()
