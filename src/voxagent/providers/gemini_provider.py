from pydantic import BaseModel

from voxagent.llm import LLMResponse, Message


class GeminiProvider:
    """Placeholder for the Google Gemini provider.

    See OpenAIProvider docstring for the design rationale.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def generate(
        self,
        messages: list[Message],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError(
            "GeminiProvider is a placeholder for the LLM provider abstraction. "
            "To implement: use the google-genai Python SDK's models.generate_content(). "
            "See README.md 'LLM provider support' and DESIGN.md §11 roadmap."
        )

    async def judge(
        self,
        user_message: str,
        agent_response: str,
        system: str,
        model: str,
        temperature: float,
        response_schema: type[BaseModel],
    ) -> BaseModel | None:
        raise NotImplementedError(
            "GeminiProvider.judge is a placeholder. "
            "To implement: use Gemini's response_schema in generation_config "
            "for native structured output. See README.md / DESIGN.md §11."
        )
