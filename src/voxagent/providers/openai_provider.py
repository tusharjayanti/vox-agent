from pydantic import BaseModel

from voxagent.llm import LLMResponse, Message


class OpenAIProvider:
    """Placeholder for the OpenAI provider.

    v1 ships with only AnthropicProvider implemented. OpenAI support is
    on the roadmap — see README.md 'LLM provider support' section and
    DESIGN.md §11 for the full list of deferred work.

    This class exists so the factory in config.py can route to it when
    VOXAGENT_LLM_PROVIDER=openai, producing a clear error that points
    to the roadmap rather than an ImportError.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key  # stored but unused until implemented

    async def generate(
        self,
        messages: list[Message],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError(
            "OpenAIProvider is a placeholder for the LLM provider abstraction. "
            "To implement: use the openai Python SDK's chat.completions.create(). "
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
            "OpenAIProvider.judge is a placeholder. "
            "To implement: use OpenAI's response_format={'type': 'json_schema', ...} "
            "for native structured output. See README.md / DESIGN.md §11."
        )
