import logging
import time

import anthropic
from pydantic import BaseModel

from voxagent.llm import LLMResponse, Message

logger = logging.getLogger("voxagent")


class AnthropicProvider:
    """LLMProvider implementation backed by the Anthropic SDK."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: list[Message],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        anthropic_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        start = time.perf_counter()
        api_response = await self._client.messages.create(
            model=model,
            system=system,
            messages=anthropic_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            content=api_response.content[0].text,
            input_tokens=api_response.usage.input_tokens,
            output_tokens=api_response.usage.output_tokens,
            latency_ms=latency_ms,
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
        # Stubbed for Step 4. Full implementation in Step 7 (LLM judge).
        raise NotImplementedError(
            "AnthropicProvider.judge() is implemented in Step 7 (Phase 1.6 — LLM judge). "
            "This stub exists so the LLMProvider protocol is complete from Step 4."
        )
