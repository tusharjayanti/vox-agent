import logging
import time

import anthropic
from pydantic import BaseModel, ValidationError

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
        """Call Haiku with a JSON-output prompt, parse into response_schema.

        Returns None if the API call fails or the response can't be
        parsed into response_schema. The caller (evaluator.run_evaluation)
        handles None by falling back to heuristics-only evaluation.
        """
        user_turn = (
            f"USER QUESTION:\n{user_message}\n\n"
            f"AGENT RESPONSE:\n{agent_response}\n\n"
            f"Evaluate the agent's response."
        )

        try:
            api_response = await self._client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user_turn}],
                temperature=temperature,
                max_tokens=256,
            )
        except Exception:
            return None

        raw_text = api_response.content[0].text.strip()

        # Strip markdown fences defensively — Haiku sometimes adds them
        # despite the prompt. Handle ```json ... ``` and ``` ... ```
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines)

        try:
            return response_schema.model_validate_json(raw_text)
        except ValidationError:
            return None
