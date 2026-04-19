import time

import anthropic

from voxagent.llm import GenerateResult


class AnthropicProvider:
    def __init__(self, api_key: str, generator_model: str, judge_model: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._generator_model = generator_model
        self._judge_model = judge_model

    async def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
    ) -> GenerateResult:
        start = time.monotonic()
        response = await self._client.messages.create(
            model=self._generator_model,
            max_tokens=1024,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return GenerateResult(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )

    async def complete(
        self,
        prompt: str,
        temperature: float,
    ) -> str:
        response = await self._client.messages.create(
            model=self._judge_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text
