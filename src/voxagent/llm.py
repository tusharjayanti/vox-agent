from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class GenerateResult(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[dict],
        system: str,
        temperature: float,
    ) -> GenerateResult: ...

    async def complete(
        self,
        prompt: str,
        temperature: float,
    ) -> str: ...
