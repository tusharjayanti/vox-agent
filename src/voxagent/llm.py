"""
LLM provider protocol and shared response types.

Agent code (agent.py, evaluator.py) imports from here only — never from
provider SDKs directly. All provider-specific code lives in providers/.
"""
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation.

    Shared type used by both the conversation memory (Step 5) and the
    LLM provider interface. Keeping this in one place avoids a layer
    of dict-to-Message conversion at every call site.
    """

    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMResponse(BaseModel):
    """Structured response from any LLMProvider.generate() call.

    Providers are responsible for normalising their SDK's response
    shape into this model. Token counts, latency, and content are
    extracted uniformly so downstream code doesn't care which
    provider was used.
    """

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol every provider must satisfy.

    Agent logic (agent.py, evaluator.py) depends on this protocol, not
    on any specific SDK. Adding a new provider means adding a new file
    in providers/ that implements these two methods.
    """

    async def generate(
        self,
        messages: list[Message],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate an open-ended text response given a conversation."""
        ...

    async def judge(
        self,
        user_message: str,
        agent_response: str,
        system: str,
        model: str,
        temperature: float,
        response_schema: type[BaseModel],
    ) -> BaseModel | None:
        """Score a response against a rubric, returning structured output.

        Returns None if the provider returned malformed JSON that could
        not be parsed into response_schema. Implementers handle provider-
        specific structured output mechanisms (Anthropic: prompting +
        validation; OpenAI: response_format; Gemini: response_schema).
        """
        ...
