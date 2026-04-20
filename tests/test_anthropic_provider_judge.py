"""Tests for AnthropicProvider.judge() — the real Haiku-based judge call.

All API calls are mocked. We test JSON parsing, error handling, and
markdown-fence stripping.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field
from voxagent.providers.anthropic_provider import AnthropicProvider


class _FakeVerdict(BaseModel):
    relevance: int = Field(ge=0, le=5)
    reasoning: str


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key")


def _fake_api_response(text: str) -> MagicMock:
    """Fake what self._client.messages.create returns."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


class TestJudgeJsonParsing:
    async def test_valid_json_parses_into_schema(self, provider):
        fake = _fake_api_response('{"relevance": 5, "reasoning": "Good."}')
        provider._client.messages.create = AsyncMock(return_value=fake)

        result = await provider.judge(
            user_message="q",
            agent_response="a",
            system="sys",
            model="claude-haiku-4-5-20251001",
            temperature=0.0,
            response_schema=_FakeVerdict,
        )

        assert result is not None
        assert result.relevance == 5

    async def test_json_with_markdown_fences_is_stripped(self, provider):
        fake = _fake_api_response(
            '```json\n{"relevance": 4, "reasoning": "Ok."}\n```'
        )
        provider._client.messages.create = AsyncMock(return_value=fake)

        result = await provider.judge(
            user_message="q", agent_response="a", system="sys",
            model="claude-haiku-4-5-20251001", temperature=0.0,
            response_schema=_FakeVerdict,
        )

        assert result is not None
        assert result.relevance == 4

    async def test_malformed_json_returns_none(self, provider):
        fake = _fake_api_response("This is not JSON at all.")
        provider._client.messages.create = AsyncMock(return_value=fake)

        result = await provider.judge(
            user_message="q", agent_response="a", system="sys",
            model="claude-haiku-4-5-20251001", temperature=0.0,
            response_schema=_FakeVerdict,
        )

        assert result is None

    async def test_json_failing_schema_validation_returns_none(self, provider):
        # Returns valid JSON but relevance=99 fails Pydantic's ge=0, le=5
        fake = _fake_api_response('{"relevance": 99, "reasoning": "Nope."}')
        provider._client.messages.create = AsyncMock(return_value=fake)

        result = await provider.judge(
            user_message="q", agent_response="a", system="sys",
            model="claude-haiku-4-5-20251001", temperature=0.0,
            response_schema=_FakeVerdict,
        )

        assert result is None


class TestJudgeErrorHandling:
    async def test_api_exception_returns_none(self, provider):
        """If the Anthropic API throws (rate limit, network error, etc.),
        judge() returns None rather than propagating."""
        provider._client.messages.create = AsyncMock(
            side_effect=RuntimeError("network failure"),
        )

        result = await provider.judge(
            user_message="q", agent_response="a", system="sys",
            model="claude-haiku-4-5-20251001", temperature=0.0,
            response_schema=_FakeVerdict,
        )

        assert result is None
