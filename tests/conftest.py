"""Shared fixtures for vox-agent tests."""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from voxagent.llm import LLMResponse, LLMProvider
from voxagent.main import app


@pytest.fixture
def mock_provider():
    """A mock LLMProvider for tests that don't need a real API call."""
    provider = AsyncMock(spec=LLMProvider)
    provider.generate.return_value = LLMResponse(
        content="Mocked response for testing.",
        input_tokens=50,
        output_tokens=10,
        latency_ms=500,
    )
    return provider


@pytest.fixture
def client(mock_provider):
    """FastAPI TestClient with lifespan overridden to install the mock.

    The real lifespan constructs a real AnthropicProvider from settings.
    For tests we skip that and install the mock directly — faster, and
    doesn't require ANTHROPIC_API_KEY in the test environment.

    Restores the original lifespan on teardown so tests can't affect
    each other.
    """
    @asynccontextmanager
    async def test_lifespan(app):
        app.state.provider = mock_provider
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.router.lifespan_context = original
