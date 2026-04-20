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
def mock_db(monkeypatch):
    """Patch db functions so /chat tests don't touch Postgres."""
    from voxagent import db

    monkeypatch.setattr(db, "get_or_create_conversation",
                        AsyncMock(return_value=1))
    monkeypatch.setattr(db, "insert_turn",
                        AsyncMock(return_value=42))
    monkeypatch.setattr(db, "update_turn_content",
                        AsyncMock(return_value=None))
    monkeypatch.setattr(db, "insert_evaluation",
                        AsyncMock(return_value=1))
    return {
        "conversation_id": 1,
        "turn_id": 42,
    }


@pytest.fixture
def client(mock_provider):
    """FastAPI TestClient with lifespan overridden to install the mock.

    The real lifespan constructs a real AnthropicProvider from settings
    and opens a Postgres pool. For tests we skip both and install mocks
    directly — faster, and doesn't require API keys or a running DB.

    Restores the original lifespan on teardown so tests can't affect
    each other.
    """
    @asynccontextmanager
    async def test_lifespan(app):
        app.state.provider = mock_provider
        app.state.pool = AsyncMock()
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.router.lifespan_context = original
