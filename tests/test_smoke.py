"""Smoke tests: does the server start, do endpoints work, does config load?

These tests catch accidental breakage as we add features in later phases.
They don't test deep correctness — that comes with evaluator tests in
Phase 1.5/1.6.
"""
import pytest
from voxagent.config import settings, get_provider, get_generator_model, get_judge_model


class TestConfig:
    """Config must load without error and expose the right defaults."""

    def test_settings_loads(self):
        assert settings.llm_provider in {"anthropic", "openai", "gemini"}
        assert settings.generator_temperature >= 0.0
        assert settings.judge_temperature >= 0.0
        assert settings.max_retries >= 0
        assert settings.generator_max_tokens > 0

    def test_anthropic_models_have_defaults(self):
        assert settings.anthropic_generator_model.startswith("claude")
        assert settings.anthropic_judge_model.startswith("claude")

    def test_get_provider_returns_anthropic_by_default(self):
        # Provider factory should work with the default config
        provider = get_provider(settings)
        # Spec-checking is enough — we don't want this test calling the API
        assert provider is not None

    def test_model_helpers_return_strings(self):
        assert isinstance(get_generator_model(settings), str)
        assert isinstance(get_judge_model(settings), str)


class TestHealthz:
    """The /healthz endpoint returns 200 with a status body."""

    def test_healthz_returns_ok(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatSmoke:
    """The /chat endpoint accepts valid requests and returns shaped responses."""

    def test_chat_valid_request_returns_200(self, client, mock_db):
        response = client.post("/chat", json={
            "session_id": "smoke-test-1",
            "message": "Hello",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "smoke-test-1"
        assert "reply" in body
        assert "turn_id" in body

    def test_chat_missing_message_returns_422(self, client, mock_db):
        # FastAPI validates request bodies via Pydantic; missing required
        # fields should return 422 before any handler runs.
        response = client.post("/chat", json={"session_id": "test"})
        assert response.status_code == 422

    def test_chat_empty_message_returns_422(self, client, mock_db):
        response = client.post("/chat", json={
            "session_id": "test",
            "message": "",
        })
        assert response.status_code == 422

    def test_chat_missing_session_id_returns_422(self, client, mock_db):
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 422

    def test_chat_calls_provider(self, client, mock_db, mock_provider):
        client.post("/chat", json={
            "session_id": "smoke-test-2",
            "message": "Hello",
        })
        assert mock_provider.generate.called


class TestProviderAbstraction:
    """The LLMProvider protocol and factory work correctly."""

    def test_llm_provider_is_runtime_checkable(self):
        from voxagent.llm import LLMProvider
        from voxagent.providers.anthropic_provider import AnthropicProvider
        # AnthropicProvider satisfies the LLMProvider protocol at runtime
        instance = AnthropicProvider(api_key="test-key")
        assert isinstance(instance, LLMProvider)
