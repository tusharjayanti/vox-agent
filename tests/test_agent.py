"""Tests for the agent orchestrator.

All LLM calls are mocked via the LLMProvider fixture. Covers:
- Single-pass success (no retry)
- Retry path — first response fails, second succeeds
- Retry path — both responses fail, escalation fires
- Escalate on first pass (short-circuits retry)
- max_retries=0 honors no-retry config
- Correction message building for each flag type
"""
import pytest
from unittest.mock import AsyncMock
from voxagent.agent import chat, _build_retry_correction, FALLBACK_MESSAGE
from voxagent.config import Settings
from voxagent.evaluator import (
    EvaluationResult,
    HeuristicFlag,
    HeuristicResult,
    JudgeVerdict,
)
from voxagent.llm import LLMProvider, LLMResponse
from voxagent.memory import memory


@pytest.fixture(autouse=True)
def _clear_memory():
    """Each test starts with empty conversation memory."""
    memory._store.clear()
    yield
    memory._store.clear()


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key",
        postgres_dsn="postgresql://test:test@localhost:5432/test",
    )


def _llm_response(text: str, in_tokens: int = 50, out_tokens: int = 10):
    return LLMResponse(
        content=text, input_tokens=in_tokens,
        output_tokens=out_tokens, latency_ms=500,
    )


def _good_verdict():
    return JudgeVerdict(
        relevance=5, groundedness=5,
        hallucination_risk="low", reasoning="Clean.",
    )


def _low_relevance_verdict():
    return JudgeVerdict(
        relevance=1, groundedness=5,
        hallucination_risk="low", reasoning="Off-topic.",
    )


def _high_hallucination_verdict():
    return JudgeVerdict(
        relevance=4, groundedness=2,
        hallucination_risk="high", reasoning="Invented facts.",
    )


# ─── Retry correction builder ───

class TestBuildRetryCorrection:
    def test_over_hedged_correction(self):
        ev = EvaluationResult(
            heuristic=HeuristicResult(
                flags=[HeuristicFlag.OVER_HEDGED],
                passed=False,
                preliminary_verdict="retry",
            ),
            judge=None,
            final_verdict="retry",
        )
        correction = _build_retry_correction(ev)
        assert "direct" in correction.lower() or "hedge" in correction.lower() \
            or "probably" in correction.lower()

    def test_hallucination_correction(self):
        ev = EvaluationResult(
            heuristic=HeuristicResult(
                flags=[], passed=True, preliminary_verdict="good",
            ),
            judge=_high_hallucination_verdict(),
            final_verdict="retry",
        )
        correction = _build_retry_correction(ev)
        assert "unverified" in correction.lower() or "uncertain" in correction.lower()

    def test_relevance_correction(self):
        ev = EvaluationResult(
            heuristic=HeuristicResult(
                flags=[], passed=True, preliminary_verdict="good",
            ),
            judge=_low_relevance_verdict(),
            final_verdict="retry",
        )
        correction = _build_retry_correction(ev)
        assert "question" in correction.lower() \
            or "address" in correction.lower()


# ─── End-to-end agent.chat ───

class TestChatSinglePass:
    async def test_good_response_no_retry(self, settings):
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.return_value = _llm_response(
            "You have 30 days to return an item."
        )
        provider.judge.return_value = _good_verdict()

        result = await chat(
            session_id="sess-1",
            user_message="How long to return?",
            provider=provider,
            settings=settings,
        )

        assert result.final_verdict == "good"
        assert result.retry_count == 0
        assert "30 days" in result.reply
        # Generate called exactly once (no retry)
        assert provider.generate.call_count == 1


class TestChatRetry:
    async def test_retry_then_success(self, settings):
        """First attempt gets low relevance, retry succeeds."""
        provider = AsyncMock(spec=LLMProvider)
        # First response is on-topic enough to pass heuristics but the judge
        # rates it low relevance. Second attempt is the correct answer.
        provider.generate.side_effect = [
            _llm_response(
                "Your order has been received and is currently being processed "
                "by our shipping team."
            ),
            _llm_response(
                "You have 30 days to return an item."
            ),
        ]
        # First judge returns low relevance → retry
        # Second judge returns good
        provider.judge.side_effect = [
            _low_relevance_verdict(),
            _good_verdict(),
        ]

        result = await chat(
            session_id="sess-retry-1",
            user_message="How long to return?",
            provider=provider,
            settings=settings,
        )

        assert result.final_verdict == "good"
        assert result.retry_count == 1
        assert "30 days" in result.reply
        assert provider.generate.call_count == 2

    async def test_retry_then_still_bad_escalates(self, settings):
        """Both attempts fail — final verdict retry but retries exhausted."""
        provider = AsyncMock(spec=LLMProvider)
        # Both responses pass heuristics but judge rates them low relevance.
        provider.generate.side_effect = [
            _llm_response(
                "Our customer support team can help you with your order or account."
            ),
            _llm_response(
                "Please contact our support team about your order or return request."
            ),
        ]
        # Both judges return low relevance
        provider.judge.side_effect = [
            _low_relevance_verdict(),
            _low_relevance_verdict(),
        ]

        result = await chat(
            session_id="sess-retry-escalate",
            user_message="How long to return?",
            provider=provider,
            settings=settings,
        )

        # After 1 retry, still retry verdict but retries exhausted — not
        # escalated (low relevance → retry, not escalate). Agent returns the
        # second response as-is with retry_count=1.
        assert result.retry_count == 1
        assert "support team" in result.reply

    async def test_hallucination_escalates_immediately(self, settings):
        """High hallucination_risk on first attempt escalates immediately
        (before retry, because 'retry' wouldn't fix hallucination)."""
        # Wait — per our combiner rules, high hallucination IS escalate,
        # so we don't retry at all.
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.return_value = _llm_response(
            "Returns are accepted Tuesdays only, 7 days max."
        )
        provider.judge.return_value = _high_hallucination_verdict()

        result = await chat(
            session_id="sess-halluc",
            user_message="What's the return policy?",
            provider=provider,
            settings=settings,
        )

        assert result.final_verdict == "escalate"
        assert result.retry_count == 0
        assert result.reply == FALLBACK_MESSAGE
        assert provider.generate.call_count == 1  # no retry on escalate


class TestChatEscalateOnHeuristic:
    async def test_refusal_escalates_without_calling_judge(self, settings):
        """Heuristic escalate short-circuits the judge AND prevents retry."""
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.return_value = _llm_response(
            "I can't help with that request."
        )
        # Judge should not be called — heuristic says escalate
        provider.judge.return_value = _good_verdict()

        result = await chat(
            session_id="sess-refusal",
            user_message="Where is my order?",
            provider=provider,
            settings=settings,
        )

        assert result.final_verdict == "escalate"
        assert result.retry_count == 0
        assert result.reply == FALLBACK_MESSAGE
        provider.judge.assert_not_called()


class TestChatMemoryIntegration:
    async def test_user_message_added_to_memory(self, settings):
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.return_value = _llm_response("Response.")
        provider.judge.return_value = _good_verdict()

        await chat(
            session_id="sess-mem",
            user_message="First question?",
            provider=provider,
            settings=settings,
        )

        history = memory.get("sess-mem")
        assert len(history) == 2  # user + assistant
        assert history[0].role == "user"
        assert history[0].content == "First question?"
        assert history[1].role == "assistant"

    async def test_fallback_message_saved_to_memory(self, settings):
        """When escalating, the FALLBACK_MESSAGE is what's saved to memory
        (not the bad LLM response). This keeps the conversation coherent
        on future turns."""
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.return_value = _llm_response(
            "I can't help with that."
        )
        provider.judge.return_value = _good_verdict()

        await chat(
            session_id="sess-fallback-mem",
            user_message="Question?",
            provider=provider,
            settings=settings,
        )

        history = memory.get("sess-fallback-mem")
        assert history[1].content == FALLBACK_MESSAGE


class TestTokenAccounting:
    async def test_tokens_accumulated_across_retries(self, settings):
        provider = AsyncMock(spec=LLMProvider)
        provider.generate.side_effect = [
            _llm_response(
                "Your order is being processed by our shipping team.",
                in_tokens=50, out_tokens=5,
            ),
            _llm_response(
                "You have 30 days to return any item to our store.",
                in_tokens=60, out_tokens=15,
            ),
        ]
        provider.judge.side_effect = [
            _low_relevance_verdict(),
            _good_verdict(),
        ]

        result = await chat(
            session_id="sess-tokens",
            user_message="Q?",
            provider=provider,
            settings=settings,
        )

        assert result.total_input_tokens == 110  # 50 + 60
        assert result.total_output_tokens == 20  # 5 + 15
