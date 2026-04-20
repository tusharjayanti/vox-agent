"""Tests for the LLM judge + combiner layer in evaluator.py.

No real API calls — all LLM calls are mocked via the provider fixture.
"""
import pytest
from unittest.mock import AsyncMock
from pydantic import ValidationError
from voxagent.evaluator import (
    JudgeVerdict,
    EvaluationResult,
    HeuristicFlag,
    HeuristicResult,
    _combine,
    run_evaluation,
)
from voxagent.llm import LLMProvider


# ─── Fixtures ───

@pytest.fixture
def good_judge_verdict():
    return JudgeVerdict(
        relevance=5,
        groundedness=5,
        hallucination_risk="low",
        reasoning="Accurate and on-topic.",
    )


@pytest.fixture
def hallucinating_judge_verdict():
    return JudgeVerdict(
        relevance=4,
        groundedness=2,
        hallucination_risk="high",
        reasoning="Response includes invented policy details.",
    )


@pytest.fixture
def low_relevance_judge_verdict():
    return JudgeVerdict(
        relevance=1,
        groundedness=5,
        hallucination_risk="low",
        reasoning="Response does not address the user's actual question.",
    )


# ─── JudgeVerdict model validation ───

class TestJudgeVerdictModel:
    def test_valid_verdict_parses(self):
        v = JudgeVerdict(
            relevance=4, groundedness=5, hallucination_risk="low",
            reasoning="Fine.",
        )
        assert v.relevance == 4

    def test_relevance_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(
                relevance=7, groundedness=5, hallucination_risk="low",
                reasoning="Bad input",
            )

    def test_negative_groundedness_raises(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(
                relevance=5, groundedness=-1, hallucination_risk="low",
                reasoning="Bad input",
            )

    def test_invalid_hallucination_risk_raises(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(
                relevance=5, groundedness=5,
                hallucination_risk="medium-high",  # not one of the 3 valid values
                reasoning="Bad",
            )


# ─── _combine logic ───

class TestCombine:
    def _heuristic(self, verdict: str, flags: list | None = None) -> HeuristicResult:
        return HeuristicResult(
            flags=flags or [],
            passed=(verdict == "good"),
            preliminary_verdict=verdict,
        )

    def test_heuristic_escalate_always_wins(self, good_judge_verdict):
        h = self._heuristic("escalate", [HeuristicFlag.UNEXPECTED_REFUSAL])
        assert _combine(h, good_judge_verdict) == "escalate"

    def test_judge_high_hallucination_escalates(self, hallucinating_judge_verdict):
        h = self._heuristic("good")
        assert _combine(h, hallucinating_judge_verdict) == "escalate"

    def test_judge_low_relevance_retries(self, low_relevance_judge_verdict):
        h = self._heuristic("good")
        assert _combine(h, low_relevance_judge_verdict) == "retry"

    def test_judge_low_groundedness_retries(self):
        judge = JudgeVerdict(
            relevance=5, groundedness=1, hallucination_risk="low",
            reasoning="Facts not verifiable.",
        )
        h = self._heuristic("good")
        assert _combine(h, judge) == "retry"

    def test_heuristic_retry_when_judge_is_good(self, good_judge_verdict):
        h = self._heuristic("retry", [HeuristicFlag.OVER_HEDGED])
        assert _combine(h, good_judge_verdict) == "retry"

    def test_all_good(self, good_judge_verdict):
        h = self._heuristic("good")
        assert _combine(h, good_judge_verdict) == "good"

    def test_judge_none_falls_back_to_heuristic_retry(self):
        h = self._heuristic("retry", [HeuristicFlag.OVER_HEDGED])
        assert _combine(h, None) == "retry"

    def test_judge_none_with_clean_heuristic_is_good(self):
        h = self._heuristic("good")
        assert _combine(h, None) == "good"


# ─── run_evaluation integration ───

class TestRunEvaluation:
    async def test_escalate_heuristic_short_circuits_judge(self):
        """If heuristics escalate, the judge is not called — saves tokens."""
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.judge.return_value = None

        result = await run_evaluation(
            user_message="Where is my order?",
            agent_response="I can't help with that.",  # triggers UNEXPECTED_REFUSAL
            provider=mock_provider,
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.final_verdict == "escalate"
        assert result.judge is None
        mock_provider.judge.assert_not_called()

    async def test_clean_response_calls_judge_and_returns_good(
        self, good_judge_verdict,
    ):
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.judge.return_value = good_judge_verdict

        result = await run_evaluation(
            user_message="How long to return?",
            agent_response=(
                "You have 30 days from delivery to return an item."
            ),
            provider=mock_provider,
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.final_verdict == "good"
        assert result.judge == good_judge_verdict
        mock_provider.judge.assert_called_once()

    async def test_judge_exception_falls_back_to_heuristics(self):
        """If the judge throws, we don't crash — heuristics drive verdict."""
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.judge.side_effect = RuntimeError("API error")

        result = await run_evaluation(
            user_message="How long to return?",
            agent_response="You have 30 days to return an item.",
            provider=mock_provider,
            judge_model="claude-haiku-4-5-20251001",
        )

        # Heuristics alone: clean response → good
        assert result.final_verdict == "good"
        assert result.judge is None

    async def test_judge_returns_none_falls_back(self):
        """If the judge returns None (parse failure), heuristics drive verdict."""
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.judge.return_value = None

        result = await run_evaluation(
            user_message="How long to return?",
            agent_response="You have 30 days to return an item.",
            provider=mock_provider,
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.final_verdict == "good"
        assert result.judge is None

    async def test_high_hallucination_escalates(
        self, hallucinating_judge_verdict,
    ):
        mock_provider = AsyncMock(spec=LLMProvider)
        mock_provider.judge.return_value = hallucinating_judge_verdict

        result = await run_evaluation(
            user_message="What's the return policy?",
            agent_response="Returns accepted within 7 days, Tuesdays only.",
            provider=mock_provider,
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.final_verdict == "escalate"
        assert result.judge.hallucination_risk == "high"
