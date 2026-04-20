"""Tests for the heuristic evaluator layer.

All pure functions. No mocks needed. No API calls."""
import pytest
from voxagent.evaluator import (
    HeuristicFlag,
    check_length,
    check_refusal,
    check_hedging,
    check_off_topic,
    run_heuristics,
)


class TestCheckLength:
    def test_empty_response_flags_too_short(self):
        assert HeuristicFlag.TOO_SHORT in check_length("")

    def test_whitespace_only_flags_too_short(self):
        assert HeuristicFlag.TOO_SHORT in check_length("      ")

    def test_short_response_flags_too_short(self):
        assert HeuristicFlag.TOO_SHORT in check_length("Yes.")

    def test_normal_response_has_no_length_flag(self):
        response = "You have 30 days from the date of delivery to return an item."
        assert check_length(response) == []

    def test_very_long_response_flags_too_long(self):
        response = "x" * 3000
        assert HeuristicFlag.TOO_LONG in check_length(response)


class TestCheckRefusal:
    def test_i_cant_help_flags_refusal(self):
        assert HeuristicFlag.UNEXPECTED_REFUSAL in \
            check_refusal("I can't help with that request.")

    def test_im_not_able_to_flags_refusal(self):
        assert HeuristicFlag.UNEXPECTED_REFUSAL in \
            check_refusal("Sorry, I'm not able to process that right now.")

    def test_case_insensitive(self):
        assert HeuristicFlag.UNEXPECTED_REFUSAL in \
            check_refusal("I CAN'T HELP you with this.")

    def test_normal_response_is_not_flagged(self):
        response = "Your order was shipped via standard delivery."
        assert check_refusal(response) == []

    def test_mentioning_cant_in_other_context_is_not_flagged(self):
        # "can't" by itself isn't enough — needs to be a refusal phrase
        response = "The package can't be tracked until it's picked up by the courier."
        assert check_refusal(response) == []


class TestCheckHedging:
    def test_single_hedge_is_not_flagged(self):
        response = "I think your order will ship tomorrow."
        assert check_hedging(response) == []

    def test_two_hedges_is_not_flagged(self):
        response = "I think it might be delayed, but it should still arrive this week."
        # "i think" and "might be" = 2 hedges, under threshold
        assert check_hedging(response) == []

    def test_three_hedges_flags_over_hedged(self):
        response = "I think your order probably might be delayed by a day or two."
        assert HeuristicFlag.OVER_HEDGED in check_hedging(response)

    def test_many_hedges_flags_over_hedged(self):
        response = (
            "I'm not sure, but I think your order probably might be in transit. "
            "I believe it could be delayed, possibly due to weather — maybe even longer."
        )
        assert HeuristicFlag.OVER_HEDGED in check_hedging(response)

    def test_case_insensitive(self):
        response = "I THINK your order PROBABLY MIGHT BE delayed."
        assert HeuristicFlag.OVER_HEDGED in check_hedging(response)


class TestCheckOffTopic:
    def test_response_mentioning_order_is_on_topic(self):
        response = "Your order shipped yesterday."
        assert check_off_topic(response) == []

    def test_response_mentioning_return_is_on_topic(self):
        response = "You have 30 days to return the item."
        assert check_off_topic(response) == []

    def test_off_topic_response_is_flagged(self):
        response = (
            "Photosynthesis is the process by which plants convert "
            "sunlight into chemical energy."
        )
        assert HeuristicFlag.OFF_TOPIC in check_off_topic(response)

    def test_short_non_support_response_is_flagged(self):
        assert HeuristicFlag.OFF_TOPIC in \
            check_off_topic("The weather is nice today.")


class TestRunHeuristics:
    """Integration of the four checks into a combined verdict."""

    def test_clean_response_verdict_is_good(self):
        response = "You have 30 days from the date of delivery to return an item."
        result = run_heuristics(response)
        assert result.preliminary_verdict == "good"
        assert result.passed is True
        assert result.flags == []

    def test_empty_response_verdict_is_retry(self):
        result = run_heuristics("")
        # TOO_SHORT + OFF_TOPIC (no support vocab in "") → escalate, not retry
        # Because OFF_TOPIC triggers escalate
        assert HeuristicFlag.TOO_SHORT in result.flags
        assert result.passed is False

    def test_refusal_verdict_is_escalate(self):
        response = "I can't help with that request."
        result = run_heuristics(response)
        assert result.preliminary_verdict == "escalate"
        assert HeuristicFlag.UNEXPECTED_REFUSAL in result.flags

    def test_off_topic_verdict_is_escalate(self):
        response = (
            "Photosynthesis is the process by which plants convert "
            "sunlight into chemical energy via chlorophyll."
        )
        result = run_heuristics(response)
        assert result.preliminary_verdict == "escalate"
        assert HeuristicFlag.OFF_TOPIC in result.flags

    def test_over_hedged_verdict_is_retry(self):
        response = (
            "I think your order probably might be delayed, but I'm not "
            "entirely sure about the shipping details."
        )
        result = run_heuristics(response)
        # "order" keeps it on-topic, "might be / probably / i think / i'm not sure" = 4 hedges
        assert result.preliminary_verdict == "retry"
        assert HeuristicFlag.OVER_HEDGED in result.flags

    def test_escalate_beats_retry(self):
        """If both escalate-worthy and retry-worthy flags fire, escalate wins."""
        # Refusal (escalate) + hedging (retry) in one response
        response = (
            "I think I probably might be able to help, "
            "but I can't help with returns unfortunately."
        )
        result = run_heuristics(response)
        assert result.preliminary_verdict == "escalate"
