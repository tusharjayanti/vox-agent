"""
Heuristic evaluator — the first layer of response evaluation.

Four deterministic, sync, pure-function checks that run on every
agent response before any LLM-as-judge call. They catch obvious
failures (empty responses, refusals, heavy hedging, off-topic
content) for free, and short-circuit the judge call entirely when
a response is clearly bad.

Rationale for heuristics:
- Free (no API call), fast (microseconds), deterministic
- Explainable — flags are concrete thresholds, not black-box scores
- Short-circuit layer: if heuristics escalate, we skip the judge

The LLM judge layer (Phase 1.6) handles subtler failures that
heuristics can't catch — cases where a response "looks fine" but
is actually hallucinated, ungrounded, or semantically off.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from voxagent.llm import LLMProvider
from voxagent.prompts import JUDGE_SYSTEM_PROMPT


class HeuristicFlag(str, Enum):
    """Flags raised by individual heuristic checks."""
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    UNEXPECTED_REFUSAL = "unexpected_refusal"
    OVER_HEDGED = "over_hedged"
    OFF_TOPIC = "off_topic"


@dataclass
class HeuristicResult:
    """Result of running all heuristics on a response.

    preliminary_verdict is one of:
    - "good":     no flags raised
    - "retry":    minor issues that a retry might fix (too short/long, over-hedged)
    - "escalate": serious issues that a retry won't fix (refusal, off-topic)

    This verdict is "preliminary" because the combined evaluator
    (heuristics + LLM judge) makes the final call in Phase 1.6.
    """
    flags: list[HeuristicFlag] = field(default_factory=list)
    passed: bool = True
    preliminary_verdict: str = "good"


# ─── Individual check thresholds ───
# Tunable — these values are defensible defaults, not magic numbers.
# In Phase 3 we'll validate them against canned cases in the
# meta-eval harness.

MIN_RESPONSE_LENGTH = 20
MAX_RESPONSE_LENGTH = 2000
HEDGE_THRESHOLD = 3  # 3+ hedge phrases → flag

REFUSAL_PHRASES = [
    "i can't help",
    "i'm not able to",
    "i cannot assist",
    "i'm unable to",
    "i'm not allowed to",
    "that's outside my scope",
    "i don't have the ability to",
]

HEDGE_PHRASES = [
    "i think",
    "probably",
    "i believe",
    "might be",
    "i'm not sure",
    "i'm not certain",
    "possibly",
    "could be",
    "perhaps",
    "maybe",
]

# Minimal set of support-related terms. A response with NO words from
# this set is almost certainly off-topic.
SUPPORT_VOCABULARY = [
    "order", "return", "ship", "shipping", "deliver", "delivery",
    "refund", "exchange", "product", "item", "account", "password",
    "payment", "billing", "purchase", "customer", "help", "support",
    "acme", "store", "policy", "damaged", "tracking",
]


# ─── Individual checks ───

def check_length(response: str) -> list[HeuristicFlag]:
    """Flag responses that are suspiciously short or long."""
    flags: list[HeuristicFlag] = []
    stripped = response.strip()
    if len(stripped) < MIN_RESPONSE_LENGTH:
        flags.append(HeuristicFlag.TOO_SHORT)
    if len(stripped) > MAX_RESPONSE_LENGTH:
        flags.append(HeuristicFlag.TOO_LONG)
    return flags


def check_refusal(response: str) -> list[HeuristicFlag]:
    """Flag responses where the agent declined to help.

    Customer support agents should help, not refuse. If the agent
    says "I can't help with that" to a reasonable support question,
    something went wrong in the generator — escalate rather than
    retry.
    """
    lower = response.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return [HeuristicFlag.UNEXPECTED_REFUSAL]
    return []


def check_hedging(response: str) -> list[HeuristicFlag]:
    """Flag responses with excessive hedging.

    Some hedging is appropriate when the agent is genuinely unsure.
    But three or more hedge phrases in a single response usually
    means the model is guessing — which is exactly what we want
    to retry with a stricter prompt.
    """
    lower = response.lower()
    count = sum(1 for phrase in HEDGE_PHRASES if phrase in lower)
    if count >= HEDGE_THRESHOLD:
        return [HeuristicFlag.OVER_HEDGED]
    return []


def check_off_topic(response: str) -> list[HeuristicFlag]:
    """Flag responses that contain NO support-related vocabulary.

    A response that doesn't mention orders, returns, shipping,
    accounts, or anything customer-support-adjacent is almost
    certainly off-topic. Escalate rather than retry — retrying
    probably won't help if the model is misunderstanding the
    domain entirely.
    """
    lower = response.lower()
    if any(word in lower for word in SUPPORT_VOCABULARY):
        return []
    return [HeuristicFlag.OFF_TOPIC]


# ─── Orchestrator ───

def run_heuristics(response: str) -> HeuristicResult:
    """Run all heuristic checks and produce a combined result.

    Verdict rules:
    - UNEXPECTED_REFUSAL → "escalate" (a retry won't fix a refusal)
    - OFF_TOPIC         → "escalate" (wrong domain entirely)
    - TOO_SHORT / TOO_LONG / OVER_HEDGED → "retry" (fixable with a
                                           stricter re-prompt)
    - No flags → "good"

    When multiple flags fire, the most severe wins (escalate > retry > good).
    """
    flags: list[HeuristicFlag] = []
    flags.extend(check_length(response))
    flags.extend(check_refusal(response))
    flags.extend(check_hedging(response))
    flags.extend(check_off_topic(response))

    if HeuristicFlag.UNEXPECTED_REFUSAL in flags or HeuristicFlag.OFF_TOPIC in flags:
        verdict = "escalate"
    elif flags:
        verdict = "retry"
    else:
        verdict = "good"

    return HeuristicResult(
        flags=flags,
        passed=(verdict == "good"),
        preliminary_verdict=verdict,
    )


# ─── Judge layer ───

class JudgeVerdict(BaseModel):
    """Structured output from the LLM judge.

    Ranges are enforced by Pydantic — if the judge returns a relevance
    of 7 or a hallucination_risk of 'medium-ish', Pydantic raises
    ValidationError and run_evaluation() falls back to heuristics only.
    """
    relevance: int = Field(ge=0, le=5)
    groundedness: int = Field(ge=0, le=5)
    hallucination_risk: Literal["low", "medium", "high"]
    reasoning: str


@dataclass
class EvaluationResult:
    """Combined result of heuristic + judge evaluation."""
    heuristic: HeuristicResult
    judge: JudgeVerdict | None
    final_verdict: str  # "good" | "retry" | "escalate"
    judge_model: str | None = None  # name of the model used for judging


def _combine(
    heuristic: HeuristicResult,
    judge: JudgeVerdict | None,
) -> str:
    """Combine heuristic and judge signals into a final verdict.

    Priority:
    1. Heuristic "escalate" (refusal, off-topic) — always wins
    2. Judge says hallucination_risk="high" — escalate
    3. Judge says relevance < 3 or groundedness < 3 — retry
    4. Heuristic "retry" (length, hedging) — retry
    5. Otherwise — good
    """
    if heuristic.preliminary_verdict == "escalate":
        return "escalate"

    if judge is not None:
        if judge.hallucination_risk == "high":
            return "escalate"
        if judge.relevance < 3 or judge.groundedness < 3:
            return "retry"

    if heuristic.preliminary_verdict == "retry":
        return "retry"

    return "good"


async def run_evaluation(
    user_message: str,
    agent_response: str,
    provider: LLMProvider,
    judge_model: str,
    judge_temperature: float = 0.0,
) -> EvaluationResult:
    """Run the two-layer evaluator.

    Heuristics first (free, deterministic). If they escalate,
    skip the judge entirely — saves Haiku tokens when a response
    is clearly bad.

    Otherwise, call the judge. If the judge call fails or returns
    malformed JSON, falls back to heuristics-only (judge=None).
    """
    heuristic = run_heuristics(agent_response)

    # Short-circuit: if heuristics already say escalate, don't pay for the judge.
    if heuristic.preliminary_verdict == "escalate":
        return EvaluationResult(
            heuristic=heuristic,
            judge=None,
            final_verdict="escalate",
            judge_model=None,
        )

    # Otherwise, call the judge.
    judge: JudgeVerdict | None
    try:
        raw = await provider.judge(
            user_message=user_message,
            agent_response=agent_response,
            system=JUDGE_SYSTEM_PROMPT,
            model=judge_model,
            temperature=judge_temperature,
            response_schema=JudgeVerdict,
        )
        # provider.judge returns BaseModel | None; narrow to JudgeVerdict
        judge = raw if isinstance(raw, JudgeVerdict) else None
    except Exception:
        # Judge call failed — fall back to heuristics only.
        judge = None

    final = _combine(heuristic, judge)
    return EvaluationResult(
        heuristic=heuristic,
        judge=judge,
        final_verdict=final,
        judge_model=judge_model if judge is not None else None,
    )
