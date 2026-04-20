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
