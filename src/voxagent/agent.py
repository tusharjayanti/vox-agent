"""
Agent orchestrator — the one public entry point for a full chat turn.

agent.chat() handles the complete request lifecycle:
  1. Load conversation history from memory
  2. Call provider.generate() with history + user message
  3. Evaluate the response (heuristics + LLM judge)
  4. If verdict is 'retry', build a failure-specific correction
     prompt and re-generate once (bounded by settings.max_retries)
  5. If final verdict is 'escalate', return a hardcoded fallback
  6. Save the final assistant response to memory
  7. Return an AgentResult with everything the caller needs

This module is the single coherent place for the retry/fallback
story. main.py just delegates; future phases (Phase 1.8 Postgres
logging) hook in here.
"""
from dataclasses import dataclass

from voxagent.config import Settings, get_generator_model, get_judge_model
from voxagent.evaluator import (
    EvaluationResult,
    HeuristicFlag,
    run_evaluation,
)
from voxagent.llm import LLMProvider, LLMResponse, Message
from voxagent.memory import memory
from voxagent.prompts import GENERATOR_SYSTEM_PROMPT


# Hardcoded fallback used when a response can't be saved by retry.
# NOT an LLM call — deterministic, safe, always available.
FALLBACK_MESSAGE = (
    "I apologize, but I wasn't able to provide a helpful response to "
    "your question. Please try rephrasing, or contact our support team "
    "at support@acmestore.com."
)


@dataclass
class AgentResult:
    """The complete outcome of an agent.chat() call.

    Carries everything the caller (main.py, tests, future Postgres
    logging) might need — the final reply, the verdict, whether a
    retry happened, and the full evaluation object for observability.
    """
    reply: str
    final_verdict: str  # "good" | "retry" | "escalate"
    retry_count: int
    evaluation: EvaluationResult
    # Token accounting for the whole turn (generate + possible retry)
    total_input_tokens: int
    total_output_tokens: int


def _build_retry_correction(evaluation: EvaluationResult) -> str:
    """Build a failure-specific correction instruction for the retry.

    Uses the heuristic flags and judge reasoning to tell the model
    exactly what to fix. Much more effective than a generic 'please
    try again' — the first attempt's failure mode is the retry's
    starting signal.
    """
    corrections: list[str] = []

    flags = evaluation.heuristic.flags
    if HeuristicFlag.TOO_SHORT in flags:
        corrections.append("Provide a more complete answer.")
    if HeuristicFlag.TOO_LONG in flags:
        corrections.append("Be more concise.")
    if HeuristicFlag.OVER_HEDGED in flags:
        corrections.append(
            "Be direct. Avoid phrases like 'I think', 'probably', 'might be'."
        )

    judge = evaluation.judge
    if judge is not None:
        if judge.hallucination_risk == "high":
            corrections.append(
                "Your previous response contained unverified specifics. "
                "Stick to information you know is true, or acknowledge "
                "uncertainty."
            )
        if judge.relevance < 3:
            corrections.append(
                "Address the user's actual question more directly."
            )
        if judge.groundedness < 3:
            corrections.append(
                "Ground your answer in clearly stated policies. "
                "Don't invent specifics."
            )

    if not corrections:
        # No specific signal — generic retry. Should be rare because
        # we only retry when there's a reason to.
        corrections.append("Please try a different approach.")

    return (
        "Your previous response needs improvement. "
        + " ".join(corrections)
        + " Please answer the user's question again with these corrections."
    )


async def chat(
    session_id: str,
    user_message: str,
    provider: LLMProvider,
    settings: Settings,
) -> AgentResult:
    """Run a full chat turn: generate → evaluate → retry? → return.

    This is the single orchestration entry point. main.py's /chat
    route should delegate to this and do nothing else beyond setting
    HTTP headers.
    """
    # 1. Load history and add the user's new message to it
    memory.add(session_id, "user", user_message)
    history = memory.get(session_id)

    generator_model = get_generator_model(settings)
    judge_model = get_judge_model(settings)

    # 2. First generation attempt
    response: LLMResponse = await provider.generate(
        messages=history,
        system=GENERATOR_SYSTEM_PROMPT,
        model=generator_model,
        temperature=settings.generator_temperature,
        max_tokens=settings.generator_max_tokens,
    )
    total_input_tokens = response.input_tokens
    total_output_tokens = response.output_tokens

    # 3. Evaluate the first attempt
    evaluation = await run_evaluation(
        user_message=user_message,
        agent_response=response.content,
        provider=provider,
        judge_model=judge_model,
        judge_temperature=settings.judge_temperature,
    )

    retry_count = 0

    # 4. Retry loop — bounded by settings.max_retries
    while (
        evaluation.final_verdict == "retry"
        and retry_count < settings.max_retries
    ):
        # Build correction message and append to history
        correction = _build_retry_correction(evaluation)
        retry_history = history + [
            Message(role="assistant", content=response.content),
            Message(role="user", content=correction),
        ]

        response = await provider.generate(
            messages=retry_history,
            system=GENERATOR_SYSTEM_PROMPT,
            model=generator_model,
            temperature=settings.generator_temperature,
            max_tokens=settings.generator_max_tokens,
        )
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        evaluation = await run_evaluation(
            user_message=user_message,
            agent_response=response.content,
            provider=provider,
            judge_model=judge_model,
            judge_temperature=settings.judge_temperature,
        )
        retry_count += 1

    # 5. Decide final reply
    if evaluation.final_verdict == "escalate":
        reply = FALLBACK_MESSAGE
    else:
        reply = response.content

    # 6. Save assistant response to memory
    memory.add(session_id, "assistant", reply)

    return AgentResult(
        reply=reply,
        final_verdict=evaluation.final_verdict,
        retry_count=retry_count,
        evaluation=evaluation,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )
