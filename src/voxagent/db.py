"""
Database layer — asyncpg connection pool and query functions.

Each function is a named, typed operation. No ORM. Queries use
$1, $2 positional placeholders (asyncpg's native convention).

The pool is opened in main.py's lifespan on startup and closed
on shutdown.
"""
import json
import logging

import asyncpg

from voxagent.evaluator import EvaluationResult


logger = logging.getLogger("voxagent")


# ─── Pool lifecycle ───

async def create_pool(dsn: str) -> asyncpg.Pool:
    """Open a new asyncpg connection pool.

    min_size=2, max_size=10 is reasonable for a single-server
    deployment. Tune in production based on concurrent load.
    """
    pool = await asyncpg.create_pool(
        dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("Postgres pool opened (min_size=2, max_size=10)")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Close the pool on application shutdown."""
    await pool.close()
    logger.info("Postgres pool closed")


# ─── Conversations ───

async def get_or_create_conversation(
    pool: asyncpg.Pool,
    session_id: str,
) -> int:
    """Return the conversation ID for this session, creating if absent.

    Uses ON CONFLICT to handle the race where two concurrent turns
    from the same session both see no existing row.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (session_id)
            VALUES ($1)
            ON CONFLICT (session_id) DO UPDATE
                SET session_id = EXCLUDED.session_id
            RETURNING id
            """,
            session_id,
        )
        return row["id"]


# ─── Turns ───

async def insert_turn(
    pool: asyncpg.Pool,
    conversation_id: int,
    role: str,
    content: str,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    retry_count: int = 0,
) -> int:
    """Insert a turn row and return its ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO turns (
                conversation_id, role, content,
                latency_ms, input_tokens, output_tokens, retry_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            conversation_id, role, content,
            latency_ms, input_tokens, output_tokens, retry_count,
        )
        return row["id"]


async def update_turn_content(
    pool: asyncpg.Pool,
    turn_id: int,
    content: str,
) -> None:
    """Update a turn's content (used when fallback message needs the
    real turn_id substituted in)."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE turns SET content = $1 WHERE id = $2",
            content, turn_id,
        )


# ─── Evaluations ───

async def insert_evaluation(
    pool: asyncpg.Pool,
    turn_id: int,
    evaluation: EvaluationResult,
    provider: str,
) -> int:
    """Insert an evaluation row for an assistant turn.

    Handles both judge-present and judge-absent cases. Heuristic flags
    are stored as a JSONB array for flexible querying.
    """
    heuristic_flags_json = json.dumps(
        [flag.value for flag in evaluation.heuristic.flags]
    )

    judge_relevance: int | None = None
    judge_groundedness: int | None = None
    judge_hallucination_risk: str | None = None
    judge_reasoning: str | None = None

    if evaluation.judge is not None:
        judge_relevance = evaluation.judge.relevance
        judge_groundedness = evaluation.judge.groundedness
        judge_hallucination_risk = evaluation.judge.hallucination_risk
        judge_reasoning = evaluation.judge.reasoning

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO evaluations (
                turn_id, verdict,
                heuristic_flags, heuristic_verdict,
                judge_relevance, judge_groundedness,
                judge_hallucination_risk, judge_reasoning, judge_model,
                provider
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            turn_id,
            evaluation.final_verdict,
            heuristic_flags_json,
            evaluation.heuristic.preliminary_verdict,
            judge_relevance,
            judge_groundedness,
            judge_hallucination_risk,
            judge_reasoning,
            evaluation.judge_model,
            provider,
        )
        return row["id"]
