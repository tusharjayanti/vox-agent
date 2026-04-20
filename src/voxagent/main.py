import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Response

from voxagent import agent, db
from voxagent.config import get_provider, settings
from voxagent.llm import LLMProvider
from voxagent.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("voxagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = get_provider(settings)
    app.state.pool = await db.create_pool(settings.postgres_dsn)
    logger.info(
        "vox-agent started (llm_provider=%s, host=%s, port=%s)",
        settings.llm_provider, settings.host, settings.port,
    )
    yield
    await db.close_pool(app.state.pool)
    logger.info("vox-agent shutting down")


app = FastAPI(
    title="vox-agent",
    description="Customer-support agent with inline evaluation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    response: Response,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    """Handle a chat turn via the agent orchestrator.

    The agent handles generate → evaluate → retry → fallback. This
    endpoint's only job is to wire the HTTP layer: call the agent,
    set observability headers, write to Postgres, return the reply.
    """
    provider: LLMProvider = app.state.provider
    pool = app.state.pool

    # Run the agent's full turn
    result = await agent.chat(
        session_id=request.session_id,
        user_message=request.message,
        provider=provider,
        settings=settings,
    )

    # Write to Postgres — turn row needs to be written synchronously
    # because the fallback message references the turn_id.
    conversation_id = await db.get_or_create_conversation(
        pool, request.session_id
    )
    await db.insert_turn(
        pool, conversation_id, "user", request.message,
    )
    assistant_turn_id = await db.insert_turn(
        pool, conversation_id, "assistant", result.reply,
        latency_ms=None,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        retry_count=result.retry_count,
    )

    # If the reply is the fallback, update it with the real turn_id
    final_reply = result.reply
    if result.final_verdict == "escalate":
        final_reply = agent.format_fallback_message(
            turn_id=assistant_turn_id
        )
        await db.update_turn_content(
            pool, assistant_turn_id, final_reply
        )

    # Evaluation logging happens in background — doesn't block response
    background_tasks.add_task(
        db.insert_evaluation,
        pool, assistant_turn_id, result.evaluation,
        settings.llm_provider,
    )

    # Observability headers
    response.headers["X-Verdict"] = result.final_verdict
    response.headers["X-Retry-Count"] = str(result.retry_count)
    if result.evaluation.judge is not None:
        response.headers["X-Eval-Relevance"] = str(
            result.evaluation.judge.relevance
        )
        response.headers["X-Eval-Groundedness"] = str(
            result.evaluation.judge.groundedness
        )
        response.headers["X-Hallucination-Risk"] = (
            result.evaluation.judge.hallucination_risk
        )

    return ChatResponse(
        session_id=request.session_id,
        reply=final_reply,
        turn_id=assistant_turn_id,
    )
