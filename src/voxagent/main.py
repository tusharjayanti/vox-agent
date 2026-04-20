import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from voxagent import agent
from voxagent.config import get_provider, settings
from voxagent.llm import LLMProvider
from voxagent.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("voxagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = get_provider(settings)
    logger.info(
        "vox-agent started (llm_provider=%s, host=%s, port=%s)",
        settings.llm_provider, settings.host, settings.port,
    )
    yield
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
) -> ChatResponse:
    """Handle a chat turn via the agent orchestrator.

    The agent handles generate → evaluate → retry → fallback. This
    endpoint's only job is to wire the HTTP layer: call the agent,
    set observability headers, return the reply.
    """
    provider: LLMProvider = app.state.provider

    result = await agent.chat(
        session_id=request.session_id,
        user_message=request.message,
        provider=provider,
        settings=settings,
    )

    # Observability headers — surface eval signals to the client
    # without requiring a Postgres query
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
        reply=result.reply,
        turn_id=0,  # Postgres-backed turn_id in Phase 1.8
    )
