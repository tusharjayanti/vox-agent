import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from voxagent.config import get_generator_model, get_provider, settings
from voxagent.llm import LLMProvider, Message
from voxagent.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("voxagent")

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for Acme Store. "
    "Answer questions about orders, returns (30-day window), "
    "shipping (3-5 days standard), and account issues. "
    "If you are unsure, say so rather than guessing."
)


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
async def chat(request: ChatRequest) -> ChatResponse:
    provider: LLMProvider = app.state.provider
    # No memory yet (Step 5) — single-turn for now
    messages = [Message(role="user", content=request.message)]
    response = await provider.generate(
        messages=messages,
        system=SYSTEM_PROMPT,
        model=get_generator_model(settings),
        temperature=settings.generator_temperature,
        max_tokens=settings.generator_max_tokens,
    )
    return ChatResponse(
        session_id=request.session_id,
        reply=response.content,
        turn_id=0,  # Postgres-backed turn_id comes in Step 9
    )
