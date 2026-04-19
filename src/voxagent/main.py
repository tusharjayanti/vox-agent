import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from voxagent.config import settings
from voxagent.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger("voxagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging here (after uvicorn has set up its own handlers)
    # so our log line actually prints.
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,  # override uvicorn's config
    )
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
    return ChatResponse(
        session_id=request.session_id,
        reply="Hello from vox-agent. Hardcoded for now — LLM coming next.",
        turn_id=0,
    )
