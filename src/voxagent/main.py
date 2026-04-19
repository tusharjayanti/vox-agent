import logging

from fastapi import FastAPI, Response

from voxagent.config import build_provider, settings
from voxagent.llm import LLMProvider
from voxagent.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="vox-agent")

provider: LLMProvider = build_provider(settings)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, response: Response) -> ChatResponse:
    reply = "This is a hardcoded response. Real agent coming in step 1.3."
    response.headers["X-Verdict"] = "good"
    response.headers["X-Retry-Count"] = "0"
    return ChatResponse(session_id=request.session_id, reply=reply)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
