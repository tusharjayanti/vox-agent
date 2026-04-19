from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turn_id: int


class HealthResponse(BaseModel):
    status: str
