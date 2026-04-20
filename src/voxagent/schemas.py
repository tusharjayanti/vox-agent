from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turn_id: int


class HealthResponse(BaseModel):
    status: str
