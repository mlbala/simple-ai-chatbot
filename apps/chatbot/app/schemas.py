"""Request, response and event models for the HTTP API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator
from pydantic_core import PydanticCustomError

MAX_MESSAGE_CHARS = 4000

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    thread_id: Identifier | None = Field(
        default=None, description="Conversation to continue. A new one is started when omitted."
    )
    message_id: Identifier | None = Field(
        default=None,
        description="Optional ID for this message. Resending a failed message with the same ID "
        "replaces it in the conversation instead of adding a copy.",
    )

    @field_validator("message")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise PydanticCustomError("blank_message", "Message must not be blank")
        return value


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TokenEvent(BaseModel):
    content: str


class DoneEvent(BaseModel):
    thread_id: str
