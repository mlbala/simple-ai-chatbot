"""HTTP routes: chat (JSON and streaming) and health."""

import asyncio
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ..config import Settings
from ..errors import to_chat_error
from ..graph import CHATBOT_NODE, ChatGraph, State
from ..schemas import ChatRequest, ChatResponse, DoneEvent, HealthResponse, TokenEvent

router = APIRouter()


def get_graph(request: Request) -> ChatGraph:
    return request.app.state.graph


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


GraphDep = Annotated[ChatGraph, Depends(get_graph)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/api/chat")
async def chat(body: ChatRequest, graph: GraphDep, settings: SettingsDep) -> ChatResponse:
    """Send a message and wait for the full reply."""
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        async with asyncio.timeout(settings.request_timeout_s):
            result = await graph.ainvoke(_graph_input(body), _thread_config(thread_id))
    except Exception as exc:
        raise to_chat_error(exc) from exc
    return ChatResponse(reply=result["messages"][-1].text, thread_id=thread_id)


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(body: ChatRequest, graph: GraphDep) -> AsyncIterable[ServerSentEvent]:
    """Send a message and stream the reply as Server-Sent Events.

    Events: `token` (`{"content"}`), then either `done` (`{"thread_id"}`) or
    `error` (`{"error": {"code", "message"}}`).
    """
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        async for token in _reply_tokens(graph, body, thread_id):
            yield ServerSentEvent(event="token", data=TokenEvent(content=token))
    except Exception as exc:  # noqa: BLE001
        # The response (status 200) has already started, so any error is sent as the last event instead.
        yield ServerSentEvent(event="error", data=to_chat_error(exc).to_dict())
        return
    yield ServerSentEvent(event="done", data=DoneEvent(thread_id=thread_id))


def _graph_input(body: ChatRequest) -> State:
    # add_messages replaces a message whose ID already exists, so a retried
    # message_id overwrites the failed attempt instead of adding a copy.
    return {"messages": [HumanMessage(body.message, id=body.message_id)]}


def _thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def _reply_tokens(graph: ChatGraph, body: ChatRequest, thread_id: str) -> AsyncIterator[str]:
    """Yield the chatbot's reply as text chunks while the model streams it."""
    stream = graph.astream(_graph_input(body), _thread_config(thread_id), stream_mode="messages", version="v2")
    async for part in stream:
        if part["type"] != "messages":
            continue
        message, metadata = part["data"]
        if metadata.get("langgraph_node") == CHATBOT_NODE and isinstance(message, AIMessage) and message.text:
            yield message.text
