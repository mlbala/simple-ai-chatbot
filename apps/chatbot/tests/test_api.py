import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import groq
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from langchain_core.language_models import BaseChatModel

from apps.chatbot.app.config import Settings
from apps.chatbot.app.main import create_app

from .fakes import (
    TEST_API_KEY,
    TEST_SYSTEM_PROMPT,
    FakeChatModel,
    fake_llm,
    groq_status_error,
    groq_timeout_error,
)

REPLY = "Hello from the **fake** model!"


@asynccontextmanager
async def client_for(app: FastAPI) -> AsyncIterator[AsyncClient]:
    # ASGITransport doesn't run lifespan events, so enter the lifespan explicitly.
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
    ):
        yield client


@pytest.fixture
def llm() -> FakeChatModel:
    return fake_llm(REPLY, "Second reply")


@pytest.fixture
async def client(settings: Settings, llm: BaseChatModel) -> AsyncIterator[AsyncClient]:
    async with client_for(create_app(settings, llm)) as client:
        yield client


def parse_sse(response: Response) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


# --- health and UI ---------------------------------------------------------------------


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ui_is_served_at_root(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert "<title>Simple AI Chatbot</title>" in response.text


# --- POST /api/chat --------------------------------------------------------------------


async def test_chat_returns_reply_and_new_thread_id(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == REPLY
    assert uuid.UUID(body["thread_id"])


async def test_follow_up_in_the_same_thread_sees_earlier_messages(client: AsyncClient, llm: FakeChatModel) -> None:
    first = await client.post("/api/chat", json={"message": "Hi"})
    thread_id = first.json()["thread_id"]

    second = await client.post("/api/chat", json={"message": "And again", "thread_id": thread_id})

    assert second.json() == {"reply": "Second reply", "thread_id": thread_id}
    assert [m.text for m in llm.calls[1]] == [TEST_SYSTEM_PROMPT, "Hi", REPLY, "And again"]


async def test_requests_without_thread_id_start_new_conversations(client: AsyncClient, llm: FakeChatModel) -> None:
    await client.post("/api/chat", json={"message": "Hi"})
    await client.post("/api/chat", json={"message": "And again"})

    assert [m.text for m in llm.calls[1]] == [TEST_SYSTEM_PROMPT, "And again"]


async def test_retry_with_same_message_id_does_not_duplicate_it(settings: Settings) -> None:
    llm = fake_llm("Recovered", "Next", errors=[groq_timeout_error()])
    payload = {"message": "Hi", "thread_id": "t-retry", "message_id": "m-1"}
    async with client_for(create_app(settings, llm)) as client:
        failed = await client.post("/api/chat", json=payload)
        retried = await client.post("/api/chat", json=payload)
        await client.post("/api/chat", json={"message": "Next?", "thread_id": "t-retry"})

    assert failed.status_code == 504
    assert retried.json()["reply"] == "Recovered"
    assert [m.text for m in llm.calls[-1]] == [TEST_SYSTEM_PROMPT, "Hi", "Recovered", "Next?"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="missing message"),
        pytest.param({"message": ""}, id="empty message"),
        pytest.param({"message": "   \n "}, id="blank message"),
        pytest.param({"message": "x" * 4001}, id="message too long"),
        pytest.param({"message": 42}, id="wrong type"),
        pytest.param({"message": "Hi", "thread_id": "../../etc"}, id="bad thread_id"),
    ],
)
async def test_chat_rejects_invalid_input(client: AsyncClient, llm: FakeChatModel, payload: dict[str, Any]) -> None:
    response = await client.post("/api/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert llm.calls == []


async def test_chat_accepts_message_at_max_length(client: AsyncClient) -> None:
    response = await client.post("/api/chat", json={"message": "x" * 4000})

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        pytest.param(groq_status_error(groq.AuthenticationError, 401), 502, "llm_auth_failed", id="auth"),
        pytest.param(groq_status_error(groq.NotFoundError, 404), 502, "llm_model_not_found", id="not found"),
        pytest.param(groq_status_error(groq.RateLimitError, 429), 429, "llm_rate_limited", id="rate limit"),
        pytest.param(groq_timeout_error(), 504, "llm_timeout", id="timeout"),
        pytest.param(groq_status_error(groq.InternalServerError, 503), 502, "llm_error", id="upstream 5xx"),
        pytest.param(RuntimeError(f"bug near {TEST_API_KEY}"), 500, "internal_error", id="unexpected"),
    ],
)
async def test_llm_failures_map_to_clean_json_errors(
    settings: Settings, error: Exception, status: int, code: str
) -> None:
    async with client_for(create_app(settings, fake_llm(errors=[error]))) as client:
        response = await client.post("/api/chat", json={"message": "Hi"})

    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert set(body["error"]) == {"code", "message"}
    for leaked in (TEST_API_KEY, "Traceback", "upstream said"):
        assert leaked not in response.text


# --- POST /api/chat/stream -------------------------------------------------------------


async def test_stream_emits_token_and_done_events(client: AsyncClient) -> None:
    response = await client.post("/api/chat/stream", json={"message": "Hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response)
    names = [name for name, _ in events]
    assert names[-1] == "done" and set(names[:-1]) == {"token"} and len(names) > 2
    assert "".join(data["content"] for name, data in events if name == "token") == REPLY
    assert uuid.UUID(events[-1][1]["thread_id"])


async def test_stream_continues_the_given_thread(client: AsyncClient, llm: FakeChatModel) -> None:
    first = await client.post("/api/chat", json={"message": "Hi"})
    thread_id = first.json()["thread_id"]

    response = await client.post("/api/chat/stream", json={"message": "More", "thread_id": thread_id})

    assert parse_sse(response)[-1] == ("done", {"thread_id": thread_id})
    assert [m.text for m in llm.calls[1]] == [TEST_SYSTEM_PROMPT, "Hi", REPLY, "More"]


async def test_stream_reports_llm_failure_as_error_event(settings: Settings) -> None:
    error = groq_status_error(groq.RateLimitError, 429)
    async with client_for(create_app(settings, fake_llm(errors=[error]))) as client:
        response = await client.post("/api/chat/stream", json={"message": "Hi"})

    assert response.status_code == 200
    [(name, data)] = parse_sse(response)
    assert name == "error"
    assert data["error"]["code"] == "llm_rate_limited"


async def test_stream_validates_input_before_streaming(client: AsyncClient) -> None:
    response = await client.post("/api/chat/stream", json={"message": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
