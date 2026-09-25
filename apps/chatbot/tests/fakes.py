"""Test doubles. Tests use these instead of Groq, so they never touch the network."""

from typing import Any

import groq
import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

TEST_API_KEY = "test-key-not-real"
TEST_SYSTEM_PROMPT = "You are a test assistant."


class FakeChatModel(GenericFakeChatModel):
    """Returns canned replies in order and records every prompt it receives.

    Exceptions in `errors` are raised, one per call, before any reply is used.
    Streaming goes through `_generate` too, so it behaves the same way.
    """

    calls: list[list[BaseMessage]] = Field(default_factory=list)
    errors: list[Exception] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.errors:
            raise self.errors.pop(0)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def fake_llm(*replies: str, errors: list[Exception] | None = None) -> FakeChatModel:
    return FakeChatModel(messages=iter(replies), errors=errors or [])


def groq_status_error(error_cls: type[groq.APIStatusError], status: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return error_cls(f"upstream said {status}", response=response, body=None)


def groq_timeout_error() -> groq.APITimeoutError:
    return groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
