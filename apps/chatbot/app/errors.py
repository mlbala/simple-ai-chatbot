"""Error handling for the chat API, all in one place.

Every error the chat API returns has the same shape:

    {"error": {"code": "llm_timeout", "message": "The AI model took too long to respond. Please try again."}}

Users only ever see the short messages written below. The real exception goes to the server log.
"""

import logging

import groq
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ChatError(Exception):
    """A failed chat request: an HTTP status, a short error code and a message that is safe to show users."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def to_chat_error(exc: Exception) -> ChatError:
    """Log why generating a reply failed, and return the ChatError to send back to the user."""
    # APITimeoutError is a kind of APIConnectionError, so it has to be checked first.
    if isinstance(exc, (TimeoutError, groq.APITimeoutError)):
        error = ChatError(504, "llm_timeout", "The AI model took too long to respond. Please try again.")
    elif isinstance(exc, groq.APIConnectionError):
        error = ChatError(503, "llm_unavailable", "Could not reach the AI model provider. Please try again.")
    elif isinstance(exc, groq.RateLimitError):
        error = ChatError(429, "llm_rate_limited", "The AI model is busy. Please wait a moment and try again.")
    elif isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        error = ChatError(502, "llm_auth_failed", "The AI model provider rejected the server's API key.")
    elif isinstance(exc, groq.NotFoundError):
        error = ChatError(502, "llm_model_not_found", "The configured AI model is not available.")
    elif isinstance(exc, groq.APIStatusError):
        error = ChatError(502, "llm_error", "The AI model provider returned an error. Please try again.")
    else:
        # Anything else is probably a bug in this app, so log the full traceback.
        logger.error("Unexpected error while generating a reply", exc_info=exc)
        return ChatError(500, "internal_error", "Something went wrong. Please try again.")

    logger.warning("AI model call failed (%s): %r", error.code, exc)
    return error


def install_error_handlers(app: FastAPI) -> None:
    """Tell FastAPI how to turn errors into the JSON shape above."""

    @app.exception_handler(ChatError)
    async def handle_chat_error(request: Request, exc: ChatError) -> JSONResponse:
        return JSONResponse(exc.to_dict(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def handle_bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # For example: "message: String should have at most 4000 characters"
        message = "; ".join(f"{error['loc'][-1]}: {error['msg']}" for error in exc.errors())
        return JSONResponse({"error": {"code": "validation_error", "message": message}}, status_code=422)
