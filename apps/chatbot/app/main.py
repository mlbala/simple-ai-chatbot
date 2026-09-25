"""ASGI entrypoint. Run with: `uv run uvicorn apps.chatbot.app.main:app --reload`."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from .api.routes import router
from .config import Settings, load_settings
from .errors import install_error_handlers
from .graph import build_graph
from .llm import build_llm

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, llm: BaseChatModel | None = None) -> FastAPI:
    """Build the app. Pass `llm` to use a different chat model (tests pass a fake one)."""
    if settings is None:
        settings = load_settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once per process and shared by all requests.
        app.state.settings = settings
        app.state.graph = build_graph(
            llm if llm is not None else build_llm(settings),
            InMemorySaver(),  # conversations live in this process only; no database
            system_prompt=settings.system_prompt,
            max_history_messages=settings.max_history_messages,
        )
        yield

    app = FastAPI(title="Simple AI Chatbot", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    install_error_handlers(app)
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


app = create_app()
