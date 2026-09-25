import os

import pytest
from pydantic import SecretStr

from apps.chatbot.app.config import Settings

from .fakes import TEST_API_KEY, TEST_SYSTEM_PROMPT

# Importing apps.chatbot.app.main builds the module-level `app`, which needs a key.
# It is never started in tests; each test builds its own app with a fake model.
os.environ.setdefault("GROQ_API_KEY", TEST_API_KEY)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        groq_api_key=SecretStr(TEST_API_KEY),
        system_prompt=TEST_SYSTEM_PROMPT,
        request_timeout_s=5,
    )
