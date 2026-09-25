"""Application settings, read from environment variables and `.env`."""

from typing import Annotated, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer clearly and concisely, "
    "and use Markdown when formatting makes the answer easier to read."
)


class ConfigError(RuntimeError):
    """Raised at startup when a setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `GROQ_API_KEY=` counts as missing, not as a key
        extra="ignore",
    )

    groq_api_key: SecretStr
    model_name: str = "groq:qwen/qwen3.8-27b"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_messages: int = Field(default=7, ge=1)
    request_timeout_s: float = Field(default=60.0, gt=0)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"]
    )
    log_level: LogLevel = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


def load_settings() -> Settings:
    """Load settings, or stop with a short message that names each bad setting."""
    try:
        return Settings()  # pyright: ignore[reportCallIssue]  (fields come from the environment)
    except ValidationError as exc:
        # For example: "GROQ_API_KEY: Field required". Setting values are never printed.
        problems = [f"{str(error['loc'][0]).upper()}: {error['msg']}" for error in exc.errors()]
        message = "Missing or invalid settings. Set them in .env or as environment variables:"
        # `from None` hides pydantic's longer version of the same error.
        raise ConfigError(message + "\n  - " + "\n  - ".join(problems)) from None
