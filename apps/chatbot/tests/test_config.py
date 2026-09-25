from pathlib import Path

import pytest

from apps.chatbot.app.config import ConfigError, load_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run from an empty directory so a developer's real .env is never read.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_missing_api_key_fails_fast_with_a_clear_message() -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY: Field required"):
        load_settings()


def test_empty_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")

    with pytest.raises(ConfigError, match="GROQ_API_KEY: Field required"):
        load_settings()


def test_invalid_values_are_named_in_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    monkeypatch.setenv("TEMPERATURE", "9")

    with pytest.raises(ConfigError, match="TEMPERATURE") as excinfo:
        load_settings()
    assert "gsk_secret_value" not in str(excinfo.value)


def test_settings_parse_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = load_settings()

    assert settings.cors_origins == ["http://a.test", "http://b.test"]
    assert settings.log_level == "DEBUG"
    assert settings.groq_api_key.get_secret_value() == "gsk_secret_value"
    assert "gsk_secret_value" not in repr(settings)
