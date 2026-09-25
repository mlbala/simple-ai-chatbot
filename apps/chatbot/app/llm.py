"""Chat model factory."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model named by MODEL_NAME, in `provider:model` form."""
    return init_chat_model(
        settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout_s,
        api_key=settings.groq_api_key.get_secret_value(),
    )
