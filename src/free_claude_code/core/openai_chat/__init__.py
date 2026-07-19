"""OpenAI Chat Completions protocol adapter."""

from .adapter import OpenAIChatAdapter
from .errors import ChatCompletionsConversionError
from .models import OpenAIChatCompletionsRequest

__all__ = [
    "ChatCompletionsConversionError",
    "OpenAIChatAdapter",
    "OpenAIChatCompletionsRequest",
]
