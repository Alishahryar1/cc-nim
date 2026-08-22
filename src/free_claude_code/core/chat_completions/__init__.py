"""OpenAI Chat Completions protocol adapter."""

from .converter import ChatCompletionsToAnthropicConverter
from .models import ChatCompletionsRequest
from .stream import chat_completions_sse_from_anthropic

__all__ = [
    "ChatCompletionsRequest",
    "ChatCompletionsToAnthropicConverter",
    "chat_completions_sse_from_anthropic",
]
