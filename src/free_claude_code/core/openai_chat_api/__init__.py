"""Inbound OpenAI Chat Completions protocol adapter.

FCC speaks Anthropic Messages internally. This package translates the widely
adopted Chat Completions wire format onto that pipeline so any OpenAI-compatible
client can drive FCC's configured providers.
"""

from .models import (
    ChatCompletionsRequest,
    ChatFunctionDef,
    ChatMessage,
    ChatTool,
    ChatToolCall,
)
from .translate import (
    anthropic_message_to_chat_completion,
    anthropic_sse_to_chat_sse,
    chat_request_to_messages_request,
    iter_sse_data,
    new_completion_id,
)

__all__ = [
    "ChatCompletionsRequest",
    "ChatFunctionDef",
    "ChatMessage",
    "ChatTool",
    "ChatToolCall",
    "anthropic_message_to_chat_completion",
    "anthropic_sse_to_chat_sse",
    "chat_request_to_messages_request",
    "iter_sse_data",
    "new_completion_id",
]
