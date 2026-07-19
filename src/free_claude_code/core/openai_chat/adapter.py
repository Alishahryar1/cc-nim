"""Facade for OpenAI Chat Completions protocol adaptation."""

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, ClassVar

from free_claude_code.core.openai_responses import openai_error_payload

from .completion import anthropic_message_to_chat_completion
from .errors import ChatCompletionsConversionError
from .events import OPENAI_CHAT_SSE_HEADERS
from .input import convert_request_to_anthropic_payload
from .models import OpenAIChatCompletionsRequest
from .streaming import (
    PostStartTerminalFailureObserver,
    iter_chat_completions_sse_from_anthropic,
)


class OpenAIChatAdapter:
    """Convert between OpenAI Chat Completions and the proxy's Anthropic core path."""

    ConversionError: ClassVar[type[ChatCompletionsConversionError]] = (
        ChatCompletionsConversionError
    )
    sse_headers: ClassVar[dict[str, str]] = OPENAI_CHAT_SSE_HEADERS

    def to_anthropic_payload(
        self, request: OpenAIChatCompletionsRequest
    ) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def iter_sse_from_anthropic(
        self,
        chunks: AsyncIterable[Any],
        request: OpenAIChatCompletionsRequest,
        *,
        on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
    ) -> AsyncIterator[str]:
        return iter_chat_completions_sse_from_anthropic(
            chunks,
            request,
            on_post_start_terminal_failure=on_post_start_terminal_failure,
        )

    def message_to_completion(
        self,
        message: dict[str, Any],
        request: OpenAIChatCompletionsRequest,
        *,
        completion_id: str | None = None,
    ) -> dict[str, Any]:
        return anthropic_message_to_chat_completion(
            message, request, completion_id=completion_id
        )

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)
