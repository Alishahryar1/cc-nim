"""Anthropic protocol helpers shared across API, providers, and integrations."""

from .content import extract_text_from_content, get_block_attr, get_block_type
from .conversion import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)
from .errors import (
    append_request_id,
    format_user_error_preview,
    get_user_facing_error_message,
)
from .native_messages_request import sanitize_native_messages_thinking_policy
from .provider_stream_error import iter_provider_stream_error_sse_events
from .sse import (
    ContentBlockManager,
    SSEBuilder,
    _strip_markdown_fences,
    format_sse_event,
    map_stop_reason,
)
from .thinking import ContentChunk, ContentType, ThinkTagParser
from .tokens import get_token_count
from .tools import HeuristicToolParser, _KNOWN_TOOL_NAMES
from .utils import set_if_not_none

__all__ = [
    "AnthropicToOpenAIConverter",
    "ContentBlockManager",
    "ContentChunk",
    "ContentType",
    "HeuristicToolParser",
    "OpenAIConversionError",
    "ReasoningReplayMode",
    "SSEBuilder",
    "ThinkTagParser",
    "append_request_id",
    "build_base_request_body",
    "extract_text_from_content",
    "format_sse_event",
    "format_user_error_preview",
    "get_block_attr",
    "get_block_type",
    "get_token_count",
    "get_user_facing_error_message",
    "iter_provider_stream_error_sse_events",
    "map_stop_reason",
    "sanitize_native_messages_thinking_policy",
    "set_if_not_none",
]
