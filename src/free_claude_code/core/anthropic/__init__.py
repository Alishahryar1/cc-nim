"""Anthropic protocol helpers shared across API, providers, and integrations."""

from .content import extract_text_from_content, get_block_attr, get_block_type
from .conversion import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)
from .errors import (
    anthropic_error_payload,
    anthropic_status_for_error_type,
    append_request_id,
    format_user_error_preview,
    get_user_facing_error_message,
    redact_sensitive_error_text,
)
from .models import (
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockRedactedThinking,
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ContentBlockWebFetchToolResult,
    ContentBlockWebSearchToolResult,
    Message,
    MessagesRequest,
    MessagesResponse,
    Role,
    SystemContent,
    ThinkingConfig,
    TokenCountRequest,
    TokenCountResponse,
    Tool,
    Usage,
)
from .native_messages_request import sanitize_native_messages_thinking_policy
from .request_serialization import serialize_tool_result_content
from .request_snapshot import anthropic_request_snapshot
from .sse_aggregation import aggregate_anthropic_sse_to_message
from .streaming import (
    AnthropicStreamLedger,
    StreamBlockLedger,
    ToolBlockState,
    format_sse_event,
    map_stop_reason,
)
from .thinking import ContentChunk, ContentType, ThinkTagParser
from .tokens import get_token_count
from .tools import HeuristicToolParser
from .utils import set_if_not_none

__all__ = [
    "AnthropicStreamLedger",
    "AnthropicToOpenAIConverter",
    "ContentBlockDocument",
    "ContentBlockImage",
    "ContentBlockRedactedThinking",
    "ContentBlockServerToolUse",
    "ContentBlockText",
    "ContentBlockThinking",
    "ContentBlockToolResult",
    "ContentBlockToolUse",
    "ContentBlockWebFetchToolResult",
    "ContentBlockWebSearchToolResult",
    "ContentChunk",
    "ContentType",
    "HeuristicToolParser",
    "Message",
    "MessagesRequest",
    "MessagesResponse",
    "OpenAIConversionError",
    "ReasoningReplayMode",
    "Role",
    "StreamBlockLedger",
    "SystemContent",
    "ThinkTagParser",
    "ThinkingConfig",
    "TokenCountRequest",
    "TokenCountResponse",
    "Tool",
    "ToolBlockState",
    "Usage",
    "aggregate_anthropic_sse_to_message",
    "anthropic_error_payload",
    "anthropic_request_snapshot",
    "anthropic_status_for_error_type",
    "append_request_id",
    "build_base_request_body",
    "extract_text_from_content",
    "format_sse_event",
    "format_user_error_preview",
    "get_block_attr",
    "get_block_type",
    "get_token_count",
    "get_user_facing_error_message",
    "map_stop_reason",
    "redact_sensitive_error_text",
    "sanitize_native_messages_thinking_policy",
    "serialize_tool_result_content",
    "set_if_not_none",
]
