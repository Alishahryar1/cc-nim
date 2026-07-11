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
    anthropic_error_type_for_failure,
    anthropic_failure_payload,
    anthropic_status_for_error_type,
    execution_failure_from_anthropic_error,
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
    "anthropic_error_type_for_failure",
    "anthropic_failure_payload",
    "anthropic_request_snapshot",
    "anthropic_status_for_error_type",
    "build_base_request_body",
    "execution_failure_from_anthropic_error",
    "extract_text_from_content",
    "format_sse_event",
    "get_block_attr",
    "get_block_type",
    "get_token_count",
    "map_stop_reason",
    "sanitize_native_messages_thinking_policy",
    "serialize_tool_result_content",
    "set_if_not_none",
]
