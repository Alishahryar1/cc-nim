"""Mixins for NVIDIA NIM provider - decoupling responsibilities.

This module contains focused mixins that handle specific aspects of the
NVIDIA NIM provider functionality:
- RequestBuilderMixin: Builds request bodies
- StreamProcessorMixin: Processes streaming responses
- ErrorMapperMixin: Maps HTTP errors to provider exceptions
- ResponseConverterMixin: Converts responses between formats
"""

import json
import logging
from typing import Any, Dict, AsyncIterator

from .utils import AnthropicToOpenAIConverter, map_stop_reason, extract_think_content
from .exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    APIError,
)

logger = logging.getLogger(__name__)


class RequestBuilderMixin:
    """Mixin for building OpenAI-format request bodies.

    Handles conversion from Anthropic request format to OpenAI format,
    including system prompts, tools, thinking mode, and NIM-specific parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nim_params: Dict[str, Any] = {}

    def _load_nim_params(self) -> Dict[str, Any]:
        """Load NIM-specific parameters from environment.

        Reads NVIDIA_NIM_* environment variables to configure defaults.

        Returns:
            Dictionary of NIM-specific parameters
        """
        import os

        params: Dict[str, Any] = {}
        if val := os.getenv("NVIDIA_NIM_TEMPERATURE"):
            params["temperature"] = float(val)
        if val := os.getenv("NVIDIA_NIM_TOP_P"):
            params["top_p"] = float(val)
        if val := os.getenv("NVIDIA_NIM_MAX_TOKENS"):
            params["max_tokens"] = int(val)
        return params

    def _build_request_body(self, request_data: Any, stream: bool = False) -> dict:
        """Build OpenAI-format request body from Anthropic request.

        Args:
            request_data: The incoming Anthropic-format request
            stream: Whether this is a streaming request

        Returns:
            OpenAI-format request body dictionary
        """
        messages = AnthropicToOpenAIConverter.convert_messages(request_data.messages)

        # Add system prompt
        if request_data.system:
            system_msg = AnthropicToOpenAIConverter.convert_system_prompt(
                request_data.system
            )
            if system_msg:
                messages.insert(0, system_msg)

        body = {
            "model": request_data.model,
            "messages": messages,
            "max_tokens": request_data.max_tokens,
            "stream": stream,
        }

        if request_data.temperature is not None:
            body["temperature"] = request_data.temperature
        if request_data.top_p is not None:
            body["top_p"] = request_data.top_p
        if request_data.stop_sequences:
            body["stop"] = request_data.stop_sequences
        if request_data.tools:
            body["tools"] = AnthropicToOpenAIConverter.convert_tools(request_data.tools)

        # Handle thinking/reasoning mode
        extra_body = request_data.extra_body.copy() if request_data.extra_body else {}
        if request_data.thinking and getattr(request_data.thinking, "enabled", True):
            extra_body.setdefault("thinking", {"type": "enabled"})
            extra_body.setdefault("reasoning_split", True)

        body.update(extra_body)

        # Apply NIM defaults
        for key, val in self._nim_params.items():
            if key not in body:
                body[key] = val

        return body


class ErrorMapperMixin:
    """Mixin for mapping HTTP errors to provider exceptions.

    Converts HTTP status codes and error responses to appropriate
    ProviderError subclasses for standardized error handling.
    """

    def _map_error(self, response_status: int, error_text: str) -> Exception:
        """Map HTTP status and error body to specific ProviderError.

        Args:
            response_status: HTTP status code
            error_text: Raw error response body

        Returns:
            Appropriate ProviderError subclass instance
        """
        try:
            error_data = json.loads(error_text)
            message = error_data.get("error", {}).get("message", error_text)
        except Exception:
            message = error_text

        if response_status == 401:
            return AuthenticationError(message, raw_error=error_text)
        if response_status == 429:
            # Trigger global rate limit block
            from .rate_limit import GlobalRateLimiter

            GlobalRateLimiter.get_instance().set_blocked(60)  # Default 60s cooldown
            return RateLimitError(message, raw_error=error_text)
        if response_status in (400, 422):
            return InvalidRequestError(message, raw_error=error_text)
        if response_status >= 500:
            if "overloaded" in message.lower() or "capacity" in message.lower():
                return OverloadedError(message, raw_error=error_text)
            return APIError(message, status_code=response_status, raw_error=error_text)

        return APIError(message, status_code=response_status, raw_error=error_text)


class ResponseConverterMixin:
    """Mixin for converting OpenAI responses to Anthropic format.

    Handles content extraction, reasoning/thinking blocks, tool calls,
    and response structure transformation.
    """

    def convert_response(self, response_json: dict, original_request: Any) -> dict:
        """Convert OpenAI response to Anthropic format.

        Args:
            response_json: OpenAI-format response JSON
            original_request: Original Anthropic-format request

        Returns:
            Anthropic-format response dictionary
        """
        import uuid

        choice = response_json["choices"][0]
        message = choice["message"]
        content = []

        # Extract reasoning from various sources
        reasoning = message.get("reasoning_content")
        if not reasoning:
            reasoning_details = message.get("reasoning_details")
            if reasoning_details and isinstance(reasoning_details, list):
                reasoning = "\n".join(
                    item.get("text", "")
                    for item in reasoning_details
                    if isinstance(item, dict)
                )

        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})

        # Extract text content (with think tag handling)
        if message.get("content"):
            raw_content = message["content"]
            if isinstance(raw_content, str):
                if not reasoning:
                    think_content, raw_content = extract_think_content(raw_content)
                    if think_content:
                        content.append(
                            {"type": "thinking", "thinking": think_content}
                        )
                if raw_content:
                    content.append({"type": "text", "text": raw_content})
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content.append(item)

        # Extract tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = tc["function"].get("arguments", {})
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    }
                )

        if not content:
            # NIM models (especially Mistral-based) often require non-empty content.
            # Adding a single space satisfies this requirement while avoiding
            # the "(no content)" display issue in Claude Code.
            content.append({"type": "text", "text": " "})

        usage = response_json.get("usage", {})

        return {
            "id": response_json.get("id", f"msg_{uuid.uuid4()}"),
            "type": "message",
            "role": "assistant",
            "model": original_request.model,
            "content": content,
            "stop_reason": map_stop_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }


class StreamProcessorMixin:
    """Mixin for processing streaming responses from NIM API.

    Handles SSE parsing, content block management, tool call processing,
    and error handling during streaming.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_sse_event(self, event_data: str) -> Any:
        """Parse a single SSE event, return None if invalid/done.

        Args:
            event_data: Raw SSE event data

        Returns:
            Parsed JSON data or None for [DONE] or invalid events
        """
        if not event_data.strip():
            return None

        for line in event_data.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_content = line[5:].lstrip()
                if data_content == "[DONE]":
                    return None
                try:
                    return json.loads(data_content)
                except json.JSONDecodeError:
                    logger.debug(f"JSON decode failed for SSE data: {data_content}")
                    return None
        return None

    def _process_tool_call(self, tc: dict, sse: Any):
        """Process a single tool call delta and yield SSE events.

        Args:
            tc: Tool call delta from OpenAI stream
            sse: SSEBuilder instance for generating events

        Yields:
            SSE event strings
        """
        import uuid

        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_indices)

        # Update accumulated name if present
        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )

        # Check if we should start the tool block
        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            # Only start if name is non-empty or we have an ID (start of tool call)
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            # Block index exists (due to ID in previous chunk) but not started due to empty name
            tool_id = f"tool_{uuid.uuid4()}"  # Should ideally reuse ID if we saved it
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            # Ensure block is started before emitting args (with a fallback name if still empty)
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            yield sse.emit_tool_delta(tc_index, args)
