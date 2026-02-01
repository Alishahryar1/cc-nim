"""Mixins for NVIDIA NIM provider - decoupling responsibilities.

This module contains focused mixins that handle specific aspects of the
NVIDIA NIM provider functionality:
- RequestBuilderMixin: Builds request bodies
- ErrorMapperMixin: Maps HTTP errors to provider exceptions
- ResponseConverterMixin: Converts responses between formats
"""

import json
import logging
from typing import Any, Dict

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
        }

        if request_data.temperature is not None:
            body["temperature"] = request_data.temperature
        if request_data.top_p is not None:
            body["top_p"] = request_data.top_p
        if request_data.stop_sequences:
            body["stop"] = request_data.stop_sequences
        if request_data.tools:
            body["tools"] = AnthropicToOpenAIConverter.convert_tools(request_data.tools)

        # Handle non-standard parameters via extra_body
        extra_params = request_data.extra_body.copy() if request_data.extra_body else {}

        # Handle thinking/reasoning mode
        if request_data.thinking and getattr(request_data.thinking, "enabled", True):
            extra_params.setdefault("thinking", {"type": "enabled"})
            extra_params.setdefault("reasoning_split", True)
            extra_params.setdefault(
                "chat_template_kwargs",
                {"thinking": True, "reasoning_split": True, "clear_thinking": False},
            )

        if extra_params:
            body["extra_body"] = extra_params

        # Apply NIM defaults
        for key, val in self._nim_params.items():
            if key not in body and key not in extra_params:
                body[key] = val

        return body


class ErrorMapperMixin:
    """Mixin for mapping HTTP errors to provider exceptions.

    Converts HTTP status codes and error responses to appropriate
    ProviderError subclasses for standardized error handling.
    """

    def _map_error(self, e: Exception) -> Exception:
        """Map OpenAI exception to specific ProviderError.

        Args:
            e: The OpenAI exception to map

        Returns:
            Appropriate ProviderError subclass instance
        """
        import openai

        if isinstance(e, openai.AuthenticationError):
            return AuthenticationError(str(e), raw_error=str(e))
        if isinstance(e, openai.RateLimitError):
            # Trigger global rate limit block
            from .rate_limit import GlobalRateLimiter

            GlobalRateLimiter.get_instance().set_blocked(60)  # Default 60s cooldown
            return RateLimitError(str(e), raw_error=str(e))
        if isinstance(e, openai.BadRequestError):
            return InvalidRequestError(str(e), raw_error=str(e))
        if isinstance(e, openai.InternalServerError):
            message = str(e)
            if "overloaded" in message.lower() or "capacity" in message.lower():
                return OverloadedError(message, raw_error=str(e))
            return APIError(message, status_code=500, raw_error=str(e))
        if isinstance(e, openai.APIError):
            return APIError(
                str(e), status_code=getattr(e, "status_code", 500), raw_error=str(e)
            )

        return e


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
                        content.append({"type": "thinking", "thinking": think_content})
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
