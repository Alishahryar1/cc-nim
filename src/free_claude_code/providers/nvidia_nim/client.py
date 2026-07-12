"""NVIDIA NIM provider implementation."""

import json
from collections.abc import Mapping
from typing import Any

import openai
from loguru import logger

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    overloaded_provider_failure,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .request_options import build_nim_request_body
from .retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
)
from .tool_schema import (
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)

_DEGRADED_FUNCTION_STATE = "degraded function cannot be invoked"
_PROFILE = OpenAIChatProfile(OpenAIChatRequestPolicy(provider_name="NIM"))


class NvidiaNimProvider(OpenAIChatProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        nim_settings: NimSettings,
        rate_limiter: ProviderRateLimiter,
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
        )
        self._nim_settings = nim_settings

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_nim_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        bad_request_like = isinstance(error, openai.BadRequestError) or (
            status_code == 400
        )

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if _is_reasoning_budget_rejection(error_text) and (
            bad_request_like or status_code == 500
        ):
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning budget after upstream rejection"
            )
            return retry_body

        if not bad_request_like:
            return None

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            logger.warning("NIM_STREAM: retrying without chat_template after 400 error")
            return retry_body

        if "reasoning_content" in error_text:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_content after 400 error"
            )
            return retry_body

        return None

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Map NVIDIA Cloud Function deployment failure onto canonical overload."""
        if not isinstance(error, openai.BadRequestError):
            return None
        if getattr(error, "status_code", None) != 400:
            return None
        body = getattr(error, "body", None)
        if not isinstance(body, Mapping):
            return None
        detail = body.get("detail")
        if not isinstance(detail, str):
            return None
        function_ref, separator, state = detail.lower().partition(": ")
        function_id = function_ref.removeprefix("function id ").strip(" '\"")
        if (
            not separator
            or not function_ref.startswith("function id ")
            or not function_id
            or state.strip() != _DEGRADED_FUNCTION_STATE
        ):
            return None
        return overloaded_provider_failure()


def _is_reasoning_budget_rejection(error_text: str) -> bool:
    """Return whether NIM rejected optional thinking budget control."""
    if "reasoning_budget" in error_text:
        return True
    return "thinking_token_budget" in error_text and "reasoning_config" in error_text
