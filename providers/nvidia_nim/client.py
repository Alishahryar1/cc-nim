"""NVIDIA NIM provider implementation."""

import json
from typing import Any

import openai
from loguru import logger

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport

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


class NvidiaNimProvider(OpenAIChatTransport):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
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


def _is_reasoning_budget_rejection(error_text: str) -> bool:
    """Return whether NIM rejected optional thinking budget control."""
    if "reasoning_budget" in error_text:
        return True
    return "thinking_token_budget" in error_text and "reasoning_config" in error_text
