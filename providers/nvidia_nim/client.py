"""NVIDIA NIM provider implementation."""

import json
import threading
from typing import Any

import httpx
import openai
from loguru import logger

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import (
    body_without_nim_tool_argument_aliases,
    build_request_body,
    clone_body_without_all_thinking,
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
    nim_tool_argument_aliases_from_body,
)


_RETRYABLE_EXC = (
    openai.RateLimitError,
    openai.APITimeoutError,
    httpx.ReadTimeout,
)


class NvidiaNimProvider(OpenAIChatTransport):
    """NVIDIA NIM provider using official OpenAI client.

    Supports round-robin across multiple API keys (comma-separated in config.api_key).
    On 429 or read timeout, remaining keys are tried before giving up.
    """

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings
        # Round-robin key rotation
        raw_key = config.api_key or ""
        self._api_keys: list[str] = [k.strip() for k in raw_key.split(",") if k.strip()]
        if not self._api_keys:
            self._api_keys = [raw_key]
        self._key_index = 0
        self._key_lock = threading.Lock()

    def _next_api_key(self) -> str:
        """Return the next API key in round-robin rotation."""
        with self._key_lock:
            key = self._api_keys[self._key_index % len(self._api_keys)]
            self._key_index += 1
            return key

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
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
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if "reasoning_budget" in error_text:
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_budget after 400 error"
            )
            return retry_body

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

        # Last resort: strip all thinking/reasoning fields when error hints at them
        if any(kw in error_text for kw in ("thinking", "reasoning", "budget")):
            retry_body = clone_body_without_all_thinking(body)
            if retry_body is not None:
                logger.warning(
                    "NIM_STREAM: retrying without all thinking fields after 400 error"
                )
                return retry_body

        return None

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        """Create a streaming completion with round-robin key rotation on retryable errors."""
        try:
            create_body = self._prepare_create_body(body)
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **create_body, stream=True
            )
            return stream, body
        except _RETRYABLE_EXC as error:
            # Try next key
            if len(self._api_keys) > 1:
                next_key = self._next_api_key()
                logger.warning(
                    "NIM_STREAM: retryable error ({}), rotating to next API key",
                    type(error).__name__,
                )
                self._client.api_key = next_key
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is None:
                raise
            create_retry_body = self._prepare_create_body(retry_body)
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **create_retry_body, stream=True
            )
            return stream, retry_body
        except Exception as error:
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is None:
                raise
            create_retry_body = self._prepare_create_body(retry_body)
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **create_retry_body, stream=True
            )
            return stream, retry_body
