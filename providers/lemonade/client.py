"""Lemonade provider implementation (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from providers.base import ProviderConfig
from providers.defaults import LEMONADE_DEFAULT_BASE
from core.anthropic.streaming import TruncatedProviderStreamError
from providers.model_listing import extract_openai_model_ids
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="LEMONADE",
    include_extra_body=True,
)

_MODEL_LOAD_DELAY = 3.0
_MAX_RETRIES = 2


class LemonadeProvider(OpenAIChatTransport):
    """Lemonade provider using OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LEMONADE",
            base_url=config.base_url or LEMONADE_DEFAULT_BASE,
            api_key=config.api_key or "lemonade",
        )
        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._http_client.aclose()
        await super().cleanup()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from Lemonade's /v1/models endpoint."""
        response = await self._http_client.get("/models")
        response.raise_for_status()
        return extract_openai_model_ids(
            response.json(), provider_name=self._provider_name
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response with retry for model loading delays."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async for event in super().stream_response(
                    request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=thinking_enabled,
                ):
                    yield event
                return
            except TruncatedProviderStreamError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "LEMONADE: stream truncated (attempt {}/{}), "
                        "model may be loading, retrying in {}s",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        _MODEL_LOAD_DELAY,
                    )
                    await asyncio.sleep(_MODEL_LOAD_DELAY)
                else:
                    raise
