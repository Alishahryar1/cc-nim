"""Telepub Voyage provider (OpenAI-compatible chat completions API)."""

from __future__ import annotations

from typing import Any

from loguru import logger

from providers.base import ProviderConfig
from providers.defaults import TELEPUB_VOYAGE_DEFAULT_BASE
from providers.model_listing import extract_openai_model_ids
from providers.transports.openai_chat import OpenAIChatTransport

from .request import build_request_body


class TelepubVoyageProvider(OpenAIChatTransport):
    """Telepub Voyage upstream using OpenAI ``/chat/completions``."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        static_model_ids: frozenset[str],
    ):
        self._static_model_ids = static_model_ids
        super().__init__(
            config,
            provider_name="TELEPUB_VOYAGE",
            base_url=config.base_url or TELEPUB_VOYAGE_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def list_model_ids(self) -> frozenset[str]:
        """Merge OpenAI ``/models`` results with ``TELEPUB_VOYAGE_MODELS`` static ids."""
        merged: set[str] = set(self._static_model_ids)
        try:
            payload = await self._client.models.list()
            merged.update(
                extract_openai_model_ids(payload, provider_name=self._provider_name)
            )
        except Exception:
            logger.debug(
                "TELEPUB_VOYAGE: models.list failed; using static list only",
            )
        return frozenset(merged)
