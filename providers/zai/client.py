"""Z.ai provider implementation (Anthropic-compatible Messages API)."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from config.constants import (
    ZAI_DEFAULT_REASONING_EFFORT,
    ZAI_REASONING_EFFORT_DISABLED_TOKENS,
    ZAI_REASONING_EFFORT_LEVELS,
)
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import ZAI_DEFAULT_BASE

from .mcp_servers import merge_zai_mcp_servers
from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"

# Falsy values that disable the defaulted-on MCP injection env flag.
_MCP_INJECTION_DISABLED_TOKENS = frozenset({"", "0", "false", "no", "off"})


def _mcp_injection_enabled() -> bool:
    """Whether z.ai HTTP MCP servers are auto-injected into each request."""
    raw = os.environ.get("ZAI_INJECT_MCP_SERVERS", "true").strip().lower()
    return raw not in _MCP_INJECTION_DISABLED_TOKENS


def _resolve_reasoning_effort() -> str | None:
    """Resolve the z.ai ``reasoning_effort`` level from config/env.

    GLM-5.x reasoning depth is selected by a discrete top-level
    ``reasoning_effort`` field (``high`` | ``max``). Defaults to ``max``
    (z.ai's recommendation for coding). Returns ``None`` when disabled or when
    the configured value is not a recognised level.
    """
    raw = os.environ.get("ZAI_REASONING_EFFORT", ZAI_DEFAULT_REASONING_EFFORT)
    value = raw.strip().lower()
    if value in ZAI_REASONING_EFFORT_DISABLED_TOKENS:
        return None
    if value not in ZAI_REASONING_EFFORT_LEVELS:
        logger.warning(
            "ZAI_REASONING_EFFORT={!r} is not a recognised level {}; ignoring.",
            raw,
            sorted(ZAI_REASONING_EFFORT_LEVELS),
        )
        return None
    return value


class ZaiProvider(AnthropicMessagesTransport):
    """Z.ai using Anthropic-compatible Messages at api.z.ai/api/anthropic/v1."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI",
            default_base_url=ZAI_DEFAULT_BASE,
        )
        self._inject_mcp_servers = _mcp_injection_enabled()
        self._reasoning_effort = _resolve_reasoning_effort()

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        thinking_on = self._is_thinking_enabled(request, thinking_enabled)
        body = build_request_body(request, thinking_enabled=thinking_on)
        # z.ai selects GLM-5.x reasoning depth via a discrete ``reasoning_effort``
        # field (separate from Anthropic ``thinking.budget_tokens``); only send it
        # when thinking is enabled, since effort is meaningless otherwise.
        if thinking_on and self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        if self._inject_mcp_servers and self._api_key:
            merge_zai_mcp_servers(body, self._api_key)
        return body

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
