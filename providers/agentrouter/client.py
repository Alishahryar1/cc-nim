"""AgentRouter provider implementation (native Anthropic-compatible Messages).

AgentRouter proxies Anthropic-compatible models (Claude, DeepSeek, GLM, etc.)
and expects Claude CLI-style headers for authentication and routing.  The SSE
stream may contain spurious ``data: null`` lines which are filtered out before
forwarding to downstream consumers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import AGENTROUTER_DEFAULT_BASE

# ---------------------------------------------------------------------------
# Claude CLI fingerprint headers (mirrored from the OpenCode JS plugin)
# ---------------------------------------------------------------------------

_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_BETA = "claude-code-20250219,oauth-2025-04-20"
_USER_AGENT = "claude-cli/1.0.108 (external, cli)"

_STAINLESS_HEADERS: dict[str, str] = {
    "x-app": "cli",
    "x-stainless-lang": "js",
    "x-stainless-package-version": "0.55.1",
    "x-stainless-os": "Windows",
    "x-stainless-arch": "x64",
    "x-stainless-runtime": "node",
    "x-stainless-runtime-version": "v22.0.0",
}

# Sentinel values emitted by AgentRouter that must be stripped from the SSE
# stream to avoid confusing downstream parsers.
_NULL_DATA_LINES = frozenset({"data: null", "data:null"})

# Default thinking budget when the client sends ``thinking.type=enabled``
# without a ``budget_tokens`` value.  AgentRouter requires this field.
_DEFAULT_THINKING_BUDGET_TOKENS = 10000


class AgentRouterProvider(AnthropicMessagesTransport):
    """AgentRouter using ``https://agentrouter.org/v1/messages``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="AGENTROUTER",
            default_base_url=AGENTROUTER_DEFAULT_BASE,
        )

    # ------------------------------------------------------------------
    # Request body
    # ------------------------------------------------------------------

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build native body; ensure ``budget_tokens`` is always present.

        AgentRouter rejects ``thinking.type=enabled`` without a
        ``budget_tokens`` value, so we inject a default when the client
        (Claude Code CLI) omits it.
        """
        body = super()._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking = body.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            thinking.setdefault("budget_tokens", _DEFAULT_THINKING_BUDGET_TOKENS)
        return body

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _request_headers(self) -> dict[str, str]:
        """Return Claude CLI-compatible headers required by AgentRouter."""
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _ANTHROPIC_BETA,
            "anthropic-dangerous-direct-browser-access": "true",
            **_STAINLESS_HEADERS,
        }

    def _model_list_headers(self) -> dict[str, str]:
        """Return auth headers for the ``/models`` endpoint."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _ANTHROPIC_BETA,
            "anthropic-dangerous-direct-browser-access": "true",
            **_STAINLESS_HEADERS,
        }

    # ------------------------------------------------------------------
    # SSE filtering - strip ``data: null`` noise from the event stream
    # ------------------------------------------------------------------

    @staticmethod
    def _is_null_data_line(line: str) -> bool:
        """Return whether *line* is a spurious ``data: null`` sentinel."""
        return line.strip() in _NULL_DATA_LINES

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[str]:
        """Group SSE lines into events, filtering ``data: null`` sentinels."""
        event_lines: list[str] = []
        async for line in response.aiter_lines():
            if self._is_null_data_line(line):
                continue
            if line:
                event_lines.append(line)
                continue
            if event_lines:
                yield "\n".join(event_lines) + "\n\n"
                event_lines.clear()
        if event_lines:
            yield "\n".join(event_lines) + "\n\n"
