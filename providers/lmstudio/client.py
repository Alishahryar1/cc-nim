"""LM Studio provider implementation (OpenAI-compatible chat completions).

Switched from LM Studio's native Anthropic Messages endpoint (2026-07-04):
the newer ``/v1/messages`` path renders Claude Code conversations through the
model's jinja chat template with strict role-alternation rules and a fragile
``[TOOL_CALLS]`` parser — observed leaking control tokens into tool names
(``[TOOL_CALLS]Read``) and dumping whole tool calls into text
(``Read[ARGS]{...}``), which ends agent runs silently. The OpenAI
``/v1/chat/completions`` path is LM Studio's mature parsing route, and fcc's
OpenAI transport layers its own tool-call assembly, think-tag parsing, and
heuristic recovery on top.
"""

import time
from typing import Any

import httpx
from loguru import logger

from core.anthropic import (
    ReasoningReplayMode,
    build_base_request_body,
    get_token_count,
)
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.defaults import LMSTUDIO_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.transports.openai_chat import OpenAIChatTransport


class LMStudioProvider(OpenAIChatTransport):
    """LM Studio via its OpenAI-compatible chat completions endpoint."""

    # LM Studio truncates the stream silently (no terminal event) when the
    # prompt exceeds the loaded context. Refuse clearly over-budget prompts
    # up front with the same "prompt is too long" invalid_request_error the
    # real Anthropic API uses, so Claude Code can compact/retry instead of
    # dying mid-stream.
    _CONTEXT_CACHE_TTL_S = 30.0

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LMSTUDIO",
            base_url=config.base_url or LMSTUDIO_DEFAULT_BASE,
            api_key=config.api_key or "lm-studio",
        )
        self._loaded_context_cache: tuple[float, int | None] = (0.0, None)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build an OpenAI chat body from the Anthropic request.

        Prior-turn thinking is never replayed: Mistral-family templates have
        no assistant reasoning field, and replaying ``<think>`` text inflates
        the local context for no benefit. New-response thinking still streams
        back via ``reasoning_content``/``<think>`` parsing in the transport.
        """
        try:
            return build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        super().preflight_stream(request, thinking_enabled=thinking_enabled)
        self._preflight_context_budget(request)

    def _preflight_context_budget(self, request: Any) -> None:
        loaded_context = self._loaded_context_length()
        if loaded_context is None:
            return
        estimate = get_token_count(
            getattr(request, "messages", []) or [],
            getattr(request, "system", None),
            getattr(request, "tools", None),
        )
        # The estimate is cl100k-based and undercounts local tokenizers
        # (observed ~8% low vs devstral); a request above 90% of the loaded
        # context is already past where client-side compaction should have
        # fired, and letting it through risks a silent LM Studio truncation.
        budget = int(loaded_context * 0.9)
        if estimate > budget:
            raise InvalidRequestError(
                f"prompt is too long: {estimate} tokens > {budget} "
                f"maximum (90% of loaded LM Studio context {loaded_context})"
            )

    def _loaded_context_length(self) -> int | None:
        """Best-effort loaded context length from LM Studio's REST API, cached."""
        cached_at, cached_value = self._loaded_context_cache
        if time.monotonic() - cached_at < self._CONTEXT_CACHE_TTL_S:
            return cached_value

        value: int | None = None
        try:
            root = self._base_url
            root = root[: -len("/v1")] if root.endswith("/v1") else root
            response = httpx.get(f"{root}/api/v0/models", timeout=2.0)
            response.raise_for_status()
            loaded = [
                model.get("loaded_context_length")
                for model in response.json().get("data", [])
                if model.get("state") == "loaded"
                and isinstance(model.get("loaded_context_length"), int)
            ]
            # ponytail: single-model setups in practice; with several loaded
            # models the most generous ceiling still makes a valid backstop.
            value = max(loaded) if loaded else None
        except Exception as error:  # backstop only — never block the request
            logger.debug(
                "LMSTUDIO context preflight unavailable: {}", type(error).__name__
            )
            value = None
        self._loaded_context_cache = (time.monotonic(), value)
        return value
