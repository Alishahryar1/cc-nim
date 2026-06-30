"""Anthropic token-count API product flow."""

from __future__ import annotations

import uuid

from fastapi import HTTPException
from loguru import logger

from api.model_router import ModelRouter
from api.models.anthropic import TokenCountRequest
from api.models.responses import TokenCountResponse
from api.provider_execution import TokenCounter
from api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.trace import api_messages_request_snapshot, trace_event
from providers.exceptions import ProviderError


class TokenCountHandler:
    """Handle Anthropic-compatible token count requests."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ) -> None:
        self._settings = settings
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def count(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                trace_event(
                    stage="routing",
                    event="api.route.resolved",
                    source="api",
                    kind="count_tokens",
                    provider_id=routed.resolved.provider_id,
                    provider_model=routed.resolved.provider_model,
                    provider_model_ref=routed.resolved.provider_model_ref,
                    gateway_model=routed.request.model,
                )
                trace_event(
                    stage="ingress",
                    event="api.count_tokens.completed",
                    source="api",
                    message_count=len(routed.request.messages),
                    input_tokens=tokens,
                    snapshot=api_messages_request_snapshot(routed.request),
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as exc:
                log_unexpected_api_exception(
                    self._settings,
                    exc,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=http_status_for_unexpected_api_exception(exc),
                    detail=get_user_facing_error_message(exc),
                ) from exc
