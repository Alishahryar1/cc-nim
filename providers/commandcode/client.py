"""Command Code provider implementation (/alpha/generate natively)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.model_listing import (
    ProviderModelInfo,
    extract_openai_model_ids,
    model_infos_from_ids,
)
from providers.rate_limit import GlobalRateLimiter
from providers.transports.anthropic_messages.http import (
    maybe_await_aclose,
    model_list_json,
    raise_for_status_with_body,
)

from .request import build_request_body
from .stream import CommandCodeStreamRunner

_COMMAND_CODE_VERSION = "v1.0.8"


class CommandCodeProvider(BaseProvider):
    """Command Code natively communicating with /alpha/generate."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._provider_name = "COMMANDCODE"
        self._api_key = config.api_key
        # Use the base domain as the client base url
        self._base_url = (config.base_url or "https://api.commandcode.ai").rstrip("/")

        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            "commandcode",
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from the provider."""
        return frozenset(info.model_id for info in await self.list_model_infos())

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model ids plus optional metadata from the models endpoint."""
        response = await self._client.get(
            "/provider/v1/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            payload = model_list_json(response, provider_name=self._provider_name)
            ids = extract_openai_model_ids(payload, provider_name=self._provider_name)
            return model_infos_from_ids(ids)
        finally:
            await maybe_await_aclose(response)

    def _get_error_message(self, error: Exception, request_id: str | None) -> str:
        """Map an exception into a user-facing provider error message."""
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        return user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via native /alpha/generate endpoint."""

        body = build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "x-command-code-version": _COMMAND_CODE_VERSION,
            "x-cli-environment": "production",
            "Accept": "text/event-stream",
        }

        req = self._client.build_request(
            "POST",
            "https://api.commandcode.ai/alpha/generate",
            json=body,
            headers=headers,
        )

        try:
            response = await self._client.send(req, stream=True)
            if response.status_code != 200:
                await raise_for_status_with_body(
                    response,
                    provider_name=self._provider_name,
                    req_tag="commandcode_generate",
                    log_api_error_tracebacks=self._config.log_api_error_tracebacks,
                )
        except Exception as e:
            # We must yield an error event formatted like Anthropic would if it fails immediately
            yield f'event: error\ndata: {{"type": "error", "error": {{"type": "api_error", "message": "{self._get_error_message(e, request_id)}" }}}}\n\n'
            return

        runner = CommandCodeStreamRunner(
            response.aiter_text(),
            request_id=request_id,
            model=body.get("params", {}).get("model", "unknown"),
        )

        try:
            async for event in runner.run():
                yield event
        finally:
            await maybe_await_aclose(response)
