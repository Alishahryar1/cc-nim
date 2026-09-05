"""Codex subscription connection and SSE decoding for Responses generation."""

import uuid
from contextlib import AsyncExitStack

import httpx

from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.http import ProviderAttemptScope
from free_claude_code.providers.openai_responses.events import ResponsesEventSource
from free_claude_code.providers.openai_responses.execution import AuthenticationRecovery
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError

from .auth import OpenAIAccess, OpenAIAuthManager, OpenAIReconnectRequired


class CodexResponsesBackend:
    """Borrow the account and HTTP client for one logical Codex request."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        auth: OpenAIAuthManager,
        body: JsonObject,
        client_headers: dict[str, str],
    ) -> None:
        self._client = client
        self._auth = auth
        self._body = body
        self._client_headers = client_headers
        self._session_id = str(uuid.uuid4())
        self._access: OpenAIAccess | None = None

    async def prepare_attempt(self) -> None:
        self._access = await self._auth.access()

    async def open_attempt(self, scope: ProviderAttemptScope) -> ResponsesEventSource:
        if self._access is None:
            raise RuntimeError("Codex credentials have not been prepared")
        resources = scope.retain(AsyncExitStack())
        response = await self._client.send(
            self._client.build_request(
                "POST",
                "responses",
                json=self._body,
                headers={
                    **self._client_headers,
                    **auth_headers(self._access),
                    "Accept": "text/event-stream",
                    "session_id": self._session_id,
                },
            ),
            stream=True,
        )
        resources.push_async_callback(response.aclose)
        if not response.is_success:
            raise await response_status_error(response)
        content_type = response.headers.get("content-type")
        if content_type and "text/event-stream" not in content_type.lower():
            body, truncated = await _read_bounded_body(response)
            error = TruncatedProviderStreamError(
                "OpenAI returned a non-streaming Responses payload."
            )
            attach_upstream_error_body(error, body, truncated=truncated)
            raise error
        source = ResponsesEventSource(response)
        resources.push_async_callback(source.aclose)
        return source

    def authentication_recovery(
        self, error: Exception
    ) -> AuthenticationRecovery | None:
        if (
            isinstance(error, httpx.HTTPStatusError)
            and error.response.status_code == 401
        ):
            return self._refresh
        return None

    async def _refresh(self) -> None:
        if self._access is None:
            raise RuntimeError("Codex credentials have not been prepared")
        await self._auth.recover_unauthorized(self._access.access_token)

    def normalize_error(self, error: Exception) -> Exception:
        if isinstance(error, OpenAIReconnectRequired):
            return ExecutionFailure(FailureKind.AUTHENTICATION, 401, str(error), False)
        return error

    async def aclose(self) -> None:
        # The generation and account managers own the borrowed resources.
        pass


async def _read_bounded_body(
    response: httpx.Response,
) -> tuple[bytes, bool]:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit + 1 - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) > limit:
            break
    truncated = len(body) > limit
    return bytes(body[:limit]), truncated


async def response_status_error(response: httpx.Response) -> httpx.HTTPStatusError:
    """Return one bounded, diagnostic-preserving HTTP status failure."""
    body, truncated = await _read_bounded_body(response)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        attach_upstream_error_body(error, body, truncated=truncated)
        return error
    raise RuntimeError("response status is successful")


def auth_headers(access: OpenAIAccess) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access.access_token}",
        "ChatGPT-Account-ID": access.account_id,
    }
    if access.fedramp:
        headers["X-OpenAI-Fedramp"] = "true"
    return headers
