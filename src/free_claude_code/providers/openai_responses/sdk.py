"""SDK connection mechanics for one Responses generation request."""

from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.endpoint import RequestEndpoint
from free_claude_code.providers.failure_policy import provider_authentication_status
from free_claude_code.providers.http import ProviderAttemptScope

from .events import ResponsesEventAdapter, ResponsesEventSource
from .execution import AuthenticationRecovery


class SDKResponsesBackend:
    """Borrow a provider client and resolve isolated credentials at dispatch."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        body: JsonObject,
        endpoint: RequestEndpoint | None,
        event_adapter_factory: Callable[[], ResponsesEventAdapter] | None,
    ) -> None:
        self._base_client = client
        self._client = client
        self._body = body
        self._endpoint = endpoint
        self._event_adapter_factory = event_adapter_factory

    async def prepare_attempt(self) -> None:
        if self._endpoint is not None:
            self._client = await self._endpoint.openai_client(self._base_client)

    async def open_attempt(self, scope: ProviderAttemptScope) -> ResponsesEventSource:
        adapter = (
            self._event_adapter_factory()
            if self._event_adapter_factory is not None
            else None
        )
        body = self._body
        resources = scope.retain(AsyncExitStack())
        response = await resources.enter_async_context(
            self._client.responses.with_streaming_response.create(
                model=cast(str, body["model"]),
                input=cast(str | ResponseInputParam, body.get("input")),
                stream=True,
                store=False,
                extra_body={
                    key: value
                    for key, value in body.items()
                    if key not in {"model", "input", "stream", "store"}
                }
                or None,
                extra_headers=self._endpoint.openai_headers()
                if self._endpoint is not None
                else None,
            )
        )
        source = ResponsesEventSource(response.http_response, adapter=adapter)
        resources.push_async_callback(source.aclose)
        return source

    def authentication_recovery(
        self, error: Exception
    ) -> AuthenticationRecovery | None:
        if self._endpoint is not None and provider_authentication_status(error) in {
            401,
            403,
        }:
            return self._refresh
        return None

    async def _refresh(self) -> None:
        if self._endpoint is not None:
            self._endpoint.request_refresh()

    def normalize_error(self, error: Exception) -> Exception:
        return error

    async def aclose(self) -> None:
        if self._endpoint is not None:
            await self._endpoint.aclose()
