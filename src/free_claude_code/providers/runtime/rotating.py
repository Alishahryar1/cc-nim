"""Provider wrapper that rotates requests across multiple credentials."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.credential_rotation import CredentialRotationState
from free_claude_code.providers.http import maybe_await_aclose


class RotatingProvider(BaseProvider):
    """Fan requests out to one sub-provider per configured credential.

    Failover only happens before the first SSE chunk of a request: once output
    has started streaming to the client, switching credentials would duplicate
    or corrupt the response, so mid-stream errors propagate unchanged.
    """

    def __init__(
        self,
        config: ProviderConfig,
        providers: Sequence[BaseProvider],
        state: CredentialRotationState,
    ) -> None:
        super().__init__(config)
        if not providers:
            raise ValueError("RotatingProvider requires at least one sub-provider")
        self._providers = tuple(providers)
        self._state = state

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate the request once; all sub-providers share the same policy."""
        self._providers[0].preflight_stream(request, reasoning=reasoning)

    async def list_model_ids(self) -> frozenset[str]:
        return await self._providers[0].list_model_ids()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return await self._providers[0].list_model_infos()

    async def cleanup(self) -> None:
        errors: list[Exception] = []
        for provider in self._providers:
            try:
                await provider.cleanup()
            except Exception as exc:  # noqa: BLE001 - aggregate cleanup failures
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("One or more sub-provider cleanups failed", errors)

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        return self._stream_with_rotation(
            request,
            input_tokens,
            request_id=request_id,
            reasoning=reasoning,
        )

    async def _stream_with_rotation(
        self,
        request: MessagesRequest,
        input_tokens: int,
        *,
        request_id: str | None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        attempted: set[int] = set()
        last_error: Exception | None = None

        while len(attempted) < len(self._providers):
            index = await self._state.acquire()
            if index < 0:
                # Every credential is benched (cooldown/circuit-open/lockout).
                wait = await self._state.shortest_cooldown_remaining()
                raise ApplicationUnavailableError(
                    "All API keys for this provider are in cooldown. "
                    f"Retry in {max(1, int(wait))}s."
                )
            if index in attempted:
                remaining = [
                    i for i in range(len(self._providers)) if i not in attempted
                ]
                if not remaining:
                    break
                index = remaining[0]
            attempted.add(index)

            iterator = self._providers[index].stream_response(
                request,
                input_tokens,
                request_id=request_id,
                reasoning=reasoning,
            )
            try:
                first_chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except Exception as error:
                last_error = error
                await maybe_await_aclose(iterator)
                rotate = await self._state.report_failure(index, error)
                if not rotate:
                    raise
                continue

            try:
                yield first_chunk
                async for chunk in iterator:
                    yield chunk
            finally:
                await maybe_await_aclose(iterator)
            await self._state.report_success(index)
            return

        if last_error is not None:
            raise last_error

    def key_health(self) -> list[dict[str, Any]]:
        """Per-credential health snapshots (index-aligned with api_keys)."""
        return self._state.get_metrics()
