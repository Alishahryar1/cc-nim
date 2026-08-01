"""Provider execution shared by inbound API adapters."""

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    Message,
    MessagesRequest,
    SystemContent,
    Tool,
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.failures import (
    failure_permits_failover,
    find_execution_failure,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.trace import (
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .ports import ProviderResolver
from .routing import ResolvedModel, RoutedMessagesRequest

TokenCounter = Callable[
    [list[Message], str | list[SystemContent] | None, list[Tool] | None],
    int,
]
WireApi = Literal["messages", "responses"]


@dataclass(slots=True)
class _Candidate:
    """One provider route that may serve a request, with its rewritten body."""

    resolved: ResolvedModel
    source: MessagesRequest
    rewrite_model: bool
    _request: MessagesRequest | None = field(default=None, init=False)

    @property
    def request(self) -> MessagesRequest:
        """Return this route's body, rewriting the model only when it is used.

        Routing already rewrote the model for the primary. Every other
        candidate needs its own deep copy, which is built on first use so a
        healthy primary never pays for a configured backup.
        """
        if self._request is None:
            self._request = (
                self.source.model_copy(
                    update={"model": self.resolved.provider_model},
                    deep=True,
                )
                if self.rewrite_model
                else self.source
            )
        return self._request


class ProviderExecutor:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        token_counter: TokenCounter = get_token_count,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
    ) -> None:
        self._provider_resolver = provider_resolver
        self._token_counter = token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Preflight synchronously, then return the traced provider stream."""
        gateway_model = routed.resolved.original_model
        # D11: preflight stays synchronous so its failures keep today's clean
        # HTTP-error contract; an eligible failure swaps to the backup here,
        # before any SSE has started.
        served = self._preflight_candidates(
            _candidates(routed),
            reasoning=routed.reasoning,
            gateway_model=gateway_model,
            request_id=request_id,
        )
        serving = served[0]

        route_trace: dict[str, object] = {
            "stage": "routing",
            "event": "free_claude_code.api.route.resolved",
            "source": "api",
            "request_id": request_id,
            "provider_id": serving.resolved.provider_id,
            "provider_model": serving.resolved.provider_model,
            "provider_model_ref": serving.resolved.provider_model_ref,
            "gateway_model": gateway_model,
            "reasoning_control": routed.reasoning.control.value,
            "reasoning_effort": (
                routed.reasoning.effort.value
                if routed.reasoning.effort is not None
                else None
            ),
            "reasoning_budget_tokens": routed.reasoning.budget_tokens,
        }
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        request_snapshot = anthropic_request_snapshot(serving.request)
        request_snapshot["model"] = gateway_model
        trace_event(
            stage="ingress",
            event=(
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=request_snapshot,
            request_id=request_id,
        )

        if self._log_raw_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        async def provider_body() -> AsyncIterator[str]:
            for position, candidate in enumerate(served):
                provider = self._provider_resolver(candidate.resolved.provider_id)
                provider_stream: AsyncIterator[str] | None = None
                # D2: only a request that has emitted nothing may be moved to
                # another provider, so track output rather than exception type.
                yielded = False
                failover_error: Exception | None = None
                try:
                    if position:
                        # Reached by stream-time failover, so this candidate has
                        # not been preflighted yet.
                        provider.preflight_stream(
                            candidate.request,
                            reasoning=routed.reasoning,
                        )
                    provider_stream = provider.stream_response(
                        candidate.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        response_model=gateway_model,
                        reasoning=routed.reasoning,
                    )
                    async for chunk in provider_stream:
                        yielded = True
                        yield chunk
                    return
                except asyncio.CancelledError, GeneratorExit:
                    raise
                except Exception as error:
                    if (
                        yielded
                        or position == len(served) - 1
                        or not failure_permits_failover(error)
                    ):
                        raise
                    failover_error = error
                    _trace_failover(
                        phase="stream",
                        failed=candidate,
                        successor=served[position + 1],
                        gateway_model=gateway_model,
                        request_id=request_id,
                        error=error,
                    )
                finally:
                    if provider_stream is not None:
                        await close_stream_input(
                            provider_stream,
                            owner="provider_executor",
                            source="api",
                            preserved_error=sys.exception() or failover_error,
                        )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "provider_id": serving.resolved.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            provider_body(),
            stage="egress",
            source="api",
            complete_event=(
                "free_claude_code.api.responses.stream_completed"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_completed"
            ),
            interrupted_event=(
                "free_claude_code.api.responses.stream_interrupted"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra=stream_trace,
        )

    def _preflight_candidates(
        self,
        candidates: list[_Candidate],
        *,
        reasoning: ReasoningPolicy,
        gateway_model: str,
        request_id: str,
    ) -> list[_Candidate]:
        """Return the candidates starting at the first one that preflights."""
        index = 0
        while True:
            candidate = candidates[index]
            try:
                provider = self._provider_resolver(candidate.resolved.provider_id)
                provider.preflight_stream(candidate.request, reasoning=reasoning)
            except Exception as error:
                if index + 1 >= len(candidates) or not failure_permits_failover(error):
                    raise
                _trace_failover(
                    phase="preflight",
                    failed=candidate,
                    successor=candidates[index + 1],
                    gateway_model=gateway_model,
                    request_id=request_id,
                    error=error,
                )
                index += 1
                continue
            return candidates[index:]


def _candidates(routed: RoutedMessagesRequest) -> list[_Candidate]:
    """Return the ordered routes allowed to serve one request."""
    return [
        _Candidate(routed.resolved, routed.request, rewrite_model=False),
        *(
            _Candidate(backup, routed.request, rewrite_model=True)
            for backup in routed.backups
        ),
    ]


def _trace_failover(
    *,
    phase: Literal["preflight", "stream"],
    failed: _Candidate,
    successor: _Candidate,
    gateway_model: str,
    request_id: str,
    error: Exception,
) -> None:
    logger.warning(
        "FAILOVER ({}): '{}' failed before any output for '{}'; "
        "serving from backup provider '{}' [{}]",
        phase,
        failed.resolved.provider_id,
        gateway_model,
        successor.resolved.provider_id,
        request_id,
    )
    trace_event(
        stage="routing",
        event="free_claude_code.api.failover",
        source="api",
        phase=phase,
        from_provider=failed.resolved.provider_id,
        to_provider=successor.resolved.provider_id,
        gateway_model=gateway_model,
        exc_type=type(error).__name__,
        failure_kind=_failure_kind(error),
        request_id=request_id,
    )


def _failure_kind(error: Exception) -> str | None:
    failure = find_execution_failure(error)
    return failure.kind.value if failure is not None else None
