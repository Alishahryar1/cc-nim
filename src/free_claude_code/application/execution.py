"""Provider execution shared by inbound API adapters."""

import sys
from collections.abc import AsyncIterator, Callable
from typing import Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    Message,
    SystemContent,
    Tool,
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    find_execution_failure,
)
from free_claude_code.core.trace import (
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .active_model import write_active_model
from .model_fallback import (
    build_fallback_chain,
    eligible_candidate_refs,
    prioritize_large_context,
)
from .ports import ProviderResolver, RequestRuntimeLease, UsageStatsPort
from .routing import ModelRouter, RoutedMessagesRequest

TokenCounter = Callable[
    [list[Message], str | list[SystemContent] | None, list[Tool] | None],
    int,
]
WireApi = Literal["messages", "responses"]


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
        self._model_router: ModelRouter | None = None
        self._model_cache: RequestRuntimeLease | None = None
        self._usage_stats: UsageStatsPort | None = None

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_id: str,
        model_router: ModelRouter | None = None,
        model_cache: RequestRuntimeLease | None = None,
        usage_stats: UsageStatsPort | None = None,
    ) -> AsyncIterator[str]:
        """Preflight synchronously, then return the traced provider stream."""
        # Per-request fallback wiring (the executor is reused across requests).
        if model_router is not None:
            self._model_router = model_router
        if model_cache is not None:
            self._model_cache = model_cache
        if usage_stats is not None:
            self._usage_stats = usage_stats

        # Derivation mode pins no primary model, so there is nothing to preflight
        # up front - every candidate is resolved and preflighted inside the loop.
        provider = None
        if not routed.resolved.derivation:
            provider = self._provider_resolver(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

        route_trace: dict[str, object] = {
            "stage": "routing",
            "event": "free_claude_code.api.route.resolved",
            "source": "api",
            "request_id": request_id,
            "provider_id": routed.resolved.provider_id,
            "provider_model": routed.resolved.provider_model,
            "provider_model_ref": routed.resolved.provider_model_ref,
            "gateway_model": routed.request.model,
            "thinking_enabled": routed.resolved.thinking_enabled,
        }
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        trace_event(
            stage="ingress",
            event=(
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=anthropic_request_snapshot(routed.request),
            request_id=request_id,
        )

        if self._log_raw_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        derivation = routed.resolved.derivation

        async def provider_body() -> AsyncIterator[str]:
            chain = self._fallback_chain(routed, input_tokens)
            if derivation and not chain:
                raise ExecutionFailure(
                    kind=FailureKind.UNAVAILABLE,
                    status_code=503,
                    message=(
                        "No accessible models available for derivation. "
                        "Configure at least one provider API key."
                    ),
                    retryable=True,
                )
            last_error: BaseException | None = None
            for index, candidate_ref in enumerate(chain):
                # In derivation mode there is no pinned primary, so every
                # candidate is resolved here; otherwise index 0 reuses the
                # already-resolved and preflighted default model exactly.
                if index == 0 and not derivation:
                    # `provider` is only None in derivation mode, excluded here.
                    assert provider is not None
                    candidate_request = routed.request
                    candidate_provider_id = routed.resolved.provider_id
                    candidate_provider_model = routed.resolved.provider_model
                    candidate_thinking = routed.resolved.thinking_enabled
                    candidate_provider = provider
                else:
                    # `_model_router` is configured before any request reaches
                    # the fallback loop; it is only None prior to setup.
                    assert self._model_router is not None
                    # Skip models a recent failure put in cooldown BEFORE
                    # resolving and logging them. A derivation chain can carry
                    # ~30 parked candidates (paid, tiny-window, phantom); a cheap
                    # partition of the "provider/model" ref matches what resolve()
                    # would produce, so we drop them without re-resolving and
                    # re-logging a MODEL DIRECT line for each one every request.
                    probe_id, separator, probe_model = candidate_ref.partition("/")
                    if (
                        derivation
                        and separator
                        and self._provider_resolver(probe_id).is_model_in_cooldown(
                            probe_model
                        )
                    ):
                        continue
                    resolved = self._model_router.resolve(candidate_ref)
                    candidate_provider_id = resolved.provider_id
                    candidate_provider_model = resolved.provider_model
                    candidate_thinking = resolved.thinking_enabled
                    candidate_provider = self._provider_resolver(candidate_provider_id)
                    candidate_request = routed.request.model_copy(deep=True)
                    candidate_request.model = candidate_provider_model
                    candidate_provider.preflight_stream(
                        candidate_request, thinking_enabled=candidate_thinking
                    )
                    trace_event(
                        stage="routing",
                        event="free_claude_code.api.route.fallback",
                        source="api",
                        request_id=request_id,
                        provider_id=candidate_provider_id,
                        provider_model=candidate_provider_model,
                        provider_model_ref=candidate_ref,
                        gateway_model=candidate_request.model,
                        thinking_enabled=candidate_thinking,
                        fallback_index=index,
                    )

                committed = False
                provider_stream: AsyncIterator[str] | None = None
                try:
                    provider_stream = candidate_provider.stream_response(
                        candidate_request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=candidate_thinking,
                    )
                    async for chunk in provider_stream:
                        if not committed:
                            committed = True
                            write_active_model(candidate_provider_model)
                        yield chunk
                    if self._usage_stats is not None:
                        self._usage_stats.record_success(
                            candidate_provider_id,
                            candidate_provider_model,
                            input_tokens=input_tokens,
                        )
                    return
                except BaseException as exc:
                    last_error = exc
                    if self._usage_stats is not None and isinstance(exc, Exception):
                        # Excludes CancelledError/GeneratorExit: a cancelled
                        # stream isn't evidence the model itself is unreliable.
                        self._usage_stats.record_error(
                            candidate_provider_id, candidate_provider_model
                        )
                    failure = find_execution_failure(exc)
                    # In derivation there is no user-chosen model to honor, so any
                    # pre-commit failure just means "this model can't serve it" -
                    # try the next candidate (auth, 404, odd 400s, capacity...).
                    # Fixed-model routing only switches on capacity/context errors.
                    non_switchable_failure = failure is not None and not (
                        failure.retryable or failure.model_fallback_eligible
                    )
                    if committed or (non_switchable_failure and not derivation):
                        # Already delivered output, or a non-switchable error on a
                        # user-pinned model -> do not switch models.
                        raise
                    logger.warning(
                        "MODEL FALLBACK: '{}' failed ({}), trying next candidate",
                        candidate_provider_model,
                        type(exc).__name__,
                    )
                    trace_event(
                        stage="provider",
                        event="provider.fallback.skip",
                        source="provider",
                        request_id=request_id,
                        provider_id=candidate_provider_id,
                        provider_model=candidate_provider_model,
                        fallback_index=index,
                        exc_type=type(exc).__name__,
                    )
                    continue
                finally:
                    if provider_stream is not None:
                        await close_stream_input(
                            provider_stream,
                            owner="provider_executor",
                            source="api",
                            preserved_error=sys.exception(),
                        )
            if last_error is not None:
                raise last_error
            if derivation:
                # Every candidate was in cooldown (all recently rate limited).
                raise ExecutionFailure(
                    kind=FailureKind.OVERLOADED,
                    status_code=429,
                    message=(
                        "All accessible models are rate limited right now. "
                        "Please retry shortly."
                    ),
                    retryable=True,
                )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "provider_id": routed.resolved.provider_id,
            "gateway_model": routed.request.model,
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

    def _fallback_chain(
        self, routed: RoutedMessagesRequest, input_tokens: int
    ) -> list[str]:
        """Ordered ``provider/model`` refs to try for this request's model.

        In derivation mode the role pins no model, so the chain is every
        accessible candidate strongest-first. Otherwise the resolved model
        leads, followed by the discovered candidates when the cache is
        available; without a cache, just the single resolved model. A giant
        request (compaction) front-loads the large-window models.
        """

        derivation = routed.resolved.derivation
        if self._model_router is None or self._model_cache is None:
            return [] if derivation else [routed.resolved.provider_model_ref]
        try:
            model_infos = self._model_cache.cached_prefixed_model_infos()
        except Exception as exc:
            logger.warning("MODEL FALLBACK: model cache unavailable: {}", exc)
            return [] if derivation else [routed.resolved.provider_model_ref]
        if derivation:
            return prioritize_large_context(
                eligible_candidate_refs(model_infos), input_tokens
            )
        chain = build_fallback_chain(routed.resolved.provider_model_ref, model_infos)
        if not chain:
            return [routed.resolved.provider_model_ref]
        # Keep the pinned primary first; reorder/filter only the fallback tail
        # so giant requests still prefer large-window models after a failure.
        primary, *rest = chain
        return [primary, *prioritize_large_context(rest, input_tokens)]
