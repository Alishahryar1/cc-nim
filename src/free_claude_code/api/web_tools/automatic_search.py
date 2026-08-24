"""One provider decision for Claude Code's automatic WebSearch request."""

import sys
from collections.abc import AsyncIterator, Mapping
from urllib.parse import urlsplit

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import RoutedInferenceRequest
from free_claude_code.core.anthropic import (
    aggregate_inference_events_to_message,
    iter_anthropic_sse,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import InferenceEvent
from free_claude_code.core.trace import close_stream_input, trace_event

from .request import (
    HIDDEN_WEB_SEARCH_NAME,
    AutomaticWebSearchPlan,
    WebSearchDomainFilter,
)
from .streaming import stream_selected_web_search_response


async def stream_automatic_web_search_response(
    provider_executor: ProviderExecutor,
    routed: RoutedInferenceRequest,
    plan: AutomaticWebSearchPlan,
    *,
    request_id: str,
    fallback_input_tokens: int,
    verbose_client_errors: bool,
    log_raw_events: bool,
) -> AsyncIterator[str]:
    """Buffer one model decision, then replay it or execute one local search."""
    trace_event(
        stage="routing",
        event="free_claude_code.api.web_search.automatic_recognized",
        source="api",
        request_id=request_id,
        model=routed.resolved.original_model,
    )
    provider_stream = provider_executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload=plan.request.model_dump(),
        request_id=request_id,
    )
    events: list[InferenceEvent] = []
    try:
        events.extend([event async for event in provider_stream])
    finally:
        await close_stream_input(
            provider_stream,
            owner="automatic_web_search",
            source="api",
            preserved_error=sys.exception(),
        )

    message = await aggregate_inference_events_to_message(_iterate_events(events))

    tool_calls = _tool_calls(message)
    if not tool_calls:
        trace_event(
            stage="execution",
            event="free_claude_code.api.web_search.automatic_declined",
            source="api",
            request_id=request_id,
            model=routed.resolved.original_model,
        )
        async for chunk in iter_anthropic_sse(
            _iterate_events(events),
            log_raw_events=log_raw_events,
        ):
            yield chunk
        return
    if len(tool_calls) != 1:
        raise _protocol_failure(
            "Upstream model returned multiple tool calls for automatic WebSearch.",
            request_id=request_id,
        )

    call = tool_calls[0]
    if call.get("name") != HIDDEN_WEB_SEARCH_NAME:
        raise _protocol_failure(
            "Upstream model returned an unexpected tool for automatic WebSearch.",
            request_id=request_id,
        )
    arguments = call.get("input")
    if not isinstance(arguments, dict) or set(arguments) != {"query"}:
        raise _protocol_failure(
            "Upstream model returned malformed arguments for automatic WebSearch.",
            request_id=request_id,
        )
    raw_query = arguments.get("query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise _protocol_failure(
            "Upstream model returned an empty query for automatic WebSearch.",
            request_id=request_id,
        )
    query = raw_query.strip()
    provider_usage = _provider_usage(message)
    input_tokens = _integer_field(provider_usage, "input_tokens")
    if input_tokens is None:
        input_tokens = fallback_input_tokens

    trace_event(
        stage="execution",
        event="free_claude_code.api.web_search.automatic_selected",
        source="api",
        request_id=request_id,
        model=routed.resolved.original_model,
    )
    async for frame in stream_selected_web_search_response(
        query,
        input_tokens=input_tokens,
        response_model=routed.resolved.original_model,
        provider_usage=provider_usage,
        result_filter=lambda results: _filter_results(results, plan.domains),
        verbose_client_errors=verbose_client_errors,
    ):
        yield frame
    trace_event(
        stage="execution",
        event="free_claude_code.api.web_search.automatic_completed",
        source="api",
        request_id=request_id,
        model=routed.resolved.original_model,
    )


async def _iterate_events(
    events: list[InferenceEvent],
) -> AsyncIterator[InferenceEvent]:
    for event in events:
        yield event


def _tool_calls(message: Mapping[str, object]) -> list[dict[str, object]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[dict[str, object]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append({str(key): value for key, value in block.items()})
    return calls


def _provider_usage(message: Mapping[str, object]) -> dict[str, object]:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {str(key): value for key, value in usage.items()}


def _integer_field(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _filter_results(
    results: list[dict[str, str]],
    domains: WebSearchDomainFilter,
) -> list[dict[str, str]]:
    if not domains.allowed and not domains.blocked:
        return results
    return [
        result
        for result in results
        if _hostname_is_allowed(result.get("url", ""), domains)
    ]


def _hostname_is_allowed(url: str, domains: WebSearchDomainFilter) -> bool:
    hostname = urlsplit(url).hostname
    if hostname is None:
        return False
    try:
        normalized = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return False
    if domains.allowed and not any(
        _is_hostname_or_subdomain(normalized, domain) for domain in domains.allowed
    ):
        return False
    return not any(
        _is_hostname_or_subdomain(normalized, domain) for domain in domains.blocked
    )


def _is_hostname_or_subdomain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def _protocol_failure(message: str, *, request_id: str) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=500,
        message=f"{message}\n\nRequest ID: {request_id}",
        retryable=False,
    )
