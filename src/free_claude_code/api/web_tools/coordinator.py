"""Request-local model/tool loop for Anthropic web server tools."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import aiohttp
import httpx

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import RoutedMessagesRequest
from free_claude_code.core.anthropic import (
    Message,
    aggregate_anthropic_sse_to_message,
    anthropic_status_for_error_type,
)
from free_claude_code.core.anthropic.server_tool_sse import (
    WEB_FETCH_TOOL_RESULT,
    WEB_FETCH_TOOL_RESULT_ERROR,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.trace import close_stream_input, trace_event

from . import outbound
from .egress import WebFetchAccessError, WebFetchEgressPolicy, WebFetchEgressViolation
from .outbound import UnsupportedWebContentError
from .request import WebServerToolKind, WebServerToolPlan, WebServerToolSpec
from .streaming import (
    message_end_frames,
    message_start_frame,
    stream_content_blocks,
)

_MAX_MODEL_ROUNDS = 10
_MAX_FETCH_URL_CHARS = 250
_MAX_SEARCH_QUERY_CHARS = 1_024
_PUBLIC_RESULT_TYPE: dict[WebServerToolKind, str] = {
    "web_search": WEB_SEARCH_TOOL_RESULT,
    "web_fetch": WEB_FETCH_TOOL_RESULT,
}
_PUBLIC_ERROR_TYPE: dict[WebServerToolKind, str] = {
    "web_search": WEB_SEARCH_TOOL_RESULT_ERROR,
    "web_fetch": WEB_FETCH_TOOL_RESULT_ERROR,
}


@dataclass(frozen=True, slots=True)
class _SelectedServerCall:
    provider_id: str
    public_id: str
    spec: WebServerToolSpec
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LocalToolResult:
    public_block: dict[str, object]
    internal_content: dict[str, object]
    discovered_urls: frozenset[str]
    error_code: str | None


class WebServerToolCoordinator:
    """Run one bounded provider/local-tool loop at the Messages boundary."""

    def __init__(
        self,
        provider_executor: ProviderExecutor,
        *,
        web_fetch_egress: WebFetchEgressPolicy,
    ) -> None:
        self._provider_executor = provider_executor
        self._web_fetch_egress = web_fetch_egress

    async def stream(
        self,
        routed: RoutedMessagesRequest,
        plan: WebServerToolPlan,
        *,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Yield one public lifecycle while internal model rounds stay hidden."""
        messages = list(plan.request.messages)
        provenance = set(plan.provenance)
        use_counts: dict[WebServerToolKind, int] = {
            "web_search": 0,
            "web_fetch": 0,
        }
        web_request_counts: dict[WebServerToolKind, int] = {
            "web_search": 0,
            "web_fetch": 0,
        }
        aggregate_usage: dict[str, object] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        public_message_id: str | None = None
        public_index = 0
        committed = False

        trace_event(
            stage="routing",
            event="free_claude_code.api.web_server_tools.recognized",
            source="api",
            request_id=request_id,
            model=routed.resolved.original_model,
            tool_kinds=[spec.kind for spec in plan.specs],
            tool_choice=plan.choice,
        )

        if plan.pending:
            public_message_id = f"msg_{uuid.uuid4().hex}"
            yield message_start_frame(
                message_id=public_message_id,
                model=routed.resolved.original_model,
                usage=aggregate_usage,
            )
            committed = True
            trace_event(
                stage="execution",
                event="free_claude_code.api.web_server_tools.resumed",
                source="api",
                request_id=request_id,
                pending_call_count=len(plan.pending),
            )
            internal_results: list[dict[str, object]] = []
            public_results: list[dict[str, object]] = []
            for pending in plan.pending:
                result = await self._execute_local_call(
                    pending.spec,
                    pending.arguments,
                    public_id=pending.public_id,
                    provenance=provenance,
                    use_counts=use_counts,
                    web_request_counts=web_request_counts,
                    request_id=request_id,
                )
                public_results.append(result.public_block)
                internal_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": pending.public_id,
                        "content": result.internal_content,
                    }
                )
                provenance.update(result.discovered_urls)
            messages.append(
                Message.model_validate({"role": "user", "content": internal_results})
            )
            async for frame in stream_content_blocks(
                public_results,
                start_index=public_index,
            ):
                yield frame
            public_index += len(public_results)

        for round_index in range(1, _MAX_MODEL_ROUNDS + 1):
            round_choice = (
                plan.request.tool_choice
                if round_index == 1 and not plan.pending
                else {"type": "auto"}
            )
            round_request = plan.request.model_copy(
                update={"messages": messages, "tool_choice": round_choice},
                deep=True,
            )
            round_routed = replace(routed, request=round_request)
            trace_event(
                stage="execution",
                event="free_claude_code.api.web_server_tools.model_round_started",
                source="api",
                request_id=request_id,
                wire_api="messages",
                model=routed.resolved.original_model,
                round_index=round_index,
            )
            provider_stream = self._provider_executor.stream(
                round_routed,
                wire_api="messages",
                raw_log_label="FULL_PAYLOAD",
                raw_log_payload=round_request.model_dump(),
                request_id=request_id,
                internal_round=round_index,
            )
            buffered: list[str] = []

            try:
                round_message, error = await aggregate_anthropic_sse_to_message(
                    _capture_stream(provider_stream, buffered)
                )
            finally:
                await close_stream_input(
                    provider_stream,
                    owner="web_server_tool_coordinator",
                    source="api",
                    preserved_error=sys.exception(),
                )
            if error is not None:
                raise _execution_failure_from_stream_error(error)
            _validate_round_frames(buffered)

            blocks = _message_blocks(round_message)
            selected = _selected_calls(blocks, plan)
            if plan.choice == "none" and selected:
                raise _protocol_failure(
                    "The provider selected a server tool after tool_choice none."
                )
            if (
                round_index == 1
                and not plan.pending
                and plan.forced_kind is not None
                and any(call.spec.kind != plan.forced_kind for call in selected)
            ):
                raise _protocol_failure(
                    "The provider selected a different tool than the forced "
                    f"{plan.forced_kind} server tool."
                )
            _add_usage(aggregate_usage, round_message.get("usage"))
            stop_reason = _message_stop_reason(round_message)
            trace_event(
                stage="execution",
                event="free_claude_code.api.web_server_tools.model_round_completed",
                source="api",
                request_id=request_id,
                wire_api="messages",
                model=routed.resolved.original_model,
                round_index=round_index,
                stop_reason=stop_reason,
                server_call_count=len(selected),
            )

            if not selected:
                if (
                    plan.choice in {"any", "tool"}
                    and round_index == 1
                    and not plan.pending
                ):
                    raise _protocol_failure(
                        "The provider did not select the required Anthropic server tool."
                    )
                if stop_reason == "tool_use":
                    raise _protocol_failure(
                        "The provider ended with tool_use but returned no supported server call."
                    )
                if not committed:
                    for chunk in buffered:
                        yield chunk
                    _trace_finished(
                        request_id,
                        round_index=round_index,
                        stop_reason=stop_reason,
                        usage=aggregate_usage,
                        web_request_counts=web_request_counts,
                    )
                    return
                public_blocks = _public_non_tool_blocks(blocks)
                async for frame in stream_content_blocks(
                    public_blocks,
                    start_index=public_index,
                ):
                    yield frame
                public_index += len(public_blocks)
                for frame in message_end_frames(
                    usage=_with_server_usage(aggregate_usage, web_request_counts),
                    stop_reason=stop_reason,
                ):
                    yield frame
                _trace_finished(
                    request_id,
                    round_index=round_index,
                    stop_reason=stop_reason,
                    usage=aggregate_usage,
                    web_request_counts=web_request_counts,
                )
                return

            public_blocks, public_call_ids = _public_selection_blocks(blocks, selected)
            if public_message_id is None:
                raw_id = round_message.get("id")
                public_message_id = (
                    raw_id if isinstance(raw_id, str) else f"msg_{uuid.uuid4().hex}"
                )
            if not committed:
                yield message_start_frame(
                    message_id=public_message_id,
                    model=routed.resolved.original_model,
                    usage=aggregate_usage,
                )
                committed = True
            async for frame in stream_content_blocks(
                public_blocks,
                start_index=public_index,
            ):
                yield frame
            public_index += len(public_blocks)

            if round_index == _MAX_MODEL_ROUNDS:
                for frame in message_end_frames(
                    usage=_with_server_usage(aggregate_usage, web_request_counts),
                    stop_reason="pause_turn",
                ):
                    yield frame
                trace_event(
                    stage="execution",
                    event="free_claude_code.api.web_server_tools.paused",
                    source="api",
                    request_id=request_id,
                    pending_call_count=len(selected),
                    round_index=round_index,
                )
                _trace_finished(
                    request_id,
                    round_index=round_index,
                    stop_reason="pause_turn",
                    usage=aggregate_usage,
                    web_request_counts=web_request_counts,
                )
                return

            internal_results: list[dict[str, object]] = []
            public_results: list[dict[str, object]] = []
            for call, public_id in zip(selected, public_call_ids, strict=True):
                result = await self._execute_local_call(
                    call.spec,
                    call.arguments,
                    public_id=public_id,
                    provenance=provenance,
                    use_counts=use_counts,
                    web_request_counts=web_request_counts,
                    request_id=request_id,
                )
                public_results.append(result.public_block)
                internal_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.provider_id,
                        "content": result.internal_content,
                    }
                )
                provenance.update(result.discovered_urls)
            async for frame in stream_content_blocks(
                public_results,
                start_index=public_index,
            ):
                yield frame
            public_index += len(public_results)
            messages.extend(
                (
                    Message.model_validate({"role": "assistant", "content": blocks}),
                    Message.model_validate(
                        {"role": "user", "content": internal_results}
                    ),
                )
            )

        raise AssertionError("bounded server-tool loop ended without a terminal state")

    async def _execute_local_call(
        self,
        spec: WebServerToolSpec,
        arguments: dict[str, object],
        *,
        public_id: str,
        provenance: set[str],
        use_counts: dict[WebServerToolKind, int],
        web_request_counts: dict[WebServerToolKind, int],
        request_id: str,
    ) -> _LocalToolResult:
        use_counts[spec.kind] += 1
        ordinal = use_counts[spec.kind]
        web_request_counts[spec.kind] += 1
        trace_event(
            stage="execution",
            event="free_claude_code.api.web_server_tools.local_started",
            source="api",
            request_id=request_id,
            tool_kind=spec.kind,
            use_ordinal=ordinal,
        )
        started = asyncio.get_running_loop().time()
        try:
            if spec.max_uses is not None and ordinal > spec.max_uses:
                result = _error_result(spec.kind, public_id, "max_uses_exceeded")
            elif spec.kind == "web_search":
                result = await _execute_search(spec, arguments, public_id=public_id)
            else:
                result = await self._execute_fetch(
                    spec,
                    arguments,
                    public_id=public_id,
                    provenance=provenance,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            fetch_url = arguments.get("url")
            outbound._log_web_tool_failure(
                spec.kind,
                exc,
                fetch_url=fetch_url if isinstance(fetch_url, str) else None,
            )
            result = _error_result(
                spec.kind,
                public_id,
                _local_error_code(spec.kind, exc),
            )
        elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1_000)
        trace_event(
            stage="execution",
            event="free_claude_code.api.web_server_tools.local_completed",
            source="api",
            request_id=request_id,
            tool_kind=spec.kind,
            use_ordinal=ordinal,
            error_code=result.error_code,
            duration_ms=elapsed_ms,
            discovered_url_count=len(result.discovered_urls),
        )
        return result

    async def _execute_fetch(
        self,
        spec: WebServerToolSpec,
        arguments: dict[str, object],
        *,
        public_id: str,
        provenance: set[str],
    ) -> _LocalToolResult:
        raw_url = arguments.get("url")
        if set(arguments) != {"url"} or not isinstance(raw_url, str):
            return _error_result(spec.kind, public_id, "invalid_tool_input")
        url = raw_url.strip()
        if not url:
            return _error_result(spec.kind, public_id, "invalid_tool_input")
        if len(url) > _MAX_FETCH_URL_CHARS:
            return _error_result(spec.kind, public_id, "url_too_long")
        if url not in provenance:
            return _error_result(spec.kind, public_id, "url_not_in_prior_context")
        egress = replace(self._web_fetch_egress, domains=spec.domains)
        fetched = await outbound._run_web_fetch(url, egress)
        content: dict[str, object] = {
            "type": "web_fetch_result",
            "url": fetched["url"],
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": fetched["media_type"],
                    "data": fetched["data"],
                },
                "title": fetched["title"],
            },
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        return _success_result(
            spec.kind,
            public_id,
            content,
            discovered_urls=frozenset({fetched["url"]}),
        )


async def _execute_search(
    spec: WebServerToolSpec,
    arguments: dict[str, object],
    *,
    public_id: str,
) -> _LocalToolResult:
    raw_query = arguments.get("query")
    if set(arguments) != {"query"} or not isinstance(raw_query, str):
        return _error_result(spec.kind, public_id, "invalid_tool_input")
    query = raw_query.strip()
    if not query:
        return _error_result(spec.kind, public_id, "invalid_tool_input")
    if len(query) > _MAX_SEARCH_QUERY_CHARS:
        return _error_result(spec.kind, public_id, "query_too_long")
    results = await outbound._run_web_search(query, spec.domains)
    content: list[dict[str, object]] = [
        {
            "type": "web_search_result",
            "title": result["title"],
            "url": result["url"],
        }
        for result in results
    ]
    return _success_result(
        spec.kind,
        public_id,
        content,
        discovered_urls=frozenset(result["url"] for result in results),
    )


async def _capture_stream(
    stream: AsyncIterator[str],
    buffer: list[str],
) -> AsyncIterator[str]:
    async for chunk in stream:
        buffer.append(chunk)
        yield chunk


def _selected_calls(
    blocks: list[dict[str, object]],
    plan: WebServerToolPlan,
) -> list[_SelectedServerCall]:
    selected: list[_SelectedServerCall] = []
    seen_ids: set[str] = set()
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        provider_id = block.get("id")
        name = block.get("name")
        arguments = block.get("input")
        if (
            not isinstance(provider_id, str)
            or not provider_id
            or provider_id in seen_ids
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
        ):
            raise _protocol_failure("The provider returned a malformed tool call.")
        spec = plan.spec_for_hidden_name(name)
        if spec is None:
            raise _protocol_failure(
                "The provider returned a tool call outside the declared server tools."
            )
        seen_ids.add(provider_id)
        selected.append(
            _SelectedServerCall(
                provider_id=provider_id,
                public_id=f"srvtoolu_{uuid.uuid4().hex}",
                spec=spec,
                arguments=_object_dict(arguments),
            )
        )
    return selected


def _public_selection_blocks(
    blocks: list[dict[str, object]],
    selected: list[_SelectedServerCall],
) -> tuple[list[dict[str, object]], list[str]]:
    by_provider_id = {call.provider_id: call for call in selected}
    public: list[dict[str, object]] = []
    public_ids: list[str] = []
    for block in blocks:
        if block.get("type") != "tool_use":
            public.append(block)
            continue
        provider_id = block.get("id")
        call = by_provider_id.get(provider_id) if isinstance(provider_id, str) else None
        if call is None:
            raise _protocol_failure("The provider returned an unknown tool call.")
        public.append(
            {
                "type": "server_tool_use",
                "id": call.public_id,
                "name": call.spec.kind,
                "input": call.arguments,
            }
        )
        public_ids.append(call.public_id)
    return public, public_ids


def _public_non_tool_blocks(
    blocks: list[dict[str, object]],
) -> list[dict[str, object]]:
    if any(block.get("type") == "tool_use" for block in blocks):
        raise _protocol_failure(
            "An internal tool call would have leaked to the client."
        )
    return blocks


def _message_blocks(message: dict[str, object]) -> list[dict[str, object]]:
    raw_content = message.get("content")
    if not isinstance(raw_content, list):
        raise _protocol_failure("The provider returned malformed message content.")
    blocks: list[dict[str, object]] = []
    for raw in raw_content:
        if not isinstance(raw, dict):
            raise _protocol_failure("The provider returned a malformed content block.")
        blocks.append(_object_dict(raw))
    return blocks


def _validate_round_frames(frames: list[str]) -> None:
    events = parse_sse_text("".join(frames))
    event_types = [event.event for event in events]
    if (
        not event_types
        or event_types[0] != "message_start"
        or event_types.count("message_start") != 1
    ):
        raise _protocol_failure(
            "The provider stream did not contain exactly one complete message start."
        )
    if event_types[-1] != "message_stop" or event_types.count("message_stop") != 1:
        raise _protocol_failure(
            "The provider stream did not contain exactly one complete message stop."
        )


def _message_stop_reason(message: dict[str, object]) -> str:
    value = message.get("stop_reason")
    return value if isinstance(value, str) and value else "end_turn"


def _add_usage(aggregate: dict[str, object], raw_usage: object) -> None:
    if not isinstance(raw_usage, dict):
        return
    for key, value in raw_usage.items():
        if (
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            previous = aggregate.get(key, 0)
            aggregate[key] = (previous if isinstance(previous, int) else 0) + value


def _with_server_usage(
    usage: dict[str, object],
    counts: dict[WebServerToolKind, int],
) -> dict[str, object]:
    combined = dict(usage)
    combined["server_tool_use"] = {
        "web_search_requests": counts["web_search"],
        "web_fetch_requests": counts["web_fetch"],
    }
    return combined


def _success_result(
    kind: WebServerToolKind,
    public_id: str,
    content: object,
    *,
    discovered_urls: frozenset[str],
) -> _LocalToolResult:
    internal = {"type": _PUBLIC_RESULT_TYPE[kind], "content": content}
    return _LocalToolResult(
        public_block={
            "type": _PUBLIC_RESULT_TYPE[kind],
            "tool_use_id": public_id,
            "content": content,
        },
        internal_content=internal,
        discovered_urls=discovered_urls,
        error_code=None,
    )


def _error_result(
    kind: WebServerToolKind,
    public_id: str,
    error_code: str,
) -> _LocalToolResult:
    error = {"type": _PUBLIC_ERROR_TYPE[kind], "error_code": error_code}
    return _LocalToolResult(
        public_block={
            "type": _PUBLIC_RESULT_TYPE[kind],
            "tool_use_id": public_id,
            "content": error,
        },
        internal_content={"type": _PUBLIC_RESULT_TYPE[kind], "content": error},
        discovered_urls=frozenset(),
        error_code=error_code,
    )


def _local_error_code(kind: WebServerToolKind, exc: BaseException) -> str:
    if isinstance(exc, WebFetchAccessError):
        return "url_not_accessible"
    if isinstance(exc, WebFetchEgressViolation):
        return "url_not_allowed"
    if isinstance(exc, UnsupportedWebContentError):
        return "unsupported_content_type"
    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, aiohttp.ClientResponseError):
        status = exc.status
    if status == 429:
        return "too_many_requests"
    if kind == "web_fetch" and isinstance(
        exc,
        (httpx.HTTPError, aiohttp.ClientError, TimeoutError, OSError),
    ):
        return "url_not_accessible"
    return "unavailable"


def _execution_failure_from_stream_error(error: dict[str, object]) -> ExecutionFailure:
    raw_type = error.get("type")
    error_type = raw_type if isinstance(raw_type, str) else "api_error"
    raw_message = error.get("message")
    message = (
        raw_message
        if isinstance(raw_message, str)
        else "Provider request failed unexpectedly."
    )
    status = anthropic_status_for_error_type(error_type)
    kind = {
        400: FailureKind.INVALID_REQUEST,
        401: FailureKind.AUTHENTICATION,
        403: FailureKind.PERMISSION,
        429: FailureKind.RATE_LIMIT,
        504: FailureKind.TIMEOUT,
        529: FailureKind.OVERLOADED,
    }.get(status, FailureKind.UPSTREAM)
    return ExecutionFailure(
        kind=kind,
        status_code=status,
        message=message,
        retryable=False,
    )


def _protocol_failure(message: str) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=500,
        message=message,
        retryable=False,
    )


def _object_dict(value: Mapping[object, object]) -> dict[str, object]:
    return {str(key): item for key, item in value.items()}


def _trace_finished(
    request_id: str,
    *,
    round_index: int,
    stop_reason: str,
    usage: dict[str, object],
    web_request_counts: dict[WebServerToolKind, int],
) -> None:
    trace_event(
        stage="egress",
        event="free_claude_code.api.web_server_tools.completed",
        source="api",
        request_id=request_id,
        round_count=round_index,
        stop_reason=stop_reason,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        web_search_requests=web_request_counts["web_search"],
        web_fetch_requests=web_request_counts["web_fetch"],
    )
