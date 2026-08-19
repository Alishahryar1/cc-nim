import json
from collections.abc import AsyncIterator, Sequence
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.api.web_tools.coordinator import WebServerToolCoordinator
from free_claude_code.api.web_tools.egress import WebFetchEgressPolicy
from free_claude_code.api.web_tools.request import (
    HIDDEN_WEB_FETCH_NAME,
    HIDDEN_WEB_SEARCH_NAME,
    plan_web_server_tools,
)
from free_claude_code.api.web_tools.streaming import stream_composite_web_response
from free_claude_code.application.execution import ProviderExecutor, WireApi
from free_claude_code.application.routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    Message,
    MessagesRequest,
    Tool,
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy


class ScriptedExecutor(ProviderExecutor):
    def __init__(self, rounds: Sequence[list[str] | BaseException]) -> None:
        self._rounds = list(rounds)
        self.requests: list[MessagesRequest] = []
        self.internal_rounds: list[int] = []

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_id: str,
        internal_round: int = 1,
    ) -> AsyncIterator[str]:
        del wire_api, raw_log_label, raw_log_payload, request_id
        self.requests.append(routed.request.model_copy(deep=True))
        self.internal_rounds.append(internal_round)
        scripted = self._rounds.pop(0)

        async def body() -> AsyncIterator[str]:
            if isinstance(scripted, BaseException):
                raise scripted
            for frame in scripted:
                yield frame

        return body()


async def _provider_round(
    content: list[dict[str, object]],
    *,
    stop_reason: str,
    message_id: str = "msg_provider",
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> list[str]:
    return [
        frame
        async for frame in stream_composite_web_response(
            message_id=message_id,
            model="claude-sonnet",
            content=content,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            stop_reason=stop_reason,
        )
    ]


async def _aiter(frames: list[str]) -> AsyncIterator[str]:
    for frame in frames:
        yield frame


def _request(
    *,
    messages: list[Message] | None = None,
    tool_choice: dict[str, object] | None = None,
    tools: list[Tool] | None = None,
) -> MessagesRequest:
    return MessagesRequest(
        model="upstream-model",
        max_tokens=500,
        messages=messages or [Message(role="user", content="Use current information")],
        tools=tools or [Tool(name="web_search", type="web_search_20250305")],
        tool_choice=tool_choice,
        stream=True,
    )


def _routed(request: MessagesRequest) -> RoutedMessagesRequest:
    target = ProviderModelTarget(
        provider_id="nvidia_nim",
        provider_model=request.model,
        provider_model_ref=f"nvidia_nim/{request.model}",
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModelRoute(
            original_model="claude-sonnet",
            primary=target,
            fallbacks=(),
            reasoning_preference=ReasoningPreference.OFF,
        ),
        reasoning=ReasoningPolicy.off(),
    )


def _coordinator(executor: ProviderExecutor) -> WebServerToolCoordinator:
    return WebServerToolCoordinator(
        executor,
        web_fetch_egress=WebFetchEgressPolicy(
            allow_private_network_targets=False,
            allowed_schemes=frozenset({"http", "https"}),
        ),
    )


def _handler(executor: ProviderExecutor) -> MessagesHandler:
    settings = Settings.model_validate({"MODEL": "nvidia_nim/test-model"})
    return MessagesHandler(
        settings,
        provider_resolver=lambda _provider_id: MagicMock(),
        provider_executor=executor,
    )


async def _response_text(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            parts.append(chunk)
        else:
            parts.append(bytes(chunk).decode())
    return "".join(parts)


@pytest.mark.asyncio
async def test_auto_model_decline_replays_provider_stream_unchanged() -> None:
    provider_frames = await _provider_round(
        [{"type": "text", "text": "No search is needed."}],
        stop_reason="end_turn",
    )
    executor = ScriptedExecutor([provider_frames])
    request = _request(tool_choice={"type": "auto"})
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None

    actual = [
        frame
        async for frame in _coordinator(executor).stream(
            _routed(request), plan, request_id="req_decline"
        )
    ]

    assert actual == provider_frames
    assert executor.internal_rounds == [1]


@pytest.mark.asyncio
async def test_required_choice_without_server_call_is_protocol_failure() -> None:
    provider_frames = await _provider_round(
        [{"type": "text", "text": "declined"}],
        stop_reason="end_turn",
    )
    request = _request(tool_choice={"type": "any"})
    request.stream = False

    response = await _handler(ScriptedExecutor([provider_frames])).create(
        request, request_id="req_required"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    body = json.loads(bytes(response.body))
    assert "did not select the required" in body["error"]["message"]


@pytest.mark.asyncio
async def test_none_normalizes_completed_server_history_without_new_tools() -> None:
    history = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_history",
                    "name": "web_search",
                    "input": {"query": "history"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_history",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "History",
                            "url": "https://example.com/history",
                        }
                    ],
                },
            ],
        }
    )
    request = _request(
        messages=[history],
        tool_choice={"type": "none"},
    )
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None and plan.active
    provider_frames = await _provider_round(
        [{"type": "text", "text": "history understood"}],
        stop_reason="end_turn",
    )
    executor = ScriptedExecutor([provider_frames])

    actual = [
        frame
        async for frame in _coordinator(executor).stream(
            _routed(request), plan, request_id="req_none_history"
        )
    ]

    assert actual == provider_frames
    provider_request = executor.requests[0]
    assert provider_request.tools is None
    assert provider_request.tool_choice is None
    assert [message.role for message in provider_request.messages] == [
        "assistant",
        "user",
    ]


@pytest.mark.asyncio
async def test_model_selected_query_drives_search_and_continuation(
    monkeypatch,
) -> None:
    first = await _provider_round(
        [
            {"type": "thinking", "thinking": "I should search.", "signature": "sig"},
            {
                "type": "tool_use",
                "id": "call_provider",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "model-selected query"},
            },
        ],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=3,
    )
    second = await _provider_round(
        [{"type": "text", "text": "The model-authored final answer."}],
        stop_reason="end_turn",
        input_tokens=20,
        output_tokens=4,
    )
    executor = ScriptedExecutor([first, second])
    seen_queries: list[str] = []

    async def search(query: str, _domains: object) -> list[dict[str, str]]:
        seen_queries.append(query)
        return [{"title": "Example", "url": "https://example.com/source"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search",
        search,
    )
    request = _request(
        messages=[
            Message(role="user", content="Prompt text must not become the query")
        ],
        tool_choice={"type": "auto"},
    )
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None

    frames = [
        frame
        async for frame in _coordinator(executor).stream(
            _routed(request), plan, request_id="req_search"
        )
    ]
    message, error = await aggregate_anthropic_sse_to_message(_aiter(frames))

    assert error is None
    assert_anthropic_stream_contract(parse_sse_text("".join(frames)))
    assert seen_queries == ["model-selected query"]
    assert executor.internal_rounds == [1, 2]
    assert executor.requests[1].tool_choice == {"type": "auto"}
    assert isinstance(executor.requests[1].messages[-1].content, list)
    assert [block["type"] for block in message["content"]] == [
        "thinking",
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]
    serialized = json.dumps(message)
    assert HIDDEN_WEB_SEARCH_NAME not in serialized
    assert "call_provider" not in serialized
    assert "The model-authored final answer." in serialized
    assert message["usage"] == {
        "input_tokens": 30,
        "output_tokens": 7,
        "server_tool_use": {
            "web_search_requests": 1,
            "web_fetch_requests": 0,
        },
    }
    assert sum(frame.startswith("event: message_start") for frame in frames) == 1
    assert sum(frame.startswith("event: message_stop") for frame in frames) == 1


@pytest.mark.asyncio
async def test_parallel_server_calls_execute_in_provider_order(monkeypatch) -> None:
    first = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "one",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "first"},
            },
            {
                "type": "tool_use",
                "id": "two",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "second"},
            },
        ],
        stop_reason="tool_use",
    )
    final = await _provider_round(
        [{"type": "text", "text": "done"}], stop_reason="end_turn"
    )
    executor = ScriptedExecutor([first, final])
    order: list[str] = []

    async def search(query: str, _domains: object) -> list[dict[str, str]]:
        order.append(query)
        return [{"title": query, "url": f"https://example.com/{query}"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search",
        search,
    )
    request = _request()
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None

    frames = [
        frame
        async for frame in _coordinator(executor).stream(
            _routed(request), plan, request_id="req_parallel"
        )
    ]
    message, _ = await aggregate_anthropic_sse_to_message(_aiter(frames))

    assert order == ["first", "second"]
    assert [block["type"] for block in message["content"]] == [
        "server_tool_use",
        "server_tool_use",
        "web_search_tool_result",
        "web_search_tool_result",
        "text",
    ]


@pytest.mark.asyncio
async def test_max_uses_returns_in_band_error_without_second_search(
    monkeypatch,
) -> None:
    first = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "one",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "first"},
            },
            {
                "type": "tool_use",
                "id": "two",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "second"},
            },
        ],
        stop_reason="tool_use",
    )
    final = await _provider_round(
        [{"type": "text", "text": "handled"}], stop_reason="end_turn"
    )
    calls: list[str] = []

    async def search(query: str, _domains: object) -> list[dict[str, str]]:
        calls.append(query)
        return [{"title": query, "url": f"https://example.com/{query}"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    limited_tool = Tool.model_validate(
        {
            "name": "web_search",
            "type": "web_search_20250305",
            "max_uses": 1,
        }
    )
    request = _request(tools=[limited_tool])
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None

    frames = [
        frame
        async for frame in _coordinator(ScriptedExecutor([first, final])).stream(
            _routed(request), plan, request_id="req_limit"
        )
    ]
    message, _ = await aggregate_anthropic_sse_to_message(_aiter(frames))

    assert calls == ["first"]
    results = [
        block
        for block in message["content"]
        if block["type"] == "web_search_tool_result"
    ]
    assert results[1]["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }


@pytest.mark.asyncio
async def test_fetch_requires_prior_url_and_uses_model_argument(monkeypatch) -> None:
    selected = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "fetch",
                "name": HIDDEN_WEB_FETCH_NAME,
                "input": {"url": "https://example.com/page"},
            }
        ],
        stop_reason="tool_use",
    )
    final = await _provider_round(
        [{"type": "text", "text": "fetched"}], stop_reason="end_turn"
    )
    fetched_urls: list[str] = []

    async def fetch(url: str, _policy: object) -> dict[str, str]:
        fetched_urls.append(url)
        return {
            "url": url,
            "title": "Example",
            "media_type": "text/plain",
            "data": "page body",
        }

    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", fetch)
    fetch_tool = Tool(name="web_fetch", type="web_fetch_20250910")
    request = _request(
        messages=[Message(role="user", content="Read https://example.com/page for me")],
        tools=[fetch_tool],
    )
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None
    frames = [
        frame
        async for frame in _coordinator(ScriptedExecutor([selected, final])).stream(
            _routed(request), plan, request_id="req_fetch"
        )
    ]
    message, _ = await aggregate_anthropic_sse_to_message(_aiter(frames))

    assert fetched_urls == ["https://example.com/page"]
    assert [block["type"] for block in message["content"]] == [
        "server_tool_use",
        "web_fetch_tool_result",
        "text",
    ]


@pytest.mark.asyncio
async def test_fetch_url_outside_prior_context_is_in_band_and_not_fetched(
    monkeypatch,
) -> None:
    selected = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "fetch",
                "name": HIDDEN_WEB_FETCH_NAME,
                "input": {"url": "https://untrusted.example/page"},
            }
        ],
        stop_reason="tool_use",
    )
    final = await _provider_round(
        [{"type": "text", "text": "cannot fetch"}], stop_reason="end_turn"
    )
    fetch = MagicMock()
    monkeypatch.setattr("free_claude_code.api.web_tools.outbound._run_web_fetch", fetch)
    request = _request(tools=[Tool(name="web_fetch", type="web_fetch_20250910")])
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None
    frames = [
        frame
        async for frame in _coordinator(ScriptedExecutor([selected, final])).stream(
            _routed(request), plan, request_id="req_provenance"
        )
    ]
    message, _ = await aggregate_anthropic_sse_to_message(_aiter(frames))

    fetch.assert_not_called()
    result = message["content"][1]
    assert result["content"] == {
        "type": "web_fetch_tool_result_error",
        "error_code": "url_not_in_prior_context",
    }


@pytest.mark.asyncio
async def test_round_ten_returns_pause_turn_and_followup_resumes(monkeypatch) -> None:
    selecting_rounds = [
        await _provider_round(
            [
                {
                    "type": "tool_use",
                    "id": f"call_{index}",
                    "name": HIDDEN_WEB_SEARCH_NAME,
                    "input": {"query": f"query {index}"},
                }
            ],
            stop_reason="tool_use",
            message_id=f"msg_{index}",
        )
        for index in range(10)
    ]
    search_count = 0

    async def search(query: str, _domains: object) -> list[dict[str, str]]:
        nonlocal search_count
        search_count += 1
        return [{"title": query, "url": f"https://example.com/{search_count}"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search",
        search,
    )
    request = _request()
    plan = plan_web_server_tools(request, enabled=True)
    assert plan is not None
    first_executor = ScriptedExecutor(selecting_rounds)
    paused_frames = [
        frame
        async for frame in _coordinator(first_executor).stream(
            _routed(request), plan, request_id="req_pause"
        )
    ]
    paused, error = await aggregate_anthropic_sse_to_message(_aiter(paused_frames))

    assert error is None
    assert paused["stop_reason"] == "pause_turn"
    assert search_count == 9
    assert paused["content"][-1]["type"] == "server_tool_use"

    final = await _provider_round(
        [{"type": "text", "text": "resumed"}], stop_reason="end_turn"
    )
    followup = _request(
        messages=[
            Message.model_validate({"role": "assistant", "content": paused["content"]})
        ]
    )
    followup_plan = plan_web_server_tools(followup, enabled=True)
    assert followup_plan is not None
    second_executor = ScriptedExecutor([final])
    resumed_frames = [
        frame
        async for frame in _coordinator(second_executor).stream(
            _routed(followup), followup_plan, request_id="req_resume"
        )
    ]
    resumed, resumed_error = await aggregate_anthropic_sse_to_message(
        _aiter(resumed_frames)
    )

    assert resumed_error is None
    assert resumed["stop_reason"] == "end_turn"
    assert search_count == 10
    assert [block["type"] for block in resumed["content"]] == [
        "web_search_tool_result",
        "text",
    ]


@pytest.mark.asyncio
async def test_messages_handler_aggregates_exact_auto_shape(monkeypatch) -> None:
    first = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "call",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "current"},
            }
        ],
        stop_reason="tool_use",
    )
    final = await _provider_round(
        [{"type": "text", "text": "finished"}], stop_reason="end_turn"
    )
    executor = ScriptedExecutor([first, final])

    async def search(_query: str, _domains: object) -> list[dict[str, str]]:
        return [{"title": "Source", "url": "https://example.com"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search",
        search,
    )
    handler = _handler(executor)
    request = MessagesRequest(
        model="claude-sonnet",
        max_tokens=500,
        messages=[Message(role="user", content="Search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "auto"},
        stream=False,
    )

    response = await handler.create(request, request_id="req_handler")

    assert isinstance(response, JSONResponse)
    body = json.loads(bytes(response.body))
    assert body["stop_reason"] == "end_turn"
    assert [block["type"] for block in body["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]


@pytest.mark.asyncio
async def test_first_hidden_round_failure_is_typed_non_retryable_json() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream rate limited",
        retryable=False,
    )
    request = _request(tool_choice={"type": "auto"})
    request.stream = False

    response = await _handler(ScriptedExecutor([failure])).create(
        request, request_id="req_precommit"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert body["error"]["type"] == "rate_limit_error"


@pytest.mark.asyncio
async def test_empty_hidden_provider_round_is_not_partial_success() -> None:
    request = _request(tool_choice={"type": "auto"})
    request.stream = False

    response = await _handler(ScriptedExecutor([[]])).create(
        request, request_id="req_empty"
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    assert response.headers["x-should-retry"] == "false"
    body = json.loads(bytes(response.body))
    assert "exactly one complete message start" in body["error"]["message"]


@pytest.mark.asyncio
async def test_later_round_failure_is_terminal_error_without_success_stop(
    monkeypatch,
) -> None:
    selected = await _provider_round(
        [
            {
                "type": "tool_use",
                "id": "call",
                "name": HIDDEN_WEB_SEARCH_NAME,
                "input": {"query": "current"},
            }
        ],
        stop_reason="tool_use",
    )
    failure = ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="upstream overloaded",
        retryable=False,
    )

    async def search(_query: str, _domains: object) -> list[dict[str, str]]:
        return [{"title": "Source", "url": "https://example.com"}]

    monkeypatch.setattr(
        "free_claude_code.api.web_tools.outbound._run_web_search", search
    )
    request = _request(tool_choice={"type": "auto"})

    response = await _handler(ScriptedExecutor([selected, failure])).create(
        request, request_id="req_postcommit"
    )

    assert isinstance(response, StreamingResponse)
    body = await _response_text(response)
    assert "event: message_start" in body
    assert "event: error" in body
    assert "overloaded_error" in body
    assert "event: message_stop" not in body
