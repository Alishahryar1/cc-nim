import pytest

from free_claude_code.api.web_tools.request import (
    HIDDEN_WEB_FETCH_NAME,
    HIDDEN_WEB_SEARCH_NAME,
    plan_web_server_tools,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import Message, MessagesRequest, Tool


def _request(
    *,
    tools: list[Tool],
    tool_choice: dict[str, object] | None = None,
    messages: list[Message] | None = None,
) -> MessagesRequest:
    return MessagesRequest(
        model="test/model",
        max_tokens=200,
        messages=messages or [Message(role="user", content="Find current information")],
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
    )


def _search(**options: object) -> Tool:
    payload: dict[str, object] = {
        "name": "web_search",
        "type": "web_search_20250305",
    }
    payload.update(options)
    return Tool.model_validate(payload)


def _fetch(**options: object) -> Tool:
    payload: dict[str, object] = {
        "name": "web_fetch",
        "type": "web_fetch_20250910",
    }
    payload.update(options)
    return Tool.model_validate(payload)


@pytest.mark.parametrize("name", ["web_search", "web_fetch", "WebSearch", "WebFetch"])
def test_ordinary_function_names_remain_ordinary(name: str) -> None:
    request = _request(
        tools=[
            Tool(
                name=name,
                description="ordinary client tool",
                input_schema={"type": "object"},
            )
        ]
    )

    assert plan_web_server_tools(request, enabled=True) is None


def test_exact_claude_auto_shape_translates_to_hidden_function() -> None:
    plan = plan_web_server_tools(
        _request(tools=[_search()], tool_choice={"type": "auto"}),
        enabled=True,
    )

    assert plan is not None
    assert plan.active
    assert plan.choice == "auto"
    assert plan.request.tool_choice == {"type": "auto"}
    assert [tool.name for tool in plan.request.tools or []] == [HIDDEN_WEB_SEARCH_NAME]
    assert plan.request.tools and plan.request.tools[0].type is None


def test_forced_fetch_translates_choice_but_not_arguments() -> None:
    plan = plan_web_server_tools(
        _request(
            tools=[_fetch()],
            tool_choice={"type": "tool", "name": "web_fetch"},
        ),
        enabled=True,
    )

    assert plan is not None
    assert plan.forced_kind == "web_fetch"
    assert plan.request.tool_choice == {
        "type": "tool",
        "name": HIDDEN_WEB_FETCH_NAME,
    }
    assert plan.request.tools and plan.request.tools[0].input_schema == {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }


def test_none_strips_server_tools_for_normal_provider_execution() -> None:
    plan = plan_web_server_tools(
        _request(tools=[_search()], tool_choice={"type": "none"}),
        enabled=True,
    )

    assert plan is not None
    assert not plan.active
    assert plan.request.tools is None
    assert plan.request.tool_choice is None


def test_search_and_fetch_can_be_listed_together() -> None:
    plan = plan_web_server_tools(
        _request(tools=[_search(max_uses=2), _fetch(max_uses=3)]),
        enabled=True,
    )

    assert plan is not None
    assert [spec.kind for spec in plan.specs] == ["web_search", "web_fetch"]
    assert [spec.max_uses for spec in plan.specs] == [2, 3]


@pytest.mark.parametrize(
    ("tools", "choice", "message"),
    [
        (
            [_search(), Tool(name="ordinary", input_schema={"type": "object"})],
            None,
            "Mixing",
        ),
        ([Tool(name="web_search", type="web_search")], None, "Unsupported"),
        ([Tool(name="web_search", type="web_search_20260101")], None, "Unsupported"),
        ([Tool(name="wrong", type="web_search_20250305")], None, "canonical name"),
        ([_search(), _search()], None, "declared once"),
        ([_search(description="custom")], None, "description"),
        ([_search(max_uses=0)], None, "positive integer"),
        ([_search(locale="en-US")], None, "unsupported option"),
        ([_search()], {"type": "tool", "name": "web_fetch"}, "not declared"),
        ([_search()], {"type": "auto", "name": "web_search"}, "valid only"),
        ([_search()], {"type": "bogus"}, "must be auto"),
        (
            [_search()],
            {"type": "auto", "disable_parallel_tool_use": True},
            "not supported",
        ),
    ],
)
def test_invalid_server_tool_contract_fails_before_execution(
    tools: list[Tool],
    choice: dict | None,
    message: str,
) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        plan_web_server_tools(
            _request(tools=tools, tool_choice=choice),
            enabled=True,
        )


def test_disabled_setting_rejects_listed_tool() -> None:
    with pytest.raises(InvalidRequestError, match="disabled"):
        plan_web_server_tools(_request(tools=[_search()]), enabled=False)


def test_domain_options_are_normalized_and_mutually_exclusive() -> None:
    plan = plan_web_server_tools(
        _request(tools=[_search(allowed_domains=["Example.com/docs"])]),
        enabled=True,
    )

    assert plan is not None
    spec = plan.specs[0]
    assert spec.domains.allowed[0].host == "example.com"
    assert spec.domains.allowed[0].path == "/docs"

    with pytest.raises(InvalidRequestError, match="not both"):
        plan_web_server_tools(
            _request(
                tools=[
                    _search(
                        allowed_domains=["example.com"],
                        blocked_domains=["private.example.com"],
                    )
                ]
            ),
            enabled=True,
        )


def test_fetch_domain_filter_rejects_paths() -> None:
    with pytest.raises(InvalidRequestError, match="must not contain a path"):
        plan_web_server_tools(
            _request(tools=[_fetch(allowed_domains=["example.com/docs"])]),
            enabled=True,
        )


def test_server_history_requires_current_definition() -> None:
    request = _request(
        tools=[],
        messages=[
            Message(
                role="assistant",
                content=[
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_prior",
                        "name": "web_search",
                        "input": {"query": "latest news"},
                    }
                ],
            )
        ],
    )

    with pytest.raises(InvalidRequestError, match="matching supported"):
        plan_web_server_tools(request, enabled=True)


def test_none_cannot_resume_pending_server_call() -> None:
    pending = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_pending",
                    "name": "web_search",
                    "input": {"query": "current"},
                }
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="cannot resume"):
        plan_web_server_tools(
            _request(
                tools=[_search()],
                tool_choice={"type": "none"},
                messages=[pending],
            ),
            enabled=True,
        )


def test_server_result_without_call_is_rejected_during_planning() -> None:
    result = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_missing",
                    "content": [],
                }
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="without a matching call"):
        plan_web_server_tools(
            _request(tools=[_search()], messages=[result]), enabled=True
        )


def test_pending_server_call_must_end_the_transcript() -> None:
    pending = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_pending",
                    "name": "web_search",
                    "input": {"query": "current"},
                }
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="end of the transcript"):
        plan_web_server_tools(
            _request(
                tools=[_search()],
                messages=[pending, Message(role="user", content="continue")],
            ),
            enabled=True,
        )


def test_pending_server_and_client_calls_cannot_be_resumed_together() -> None:
    mixed_pending = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_pending",
                    "name": "web_search",
                    "input": {"query": "current"},
                },
                {
                    "type": "tool_use",
                    "id": "client_pending",
                    "name": "Read",
                    "input": {"file_path": "README.md"},
                },
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="server and client tools"):
        plan_web_server_tools(
            _request(tools=[_search()], messages=[mixed_pending]), enabled=True
        )


def test_ordinary_history_cannot_collide_with_hidden_server_tool_name() -> None:
    ordinary = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "ordinary_call",
                    "name": HIDDEN_WEB_SEARCH_NAME,
                    "input": {"query": "ordinary"},
                }
            ],
        }
    )

    with pytest.raises(InvalidRequestError, match="name reserved"):
        plan_web_server_tools(
            _request(tools=[_search()], messages=[ordinary]), enabled=True
        )
