import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.openai_responses.chat_request import (
    build_responses_chat_request,
)
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest


def _request(**overrides: object) -> OpenAIResponsesRequest:
    payload: dict[str, object] = {
        "model": "provider/model",
        "input": "Hello",
    }
    payload.update(overrides)
    return OpenAIResponsesRequest.model_validate(payload)


def test_build_responses_chat_request_preserves_rich_supported_semantics() -> None:
    translated = build_responses_chat_request(
        _request(
            instructions="System rules",
            input=[
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                            "detail": "low",
                        },
                    ],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Use the tool."}],
                    "encrypted_content": "opaque-replay",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "namespace": "mcp.shell",
                    "name": "echo value",
                    "arguments": '{"value":"FCC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
                {"type": "message", "role": "user", "content": "Continue"},
            ],
            tools=[
                {"type": "web_search", "external_web_access": True},
                {
                    "type": "namespace",
                    "name": "mcp.shell",
                    "tools": [
                        {
                            "type": "function",
                            "name": "echo value",
                            "description": "Echo a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                            "strict": True,
                        }
                    ],
                },
                {
                    "type": "custom",
                    "name": "apply patch",
                    "description": "Apply a patch",
                    "format": {"type": "text"},
                },
            ],
            tool_choice={
                "type": "function",
                "namespace": "mcp.shell",
                "name": "echo value",
            },
            parallel_tool_calls=False,
            max_output_tokens=128,
            temperature=0.2,
            top_p=0.9,
            metadata={"trace": "abc"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                    },
                    "strict": True,
                }
            },
        ),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details=True,
    )

    body = translated.body
    namespace_name = "mcp_shell__echo_value"
    namespace_alias = translated.tool_names.encode(namespace_name)
    custom_alias = translated.tool_names.encode("apply patch")
    assert namespace_alias == namespace_name
    assert custom_alias != "apply patch"

    assert body == {
        "model": "provider/model",
        "messages": [
            {
                "role": "system",
                "content": "System rules\n\nDeveloper rules",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AA==",
                            "detail": "low",
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Use the tool.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "opaque-replay"}
                ],
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": namespace_name,
                            "arguments": '{"value":"FCC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "FCC"},
            {"role": "user", "content": "Continue"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": namespace_name,
                    "description": "Echo a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply patch",
                    "description": (
                        "Apply a patch\n\nCustom tool input format: unconstrained text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Free-form input for the custom tool.",
                            }
                        },
                        "required": ["input"],
                    },
                },
            },
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": namespace_name},
        },
        "parallel_tool_calls": False,
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "metadata": {"trace": "abc"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
                "strict": True,
            },
        },
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ReasoningReplayMode.REASONING_CONTENT, {"reasoning_content": "Think"}),
        (ReasoningReplayMode.REASONING, {"reasoning": "Think"}),
        (ReasoningReplayMode.THINK_TAGS, {"content": "<think>\nThink\n</think>"}),
        (ReasoningReplayMode.DISABLED, {}),
    ],
)
def test_build_responses_chat_request_uses_provider_reasoning_replay(
    mode: ReasoningReplayMode,
    expected: dict[str, str],
) -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Think"}],
                },
                {"type": "message", "role": "assistant", "content": "Answer"},
            ]
        ),
        reasoning_replay=mode,
    )

    messages = translated.body["messages"]
    assert isinstance(messages, list)
    assistant = messages[0]
    assert isinstance(assistant, dict)
    if mode is ReasoningReplayMode.THINK_TAGS:
        assert assistant["content"] == "<think>\nThink\n</think>\n\nAnswer"
    else:
        for key, value in expected.items():
            assert assistant[key] == value
        if mode is ReasoningReplayMode.DISABLED:
            assert assistant == {"role": "assistant", "content": "Answer"}


def test_build_responses_chat_request_quarantines_one_malformed_call_pair() -> None:
    translated = build_responses_chat_request(
        _request(
            input=[
                {"role": "user", "content": "Hello"},
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "echo",
                    "arguments": "{",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_bad",
                    "output": "stale",
                },
                {
                    "type": "function_call",
                    "call_id": "call_good",
                    "name": "echo",
                    "arguments": '{"value":"ok"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_good",
                    "output": "ok",
                },
            ]
        ),
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    )

    assert translated.body["messages"] == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_good",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"value":"ok"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_good", "content": "ok"},
    ]


def test_build_responses_chat_request_skips_unsupported_optional_items_and_choice() -> (
    None
):
    translated = build_responses_chat_request(
        _request(
            input=[
                {"type": "computer_screenshot", "image_url": "ignored"},
                {"role": "user", "content": "Hello"},
            ],
            tools=[{"type": "web_search_preview"}],
            tool_choice={"type": "web_search_preview"},
        ),
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert translated.body == {
        "model": "provider/model",
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_build_responses_chat_request_rejects_no_usable_input() -> None:
    with pytest.raises(ResponsesConversionError, match="usable input"):
        build_responses_chat_request(
            _request(input=[{"type": "computer_screenshot", "image_url": "x"}]),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_rejects_message_with_only_skipped_parts() -> None:
    with pytest.raises(ResponsesConversionError, match="usable input"):
        build_responses_chat_request(
            _request(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "file_id": "file_not_representable_in_chat",
                            }
                        ],
                    }
                ]
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )


def test_build_responses_chat_request_rejects_colliding_tool_wire_names() -> None:
    with pytest.raises(ResponsesConversionError, match="same Chat-compatible name"):
        build_responses_chat_request(
            _request(
                tools=[
                    {
                        "type": "function",
                        "name": "mcp_shell__echo_value",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "namespace",
                        "name": "mcp.shell",
                        "tools": [
                            {
                                "type": "function",
                                "name": "echo value",
                                "parameters": {"type": "object"},
                            }
                        ],
                    },
                ],
                tool_choice={
                    "type": "function",
                    "namespace": "mcp.shell",
                    "name": "echo value",
                },
            ),
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )
