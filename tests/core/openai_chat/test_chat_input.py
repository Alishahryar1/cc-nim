import json

import pytest

from free_claude_code.core.openai_chat import (
    ChatCompletionsConversionError,
    OpenAIChatCompletionsRequest,
)
from free_claude_code.core.openai_chat.input import (
    convert_request_to_anthropic_payload,
)


def _payload(**kwargs):
    return convert_request_to_anthropic_payload(OpenAIChatCompletionsRequest(**kwargs))


def test_system_messages_collapse_into_system_field() -> None:
    payload = _payload(
        model="nvidia_nim/x",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "developer", "content": "obey policy"},
            {"role": "user", "content": "hello"},
        ],
    )
    assert payload["system"] == "be terse\n\nobey policy"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is True


def test_multimodal_user_content_becomes_blocks() -> None:
    payload = _payload(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ],
    )
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


def test_assistant_tool_calls_become_tool_use_blocks() -> None:
    payload = _payload(
        model="m",
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
        ],
    )
    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {
            "type": "tool_use",
            "id": "call_abc",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]
    tool_turn = payload["messages"][2]
    assert tool_turn == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_abc", "content": "sunny"}
        ],
    }


def test_consecutive_tool_results_merge_into_one_user_turn() -> None:
    payload = _payload(
        model="m",
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "g", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ],
    )
    tool_turn = payload["messages"][-1]
    assert [block["tool_use_id"] for block in tool_turn["content"]] == ["a", "b"]


def test_tools_and_tool_choice_convert() -> None:
    payload = _payload(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "find things",
                    "parameters": {"type": "object", "properties": {"q": {}}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
    )
    assert payload["tools"] == [
        {
            "name": "lookup",
            "input_schema": {"type": "object", "properties": {"q": {}}},
            "description": "find things",
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}


def test_tool_choice_none_drops_tools() -> None:
    payload = _payload(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="none",
    )
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_tool_choice_required_maps_to_any() -> None:
    payload = _payload(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="required",
    )
    assert payload["tool_choice"] == {"type": "any"}


def test_sampling_stop_and_token_limits() -> None:
    payload = _payload(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        top_p=0.9,
        max_tokens=100,
        max_completion_tokens=64,
        stop=["STOP", "END"],
    )
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    # max_completion_tokens takes precedence over the legacy max_tokens field.
    assert payload["max_tokens"] == 64
    assert payload["stop_sequences"] == ["STOP", "END"]


def test_invalid_tool_call_arguments_raise_conversion_error() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "f", "arguments": "{not json"}}
                    ],
                }
            ],
        )


def test_missing_non_system_message_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(model="m", messages=[{"role": "system", "content": "only system"}])


def test_tool_message_without_id_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(model="m", messages=[{"role": "tool", "content": "orphan"}])


def test_json_arguments_survive_roundtrip() -> None:
    payload = _payload(
        model="m",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "f", "arguments": '{"nested": {"x": 1}}'},
                    }
                ],
            }
        ],
    )
    assert (
        json.dumps(payload["messages"][0]["content"][0]["input"])
        == '{"nested": {"x": 1}}'
    )


def test_malformed_tool_entry_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"type": "function", "function": {"name": "ok"}},
                {"type": "function"},  # missing function object
            ],
        )


def test_unsupported_tool_type_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "retrieval", "function": {"name": "x"}}],
        )


def test_tool_function_missing_name_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"description": "no name"}}],
        )


def test_unknown_string_tool_choice_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="lookuq",  # typo for a real choice
        )


def test_named_tool_choice_missing_name_raises() -> None:
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice={"type": "function", "function": {}},
        )


def test_invalid_tool_choice_raises_even_without_tools() -> None:
    # tool_choice is validated unconditionally, matching the Responses dialect.
    with pytest.raises(ChatCompletionsConversionError):
        _payload(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="bogus",
        )
