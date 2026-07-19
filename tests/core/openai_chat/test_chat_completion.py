import json

from free_claude_code.core.openai_chat import OpenAIChatCompletionsRequest
from free_claude_code.core.openai_chat.completion import (
    anthropic_message_to_chat_completion,
)


def _request(model: str = "nvidia_nim/x") -> OpenAIChatCompletionsRequest:
    return OpenAIChatCompletionsRequest(
        model=model, messages=[{"role": "user", "content": "hi"}]
    )


def test_text_message_becomes_chat_completion() -> None:
    message = {
        "content": [{"type": "text", "text": "Hello there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    completion = anthropic_message_to_chat_completion(
        message, _request(), completion_id="chatcmpl-fixed"
    )
    assert completion["id"] == "chatcmpl-fixed"
    assert completion["object"] == "chat.completion"
    assert completion["model"] == "nvidia_nim/x"
    choice = completion["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello there"}
    assert choice["finish_reason"] == "stop"
    assert completion["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


def test_tool_use_message_maps_to_tool_calls_and_finish_reason() -> None:
    message = {
        "content": [
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 8, "output_tokens": 6},
    }
    completion = anthropic_message_to_chat_completion(message, _request())
    choice = completion["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "toolu_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    # No text content -> null content alongside tool_calls, per OpenAI shape.
    assert choice["message"]["content"] is None


def test_max_tokens_stop_reason_maps_to_length() -> None:
    message = {
        "content": [{"type": "text", "text": "partial"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    completion = anthropic_message_to_chat_completion(message, _request())
    assert completion["choices"][0]["finish_reason"] == "length"


def test_missing_usage_defaults_to_zero() -> None:
    completion = anthropic_message_to_chat_completion(
        {"content": [{"type": "text", "text": "x"}]}, _request()
    )
    assert completion["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert completion["choices"][0]["finish_reason"] == "stop"
