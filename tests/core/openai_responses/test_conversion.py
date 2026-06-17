from __future__ import annotations

import pytest

from core.openai_responses import (
    ResponsesConversionError,
    anthropic_message_response_to_openai_response,
    responses_request_to_anthropic_payload,
)


def test_responses_string_input_converts_to_anthropic_message() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "instructions": "System instructions",
            "input": "Hello",
            "max_output_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "metadata": {"trace": "abc"},
        }
    )

    assert payload["model"] == "nvidia_nim/test-model"
    assert payload["system"] == "System instructions"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["metadata"] == {"trace": "abc"}


def test_responses_messages_tools_and_tool_results_convert() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "deepseek/deepseek-chat",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Developer rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Use the tool"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "echo",
                    "arguments": '{"value":"FCC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Echo a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "echo"},
        }
    )

    assert payload["system"] == "Developer rules"
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Use the tool"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "echo",
                    "input": {"value": "FCC"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "FCC",
                }
            ],
        },
    ]
    assert payload["tools"] == [
        {
            "name": "echo",
            "description": "Echo a value",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "echo"}


def test_responses_tool_choice_none_disables_forwarded_tools() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "deepseek/deepseek-chat",
            "input": "Reply without tools",
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": "none",
        }
    )

    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_responses_namespace_tools_flatten_for_anthropic() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Use JS",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__node_repl",
                    "description": "Node tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "js",
                            "description": "Run JavaScript",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        }
                    ],
                }
            ],
            "tool_choice": {
                "type": "function",
                "namespace": "mcp__node_repl",
                "name": "js",
            },
        }
    )

    assert payload["tools"] == [
        {
            "name": "mcp__node_repl__js",
            "description": "Run JavaScript",
            "input_schema": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "mcp__node_repl__js"}


def test_responses_passive_codex_built_in_tools_are_ignored() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "image_generation", "output_format": "png"},
                {
                    "type": "function",
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )

    assert payload["tools"] == [
        {"name": "echo", "input_schema": {"type": "object", "properties": {}}}
    ]


def test_responses_namespaced_prior_function_call_flattens_tool_use_name() -> None:
    payload = responses_request_to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "namespace": "mcp__node_repl",
                    "name": "js",
                    "arguments": '{"code":"1+1"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "2",
                },
            ],
        }
    )

    assert payload["messages"][0]["content"][0]["name"] == "mcp__node_repl__js"


def test_anthropic_namespaced_tool_response_restores_responses_namespace() -> None:
    response = anthropic_message_response_to_openai_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "mcp__node_repl__js",
                    "input": {"code": "1+1"},
                }
            ]
        },
        {
            "model": "nvidia_nim/test-model",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__node_repl",
                    "tools": [
                        {
                            "type": "function",
                            "name": "js",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ],
        },
    )

    call = response["output"][0]
    assert call["namespace"] == "mcp__node_repl"
    assert call["name"] == "js"


def test_responses_unsupported_tool_type_is_clear() -> None:
    with pytest.raises(
        ResponsesConversionError, match="Unsupported Responses tool type"
    ):
        responses_request_to_anthropic_payload(
            {
                "model": "nvidia_nim/test-model",
                "input": "Hello",
                "tools": [{"type": "web_search_preview"}],
            }
        )


def test_responses_invalid_function_arguments_are_rejected() -> None:
    with pytest.raises(ResponsesConversionError, match="invalid JSON"):
        responses_request_to_anthropic_payload(
            {
                "model": "nvidia_nim/test-model",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "{",
                    }
                ],
            }
        )
