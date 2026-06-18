from __future__ import annotations

import pytest

from core.openai_responses import OpenAIResponsesAdapter

_ADAPTER = OpenAIResponsesAdapter()
_CONVERSION_ERROR = OpenAIResponsesAdapter.ConversionError


def test_responses_string_input_converts_to_anthropic_message() -> None:
    payload = _ADAPTER.to_anthropic_payload(
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
    payload = _ADAPTER.to_anthropic_payload(
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
    payload = _ADAPTER.to_anthropic_payload(
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
    payload = _ADAPTER.to_anthropic_payload(
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


def test_responses_namespaced_tool_choice_type_tool_flattens_for_anthropic() -> None:
    payload = _ADAPTER.to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Use JS",
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
            "tool_choice": {
                "type": "tool",
                "namespace": "mcp__node_repl",
                "name": "js",
            },
        }
    )

    assert payload["tool_choice"] == {"type": "tool", "name": "mcp__node_repl__js"}


def test_responses_custom_tool_converts_to_anthropic_string_tool() -> None:
    payload = _ADAPTER.to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Use apply_patch",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a repo patch",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/",
                    },
                }
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }
    )

    assert payload["tools"] == [
        {
            "name": "apply_patch",
            "description": (
                "Apply a repo patch\n\n"
                "Custom tool input format: grammar (lark): start: /.+/"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Free-form input for the custom tool.",
                    }
                },
                "required": ["input"],
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "apply_patch"}


def test_responses_namespaced_custom_tool_flattens_for_anthropic() -> None:
    payload = _ADAPTER.to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Use shell",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__shell",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "exec",
                            "description": "Run shell text",
                            "format": {"type": "text"},
                        }
                    ],
                }
            ],
            "tool_choice": {
                "type": "custom",
                "namespace": "mcp__shell",
                "custom": {"name": "exec"},
            },
        }
    )

    assert payload["tools"][0]["name"] == "mcp__shell__exec"
    assert payload["tools"][0]["description"] == (
        "Run shell text\n\nCustom tool input format: unconstrained text."
    )
    assert payload["tool_choice"] == {"type": "tool", "name": "mcp__shell__exec"}


def test_responses_passive_codex_built_in_tools_are_ignored() -> None:
    payload = _ADAPTER.to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "image_generation", "output_format": "png"},
                {"type": "tool_search"},
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
    payload = _ADAPTER.to_anthropic_payload(
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


def test_responses_prior_custom_tool_call_flattens_tool_use_name() -> None:
    payload = _ADAPTER.to_anthropic_payload(
        {
            "model": "nvidia_nim/test-model",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "namespace": "mcp__shell",
                    "name": "exec",
                    "input": "printf FCC",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "FCC",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "mcp__shell__exec",
                    "input": {"input": "printf FCC"},
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


def test_responses_unsupported_tool_type_is_clear() -> None:
    with pytest.raises(_CONVERSION_ERROR, match="Unsupported Responses tool type"):
        _ADAPTER.to_anthropic_payload(
            {
                "model": "nvidia_nim/test-model",
                "input": "Hello",
                "tools": [{"type": "web_search_preview"}],
            }
        )


def test_responses_invalid_function_arguments_are_rejected() -> None:
    with pytest.raises(_CONVERSION_ERROR, match="invalid JSON"):
        _ADAPTER.to_anthropic_payload(
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
