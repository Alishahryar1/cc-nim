import json

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesRequest,
)
from free_claude_code.core.openai_responses.errors import ResponsesConversionError
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy


def test_build_responses_provider_request_preserves_multiturn_protocol() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "max_tokens": 4096,
            "system": "System instructions",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "summary"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Calling a tool"},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {"q": "value"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": {"answer": 42},
                        },
                        {"type": "text", "text": "Continue"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGVsbG8=",
                            },
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup"},
        }
    )

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.on(effort=ReasoningEffort.XHIGH),
    )

    assert body["model"] == "gpt-test"
    assert body["instructions"] == "System instructions"
    assert body["max_output_tokens"] == 4096
    assert body["stream"] is True
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["input"][0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "summary"}],
        "encrypted_content": "opaque",
    }
    assert body["input"][1]["role"] == "assistant"
    assert body["input"][2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps(
            {"q": "value"}, ensure_ascii=False, separators=(",", ":")
        ),
    }
    assert body["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"answer": 42}',
    }
    assert body["input"][4]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,aGVsbG8=",
    }


def test_responses_provider_request_uses_one_portable_tool_alias() -> None:
    original = "mcp__responses_provider__" + "x" * 70
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": original,
                            "input": {"q": "value"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "done",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": original,
                    "description": "Long tool",
                    "input_schema": {"type": "object"},
                },
                {"name": "safe_tool", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": original},
        }
    )
    snapshot = request.model_dump()

    body = build_responses_provider_request(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )

    alias = body["tools"][0]["name"]
    assert alias != original
    assert len(alias) <= 64
    assert body["tools"][1]["name"] == "safe_tool"
    assert body["tool_choice"] == {"type": "function", "name": alias}
    function_call = next(
        item for item in body["input"] if item["type"] == "function_call"
    )
    assert function_call["name"] == alias
    assert function_call["call_id"] == "call_1"
    assert function_call["arguments"] == '{"q":"value"}'
    assert request.model_dump() == snapshot


def test_responses_round_trip_preserves_encrypted_reasoning_and_tool_ids() -> None:
    adapter = OpenAIResponsesAdapter()
    ingress = OpenAIResponsesRequest.model_validate(
        {
            "model": "openai/gpt-test",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Use a tool."}],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "function_call",
                    "call_id": "call_stable",
                    "name": "lookup",
                    "arguments": '{"q":"value"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_stable",
                    "output": "done",
                },
            ],
        }
    )
    anthropic = MessagesRequest.model_validate(adapter.to_anthropic_payload(ingress))

    body = build_responses_provider_request(
        anthropic,
        reasoning=ReasoningPolicy.provider_default(),
    )

    assert body["input"][:3] == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Use a tool."}],
            "encrypted_content": "opaque-reasoning",
        },
        {
            "type": "function_call",
            "call_id": "call_stable",
            "name": "lookup",
            "arguments": '{"q":"value"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_stable",
            "output": "done",
        },
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_sequences", ["stop"]),
        ("top_k", 4),
        ("context_management", {"edits": []}),
        ("output_config", {"effort": "high"}),
        ("mcp_servers", [{"name": "server"}]),
        ("extra_body", {"unknown": True}),
    ],
)
def test_build_responses_provider_request_rejects_lossy_fields(
    field: str, value: object
) -> None:
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    with pytest.raises(ResponsesConversionError, match=field):
        build_responses_provider_request(
            MessagesRequest.model_validate(payload),
            reasoning=ReasoningPolicy.provider_default(),
        )


def test_build_responses_provider_request_rejects_provider_managed_tools() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    with pytest.raises(ResponsesConversionError, match="web_search_20250305"):
        build_responses_provider_request(
            request,
            reasoning=ReasoningPolicy.provider_default(),
        )
