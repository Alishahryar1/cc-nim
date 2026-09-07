import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.opencode.responses_tools import OpenCodeResponsesTools


def test_custom_tool_identity_and_history_preserve_namespaces() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input=[
            {
                "type": "custom_tool_call",
                "id": "custom",
                "call_id": "call_custom",
                "namespace": "editor",
                "name": "edit",
                "input": "patch",
            },
            {"type": "reasoning", "id": "reasoning", "encrypted_content": "opaque"},
        ],
        tools=[
            {"type": "custom", "name": "edit", "format": {"type": "text"}},
            {
                "type": "namespace",
                "name": "editor",
                "tools": [
                    {"type": "custom", "name": "edit", "format": {"type": "text"}}
                ],
            },
            {
                "type": "namespace",
                "name": "ordinary",
                "tools": [
                    {
                        "type": "function",
                        "name": "edit",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ],
        tool_choice={"type": "custom", "namespace": "editor", "name": "edit"},
    )
    original = request.model_dump()
    adapter = OpenCodeResponsesTools(request)
    wire = adapter.request.model_dump()
    nested = wire["tools"][1]["tools"][0]
    assert nested["type"] == "function"
    assert wire["input"][0]["name"] == nested["name"]
    assert wire["tool_choice"] == {
        "type": "function",
        "namespace": "editor",
        "name": nested["name"],
    }
    assert wire["input"][1] == original["input"][1]

    output = [
        {
            "type": "function_call",
            "name": nested["name"],
            "namespace": "editor",
            "call_id": "one",
            "arguments": '{"input":"patch"}',
        },
        {
            "type": "function_call",
            "name": "edit",
            "namespace": "ordinary",
            "call_id": "two",
            "arguments": "{}",
        },
    ]
    event: JsonObject = {
        "type": "response.completed",
        "sequence_number": 10,
        "response": {
            "id": "resp_one",
            "model": "example",
            "status": "completed",
            "output": output,
            "tools": wire["tools"],
        },
    }
    stream = "".join(adapter.presenter("example").feed("response.completed", event))
    restored = parse_sse_text(stream)[0].data["response"]
    assert restored["output"][0]["type"] == "custom_tool_call"
    assert restored["output"][0]["name"] == "edit"
    assert restored["output"][0]["namespace"] == "editor"
    assert restored["output"][0]["input"] == "patch"
    assert restored["output"][1] == output[1]
    assert restored["tools"] == original["tools"]
    assert request.model_dump() == original


def test_image_only_web_search_is_not_silently_replaced_with_text_search() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[{"type": "web_search", "search_content_types": ["image"]}],
    )
    with pytest.raises(InvalidRequestError, match="text web search only"):
        OpenCodeResponsesTools(request)


def test_custom_and_function_names_cannot_collide_after_conversion() -> None:
    request = OpenAIResponsesRequest(
        model="example",
        input="hello",
        tools=[
            {"type": "custom", "name": "edit"},
            {"type": "function", "name": "edit", "parameters": {"type": "object"}},
        ],
    )
    with pytest.raises(InvalidRequestError, match="names collide"):
        OpenCodeResponsesTools(request)
