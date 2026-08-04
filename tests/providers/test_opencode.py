"""Tests for the OpenCode OpenAI-compatible provider."""

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from tests.providers.support import (
    capture_openai_chat_wire_body,
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.mark.parametrize("provider_id", ["opencode", "opencode_go"])
def test_build_request_body_replays_tool_reasoning_natively(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should inspect the file.",
                            "signature": "sig",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I should inspect the file."
    assert "<think>" not in assistant["content"]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["opencode", "opencode_go"])
async def test_tool_only_history_sends_empty_reasoning_content_on_wire(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_missing",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_missing",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))
    wire = await capture_openai_chat_wire_body(body)

    assistant = wire["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_missing"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert wire["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_missing",
        "content": "file contents",
    }
