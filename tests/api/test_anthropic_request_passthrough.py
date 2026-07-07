"""Pydantic passthrough of Anthropic protocol fields (e.g. ``cache_control``)."""

from api.models.anthropic import (
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockWebSearchToolResult,
    Message,
    MessagesRequest,
    SystemContent,
    Tool,
)
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)


def test_cache_control_preserved_on_parsed_user_text_system_and_tool() -> None:
    raw = {
        "model": "m",
        "max_tokens": 20,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "system": [
            {
                "type": "text",
                "text": "be brief",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "alpha",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    req = MessagesRequest.model_validate(raw)
    block = req.messages[0].content[0]
    assert isinstance(block, ContentBlockText)
    assert block.model_dump()["cache_control"] == {"type": "ephemeral"}
    s0 = req.system[0] if isinstance(req.system, list) else None
    assert isinstance(s0, SystemContent)
    assert s0.model_dump()["cache_control"] == {"type": "ephemeral"}
    t0 = req.tools[0] if req.tools else None
    assert isinstance(t0, Tool)
    assert t0.model_dump()["cache_control"] == {"type": "ephemeral"}


def test_build_base_native_body_includes_cache_control() -> None:
    req = MessagesRequest(
        model="m",
        max_tokens=20,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText.model_validate(
                        {
                            "type": "text",
                            "text": "x",
                            "cache_control": {"type": "ephemeral"},
                        }
                    )
                ],
            )
        ],
        system=[
            SystemContent.model_validate(
                {
                    "type": "text",
                    "text": "s",
                    "cache_control": {"type": "ephemeral"},
                }
            )
        ],
        tools=[
            Tool.model_validate(
                {
                    "name": "n",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            )
        ],
    )
    body = build_base_native_anthropic_request_body(
        req,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
    )
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_base_native_body_drops_unknown_top_level_client_hints() -> None:
    raw = {
        "model": "m",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "x"}],
        "reasoning_effort": "none",
        "unknown_client_hint": {"mode": "local"},
    }
    req = MessagesRequest.model_validate(raw)
    body = build_base_native_anthropic_request_body(
        req,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
    )
    assert "reasoning_effort" not in body
    assert "unknown_client_hint" not in body


def test_pydantic_discriminator_still_distinguishes_blocks() -> None:
    m = Message.model_validate(
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "z": 1}],
        }
    )
    b = m.content[0]
    assert isinstance(b, ContentBlockText)
    assert b.model_dump()["z"] == 1


def test_server_tool_assistant_blocks_round_trip_in_native_body() -> None:
    """Local server-tool responses must parse as valid history for a follow-up request."""
    raw = {
        "model": "m",
        "max_tokens": 20,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "q"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "T",
                                "url": "https://example.com",
                            }
                        ],
                    },
                ],
            }
        ],
        "mcp_servers": [{"type": "url", "url": "https://example.com/mcp"}],
    }
    req = MessagesRequest.model_validate(raw)
    assert len(req.messages) == 1
    blocks = req.messages[0].content
    assert isinstance(blocks, list)
    assert isinstance(blocks[0], ContentBlockServerToolUse)
    assert isinstance(blocks[1], ContentBlockWebSearchToolResult)
    body = build_base_native_anthropic_request_body(
        req,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
    )
    assert body["mcp_servers"][0]["type"] == "url"
    assert body["messages"][0]["content"][0]["type"] == "server_tool_use"
    assert body["messages"][0]["content"][1]["type"] == "web_search_tool_result"


def test_native_body_preserves_context_and_output_config() -> None:
    raw = {
        "model": "m",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "x"}],
        "context_management": {"edits": [{"type": "clear"}]},
        "output_config": {"some": "hint"},
    }
    req = MessagesRequest.model_validate(raw)
    body = build_base_native_anthropic_request_body(
        req,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=False,
    )
    assert body["context_management"] == raw["context_management"]
    assert body["output_config"] == raw["output_config"]
