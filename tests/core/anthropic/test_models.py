import pytest
from pydantic import ValidationError

from free_claude_code.core.anthropic.ingress import (
    AnthropicIngressError,
    messages_to_inference_request,
    token_count_to_inference_request,
)
from free_claude_code.core.anthropic.models import (
    ContentBlockDocument,
    ContentBlockWebFetchToolResult,
    Message,
    MessagesRequest,
    TokenCountRequest,
)
from free_claude_code.core.inference import (
    CacheTTL,
    DocumentContent,
    ImageContent,
    InstructionItem,
    MessageItem,
    ReasoningItem,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
    TextContent,
    ToolCallItem,
)
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort
from free_claude_code.core.replay_envelope import encode_replay_envelope


def test_messages_request_parses_without_model_mapping_side_effects():
    request = MessagesRequest(
        model="claude-3-opus",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )

    assert request.model == "claude-3-opus"
    assert request.stream is False


def test_messages_request_rejects_null_stream() -> None:
    with pytest.raises(ValidationError):
        MessagesRequest.model_validate(
            {
                "model": "claude-3-opus",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": None,
            }
        )


def test_messages_request_preserves_system_role_message_order():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "second"},
            ],
        }
    )

    assert [message.role for message in request.messages] == [
        "user",
        "system",
        "user",
    ]
    assert request.messages[1].content == "system prompt"
    assert request.system is None


def test_messages_request_keeps_top_level_and_inline_system_content_distinct():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": "existing system",
            "messages": [
                {"role": "system", "content": "message system"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert request.system == "existing system"
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == "message system"


def test_messages_request_preserves_inline_system_block_metadata():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": [
                {
                    "type": "text",
                    "text": "existing system",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "message system",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert len(request.messages) == 2
    assert isinstance(request.system, list)
    assert [block.text for block in request.system] == ["existing system"]
    assert request.system[0].model_dump()["cache_control"] == {"type": "ephemeral"}
    inline_content = request.messages[0].content
    assert isinstance(inline_content, list)
    assert inline_content[0].model_dump() == {
        "type": "text",
        "text": "message system",
        "cache_control": {"type": "ephemeral"},
    }


def test_messages_ingress_rejects_internal_routing_fields_when_supplied() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "target-model",
            "original_model": "claude-3-opus",
            "resolved_provider_model": "nvidia_nim/target-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert request.model == "target-model"
    with pytest.raises(AnthropicIngressError, match="original_model"):
        messages_to_inference_request(request)


def test_token_count_request_parses_without_model_mapping_side_effects():
    request = TokenCountRequest(
        model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
    )

    assert request.model == "claude-3-sonnet"


def test_token_count_request_preserves_system_role_messages():
    request = TokenCountRequest.model_validate(
        {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "system", "content": "counting system"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].content == "counting system"
    assert request.system is None


def test_messages_request_preserves_thinking_signature():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "signed thought",
                            "signature": "sig_123",
                        }
                    ],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0]["signature"] == "sig_123"


def test_messages_request_preserves_native_thinking_budget():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "think hard"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "enabled"
    assert dumped["thinking"]["budget_tokens"] == 4096


def test_messages_request_accepts_adaptive_thinking_type():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "adaptive"


def test_messages_request_accepts_anthropic_server_tool_without_input_schema():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["tools"] == [{"name": "web_search", "type": "web_search_20250305"}]


def test_messages_request_accepts_redacted_thinking_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0] == {
        "type": "redacted_thinking",
        "data": "opaque",
    }


def test_document_and_web_fetch_blocks_preserve_protocol_extensions() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "data": "encoded"},
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "web_fetch_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": {"url": "https://example.com"},
                            "provider_extension": True,
                        },
                    ],
                }
            ],
        }
    )

    content = request.messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], ContentBlockDocument)
    assert content[0].model_dump()["cache_control"] == {"type": "ephemeral"}
    assert isinstance(content[1], ContentBlockWebFetchToolResult)
    assert content[1].model_dump()["provider_extension"] is True


def test_content_block_descriptions_remain_in_the_public_schema() -> None:
    definitions = MessagesRequest.model_json_schema()["$defs"]

    assert definitions["ContentBlockDocument"]["description"] == (
        "Anthropic document block (e.g. PDF files via the Files API)."
    )
    assert definitions["ContentBlockServerToolUse"]["description"] == (
        "Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."
    )


def test_messages_ingress_rejects_unknown_fields_after_permissive_wire_parse() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
            "original_model": "original",
            "resolved_provider_model": "provider/model",
            "betas": ["feature-beta"],
            "client_extension": {"enabled": True},
        }
    )

    with pytest.raises(AnthropicIngressError, match="client_extension"):
        messages_to_inference_request(request)


def test_token_count_ingress_rejects_unknown_internal_fields() -> None:
    request = TokenCountRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "original_model": "original",
            "resolved_provider_model": "provider/model",
            "betas": ["feature-beta"],
            "client_extension": "accepted",
        }
    )

    with pytest.raises(AnthropicIngressError, match="client_extension"):
        token_count_to_inference_request(request)


def test_anthropic_ingress_rejects_unknown_top_level_extensions() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "client_extension": True,
        }
    )

    with pytest.raises(AnthropicIngressError, match="client_extension"):
        messages_to_inference_request(request)


def test_messages_ingress_preserves_cache_controls_at_supported_locations() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "system": [
                {
                    "type": "text",
                    "text": "Top-level",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Inline",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Inspect",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.invalid/image.png",
                            },
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_1"},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )

    canonical = messages_to_inference_request(request)

    top_level, inline, *message_items = canonical.items
    assert isinstance(top_level, InstructionItem)
    assert top_level.cache_control is not None
    assert top_level.cache_control.ttl is CacheTTL.FIVE_MINUTES
    assert isinstance(inline, InstructionItem)
    assert inline.cache_control is not None
    assert inline.cache_control.ttl is CacheTTL.ONE_HOUR
    assert all(isinstance(item, MessageItem) for item in message_items)
    content = tuple(
        item.content[0] for item in message_items if isinstance(item, MessageItem)
    )
    assert [type(part) for part in content] == [
        TextContent,
        ImageContent,
        DocumentContent,
    ]
    text, image, document = content
    assert isinstance(text, TextContent)
    assert isinstance(image, ImageContent)
    assert isinstance(document, DocumentContent)
    assert text.cache_control is not None
    assert image.cache_control is not None
    assert document.cache_control is not None
    assert canonical.tools[0].cache_control is not None


def test_messages_ingress_normalizes_known_no_op_controls() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Hello"}],
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "high"},
            "mcp_servers": [],
        }
    )

    canonical = messages_to_inference_request(request)

    assert canonical.reasoning.control is ReasoningControl.DEFAULT
    assert canonical.reasoning.effort is ReasoningEffort.HIGH


def test_messages_ingress_normalizes_claude_thinking_display_hint() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
        }
    )

    canonical = messages_to_inference_request(request)

    assert canonical.reasoning.control is ReasoningControl.ON


@pytest.mark.parametrize("display", [None, "", "shown", False, {"mode": "omitted"}])
def test_messages_ingress_rejects_unknown_thinking_display_values(
    display: object,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "adaptive", "display": display},
        }
    )

    with pytest.raises(AnthropicIngressError, match=r"thinking\.display"):
        messages_to_inference_request(request)


def test_messages_replay_carriers_restore_reasoning_and_tool_artifacts() -> None:
    scope = ReplayCompatibilityScope("provider:model")
    reasoning_artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI,
        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
        attachment=ReplayAttachment.REASONING,
        scope=scope,
        payload="opaque-reasoning",
    )
    tool_artifact = ReplayArtifact(
        origin=ReplayArtifactOrigin.GOOGLE,
        kind=ReplayArtifactKind.THOUGHT_SIGNATURE,
        attachment=ReplayAttachment.TOOL_CALL,
        scope=scope,
        payload="opaque-tool",
    )
    request = MessagesRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "summary",
                            "signature": encode_replay_envelope((reasoning_artifact,)),
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {},
                            "extra_content": {
                                "fcc_replay": encode_replay_envelope((tool_artifact,))
                            },
                        },
                    ],
                }
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        }
    )

    canonical = messages_to_inference_request(request)
    reasoning = next(
        item for item in canonical.items if isinstance(item, ReasoningItem)
    )
    tool_call = next(item for item in canonical.items if isinstance(item, ToolCallItem))

    assert reasoning.artifacts == (reasoning_artifact,)
    assert tool_call.artifacts == (tool_artifact,)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "last"}]
                }
            },
            "context_management",
        ),
        ({"output_config": {"format": {"type": "json"}}}, "output_config.format"),
        (
            {"mcp_servers": [{"type": "url", "url": "https://example.invalid"}]},
            "mcp_servers",
        ),
        ({"thinking": {"type": "future"}}, "thinking.type"),
        (
            {"thinking": {"type": "adaptive", "client_extension": True}},
            "thinking.client_extension",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "client_extension": True,
                    }
                ]
            },
            "messages\\[0\\].client_extension",
        ),
        (
            {
                "system": [
                    {
                        "type": "text",
                        "text": "System",
                        "cache_control": {"type": "ephemeral", "ttl": "2h"},
                    }
                ]
            },
            "system\\[0\\].cache_control.ttl",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello",
                                "cache_control": {"type": "persistent"},
                            }
                        ],
                    }
                ]
            },
            "cache_control.type",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.invalid/image.png",
                                    "caption": "silently dropped",
                                },
                            }
                        ],
                    }
                ]
            },
            "source.caption",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "file",
                                    "file_id": "file_1",
                                    "filename": "ignored.pdf",
                                },
                            }
                        ],
                    }
                ]
            },
            "source.filename",
        ),
        (
            {
                "tools": [
                    {
                        "name": "lookup",
                        "input_schema": {"type": "object"},
                        "client_extension": True,
                    }
                ]
            },
            "tools\\[0\\].client_extension",
        ),
    ],
)
def test_messages_ingress_rejects_active_or_unknown_nested_semantics(
    override: dict[str, object],
    match: str,
) -> None:
    payload: dict[str, object] = {
        "model": "model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(override)
    request = MessagesRequest.model_validate(payload)

    with pytest.raises(AnthropicIngressError, match=match):
        messages_to_inference_request(request)
