from free_claude_code.api.models.anthropic import (
    Message,
    MessagesRequest,
    TokenCountRequest,
)


def test_messages_request_parses_without_model_mapping_side_effects():
    request = MessagesRequest(
        model="claude-3-opus",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )

    assert request.model == "claude-3-opus"


def test_messages_request_normalizes_system_role_messages():
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

    assert [message.role for message in request.messages] == ["user", "user"]
    assert request.system == "system prompt"


def test_messages_request_merges_system_role_messages_with_existing_system():
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

    assert len(request.messages) == 1
    assert request.system == "existing system\n\nmessage system"


def test_messages_request_preserves_system_block_cache_control_when_normalizing():
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

    assert len(request.messages) == 1
    assert isinstance(request.system, list)
    assert [block.text for block in request.system] == [
        "existing system",
        "message system",
    ]
    assert request.system[0].model_dump()["cache_control"] == {"type": "ephemeral"}
    assert request.system[1].model_dump()["cache_control"] == {"type": "ephemeral"}


def test_messages_request_ignores_internal_routing_fields_when_supplied():
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
    assert "original_model" not in request.model_dump()
    assert "resolved_provider_model" not in request.model_dump()


def test_token_count_request_parses_without_model_mapping_side_effects():
    request = TokenCountRequest(
        model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
    )

    assert request.model == "claude-3-sonnet"


def test_token_count_request_normalizes_system_role_messages():
    request = TokenCountRequest.model_validate(
        {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "system", "content": "counting system"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    assert len(request.messages) == 1
    assert request.messages[0].role == "user"
    assert request.system == "counting system"


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
