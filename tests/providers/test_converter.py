import json

import pytest

from api.models.anthropic import MessagesRequest
from core.anthropic import (
    AnthropicToOpenAIConverter,
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)

# --- Mock Classes ---


class MockMessage:
    def __init__(self, role, content, reasoning_content=None):
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content


class MockBlock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._data = kwargs

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockTool:
    def __init__(self, name, description, input_schema=None):
        self.name = name
        self.description = description
        self.input_schema = input_schema


# --- System Prompt Tests ---


def test_convert_system_prompt_str():
    system = "You are a helpful assistant."
    result = AnthropicToOpenAIConverter.convert_system_prompt(system)
    assert result == {"role": "system", "content": system}


def test_convert_system_prompt_list_text():
    system = [
        MockBlock(type="text", text="Part 1"),
        MockBlock(type="text", text="Part 2"),
    ]
    result = AnthropicToOpenAIConverter.convert_system_prompt(system)
    assert result == {"role": "system", "content": "Part 1\n\nPart 2"}


def test_convert_system_prompt_none():
    assert AnthropicToOpenAIConverter.convert_system_prompt(None) is None


# --- Tool Conversion Tests ---


def test_convert_tools():
    tools = [
        MockTool(
            "get_weather",
            "Get weather",
            {"type": "object", "properties": {"loc": {"type": "string"}}},
        ),
        MockTool("calculator", None, {"type": "object"}),
    ]
    result = AnthropicToOpenAIConverter.convert_tools(tools)
    assert len(result) == 2

    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["description"] == "Get weather"
    assert result[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {"loc": {"type": "string"}},
    }

    assert result[1]["function"]["name"] == "calculator"
    assert result[1]["function"]["description"] == ""  # Check default empty string


def test_convert_tool_without_input_schema_uses_empty_object_schema():
    tools = [MockTool("web_search", None)]

    result = AnthropicToOpenAIConverter.convert_tools(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        (
            {"type": "tool", "name": "echo_smoke"},
            {"type": "function", "function": {"name": "echo_smoke"}},
        ),
        ({"type": "any"}, "required"),
        ({"type": "auto"}, "auto"),
        ({"type": "none"}, "none"),
        (
            {"type": "function", "function": {"name": "already_openai"}},
            {"type": "function", "function": {"name": "already_openai"}},
        ),
    ],
)
def test_convert_tool_choice(tool_choice, expected):
    result = AnthropicToOpenAIConverter.convert_tool_choice(tool_choice)
    assert result == expected


# --- Message Conversion Tests: User ---


def test_convert_user_message_str():
    messages = [MockMessage("user", "Hello world")]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Hello world"}


def test_convert_user_message_list_text():
    content = [
        MockBlock(type="text", text="Hello"),
        MockBlock(type="text", text="World"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "user", "content": "Hello\nWorld"}


def test_convert_user_message_tool_result_str():
    content = [
        MockBlock(type="tool_result", tool_use_id="tool_123", content="Result data")
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {
        "role": "tool",
        "tool_call_id": "tool_123",
        "content": "Result data",
    }


def test_convert_user_message_tool_result_list():
    # Tool result content as a list of text blocks
    tool_content = [
        {"type": "text", "text": "Line 1"},
        {"type": "text", "text": "Line 2"},
    ]
    content = [
        MockBlock(type="tool_result", tool_use_id="tool_456", content=tool_content)
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tool_456"
    assert result[0]["content"] == "Line 1\nLine 2"


def test_convert_user_message_mixed_text_and_tool_result():
    # Note: Anthropic/OpenAI mapping usually separates these, but the converter handles lists
    # User text usually comes before tool results in a turn, or after.
    # The converter splits them into separate messages if they are different roles?
    # Let's check logic: _convert_user_message returns a list of dicts.
    content = [
        MockBlock(type="text", text="Here is the result:"),
        MockBlock(type="tool_result", tool_use_id="tool_789", content="42"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    # Order is preserved: user text first, then tool result.
    assert len(result) == 2
    assert result[0] == {"role": "user", "content": "Here is the result:"}
    assert result[1] == {"role": "tool", "tool_call_id": "tool_789", "content": "42"}


# --- Message Conversion Tests: Assistant ---


def test_convert_assistant_message_text_only():
    messages = [MockMessage("assistant", "I am ready.")]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0] == {"role": "assistant", "content": "I am ready."}


def test_convert_assistant_message_blocks_text():
    content = [MockBlock(type="text", text="Part A")]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0] == {"role": "assistant", "content": "Part A"}


def test_convert_assistant_message_thinking():
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
    # Expecting <think> tags
    expected_content = (
        "<think>\nI need to calculate this.\n</think>\n\nThe answer is 4."
    )
    assert result[0]["content"] == expected_content
    assert "reasoning_content" not in result[0]


def test_convert_assistant_message_thinking_replays_reasoning_content():
    """Top-level reasoning replay avoids duplicating thinking into content."""
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert len(result) == 1
    assert result[0]["reasoning_content"] == "I need to calculate this."
    assert result[0]["content"] == "The answer is 4."
    assert "<think>" not in result[0]["content"]


def test_convert_assistant_top_level_reasoning_content_is_preserved():
    messages = [
        MockMessage(
            "assistant",
            "The answer is 4.",
            reasoning_content="I need to calculate this.",
        )
    ]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "I need to calculate this.",
        }
    ]


def test_convert_assistant_thinking_tool_use_replays_top_level_reasoning():
    content = [
        MockBlock(type="thinking", thinking="I should call the tool."),
        MockBlock(
            type="tool_use",
            id="call_reasoning",
            name="search",
            input={"query": "python"},
        ),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )

    assert len(result) == 1
    assert result[0]["content"] == ""
    assert result[0]["reasoning_content"] == "I should call the tool."
    assert "<think>" not in result[0]["content"]
    assert result[0]["tool_calls"][0]["id"] == "call_reasoning"


def test_convert_assistant_message_thinking_removed_when_disabled():
    content = [
        MockBlock(type="thinking", thinking="I need to calculate this."),
        MockBlock(type="text", text="The answer is 4."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages,
        reasoning_replay=ReasoningReplayMode.DISABLED,
    )

    assert len(result) == 1
    assert "reasoning_content" not in result[0]
    assert "<think>" not in result[0]["content"]
    assert result[0]["content"] == "The answer is 4."


def test_convert_assistant_top_level_reasoning_stripped_when_disabled():
    messages = [
        MockMessage(
            "assistant",
            "The answer is 4.",
            reasoning_content="I need to calculate this.",
        )
    ]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.DISABLED
    )

    assert result == [{"role": "assistant", "content": "The answer is 4."}]


def test_convert_assistant_message_tool_use():
    content = [
        MockBlock(type="text", text="I will call the tool."),
        MockBlock(
            type="tool_use", id="call_1", name="search", input={"query": "python"}
        ),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
    msg = result[0]
    assert msg["role"] == "assistant"
    assert "I will call the tool." in msg["content"]
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"]) == {"query": "python"}


def test_convert_assistant_tool_use_preserves_extra_content():
    content = [
        MockBlock(
            type="tool_use",
            id="call_1",
            name="search",
            input={"query": "python"},
            extra_content={"google": {"thought_signature": "sig"}},
        ),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert result[0]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "sig"}
    }


def test_convert_assistant_message_empty_content():
    # Verify that empty content becomes a single space (NIM requirement)
    # if no tool calls are present.
    content = []
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["content"] == " "


def test_convert_assistant_message_tool_use_no_text():
    # If tool usage exists, content can be empty string?
    # Logic: if not content_str and not tool_calls: content_str = " "
    # So if tool_calls exist, content_str can be empty string?
    # Actually code says: if not content_str and not tool_calls.
    # So if tool_calls is present, content_str remains "" (empty).

    content = [MockBlock(type="tool_use", id="call_2", name="test", input={})]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert (
        result[0]["content"] == ""
    )  # Should be empty string, not space, because tools exist
    assert len(result[0]["tool_calls"]) == 1


def test_convert_mixed_blocks_and_types_and_roles():
    # comprehensive flow
    messages = [
        MockMessage("user", "Start"),
        MockMessage(
            "assistant",
            [
                MockBlock(type="thinking", thinking="Thinking..."),
                MockBlock(type="text", text="Here is a tool."),
            ],
        ),
        MockMessage(
            "assistant", [MockBlock(type="tool_use", id="t1", name="f", input={})]
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert "<think>" in result[1]["content"]
    assert result[2]["tool_calls"][0]["id"] == "t1"


# --- Edge Cases ---


def test_get_block_attr_defaults():
    # Test helper directly
    from core.anthropic import get_block_attr

    assert get_block_attr({}, "missing", "default") == "default"
    assert get_block_attr(object(), "missing", "default") == "default"


def test_input_not_dict():
    # Tool input might not be a dict (e.g. malformed or string)
    content = [MockBlock(type="tool_use", id="call_x", name="f", input="some_string")]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    # The converter calls json.dumps(tool_input) if dict, else str(tool_input)
    # So it should be "some_string"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == "some_string"


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "system_input,expected",
    [
        ("You are helpful.", {"role": "system", "content": "You are helpful."}),
        (
            [MockBlock(type="text", text="A"), MockBlock(type="text", text="B")],
            {"role": "system", "content": "A\n\nB"},
        ),
        (None, None),
        ("", {"role": "system", "content": ""}),
        ([], None),
    ],
    ids=["string", "list_text", "none", "empty_string", "empty_list"],
)
def test_convert_system_prompt_parametrized(system_input, expected):
    """Parametrized system prompt conversion covering edge cases."""
    result = AnthropicToOpenAIConverter.convert_system_prompt(system_input)
    assert result == expected


@pytest.mark.parametrize(
    "content,expected_content",
    [
        ("Hello world", "Hello world"),
        ("", ""),
        ([MockBlock(type="text", text="A"), MockBlock(type="text", text="B")], "A\nB"),
        ([MockBlock(type="text", text="")], ""),
    ],
    ids=["simple_string", "empty_string", "list_blocks", "empty_text_block"],
)
def test_convert_user_message_parametrized(content, expected_content):
    """Parametrized user message conversion."""
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) >= 1
    assert result[0]["content"] == expected_content


def test_convert_assistant_message_unknown_block_type():
    """Unknown block types should be silently skipped."""
    content = [
        MockBlock(type="unknown_type", data="something"),
        MockBlock(type="text", text="visible"),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert "visible" in result[0]["content"]


def test_convert_tool_use_none_input():
    """Tool use with None input should not crash."""
    content = [MockBlock(type="tool_use", id="call_n", name="test", input=None)]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert "tool_calls" in result[0]


def test_convert_assistant_interleaved_order_preserved():
    """Interleaved thinking, text, tool_use should preserve thinking+text order in content.

    Bug: Current implementation collects all thinking, then all text, then tool_calls.
    Original order [thinking, text, thinking, tool_use] becomes [all thinking, all text, tool_calls],
    losing the interleaving. Content string should reflect original block order for thinking+text.
    Tool calls stay at end (API constraint).
    """
    content = [
        MockBlock(type="thinking", thinking="First thought."),
        MockBlock(type="text", text="Here is the answer."),
        MockBlock(type="thinking", thinking="Second thought."),
        MockBlock(type="tool_use", id="call_1", name="search", input={"q": "x"}),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 1
    msg = result[0]
    # Expected: thinking1, text, thinking2 in that order within content; tool_calls at end
    expected_content = "<think>\nFirst thought.\n</think>\n\nHere is the answer.\n\n<think>\nSecond thought.\n</think>"
    assert msg["content"] == expected_content, (
        f"Interleaved order lost. Got: {msg['content']!r}"
    )
    assert len(msg["tool_calls"]) == 1


def test_convert_user_message_text_before_tool_result_order():
    """User message with text then tool_result should preserve order: user text first, then tool.

    Bug: Current implementation emits tool_result immediately, then user text at end.
    Anthropic order is typically: user says something, then provides tool results.
    """
    content = [
        MockBlock(type="text", text="Please use this result:"),
        MockBlock(type="tool_result", tool_use_id="t1", content="42"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)

    assert len(result) == 2
    # Expected: user text first, then tool result
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Please use this result:"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == "t1"


def test_convert_multiple_tool_results():
    """Multiple tool results in a single user message."""
    content = [
        MockBlock(type="tool_result", tool_use_id="t1", content="Result 1"),
        MockBlock(type="tool_result", tool_use_id="t2", content="Result 2"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 2
    assert result[0]["tool_call_id"] == "t1"
    assert result[1]["tool_call_id"] == "t2"


def test_convert_user_message_tool_result_dict_as_json():
    content = [
        MockBlock(
            type="tool_result",
            tool_use_id="t_dict",
            content={"ok": True, "count": 2},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["role"] == "tool"
    assert result[0]["content"] == '{"ok": true, "count": 2}'


def test_assistant_redacted_thinking_omitted_from_openai_chat():
    """Opaque redacted_thinking is not materialized as content or reasoning_content."""
    content = [
        MockBlock(type="redacted_thinking", data="secret-opaque"),
        MockBlock(type="text", text="Visible."),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(
        messages, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )
    assert result[0]["content"] == "Visible."
    assert "secret-opaque" not in result[0]["content"]
    assert "reasoning_content" not in result[0]


def test_convert_user_image_base64():
    """Base64-encoded image converts to a data: URI in OpenAI image_url format."""
    content = [
        MockBlock(type="text", text="Describe this image"),
        MockBlock(
            type="image",
            source={"type": "base64", "media_type": "image/png", "data": "iVBOR"},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    parts = result[0]["content"]
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "Describe this image"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBOR"},
    }


def test_convert_user_image_url():
    """URL-based image passes the URL through directly."""
    content = [
        MockBlock(type="text", text="What is this?"),
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/cat.jpg"},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    parts = result[0]["content"]
    assert isinstance(parts, list)
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.jpg"},
    }


def test_convert_user_image_only_no_text():
    """Image block without any text produces a content array with just the image."""
    content = [
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/img.png"},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    parts = result[0]["content"]
    assert isinstance(parts, list)
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"


def test_convert_user_mixed_text_and_images():
    """Multiple text and image blocks produce a single multipart content array."""
    content = [
        MockBlock(type="text", text="First"),
        MockBlock(type="text", text="Second"),
        MockBlock(
            type="image",
            source={"type": "base64", "media_type": "image/jpeg", "data": "abc"},
        ),
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/b.png"},
        ),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    parts = result[0]["content"]
    assert isinstance(parts, list)
    assert len(parts) == 3  # 1 merged text + 2 images
    assert parts[0] == {"type": "text", "text": "First\nSecond"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,abc"
    assert parts[2]["image_url"]["url"] == "https://example.com/b.png"


def test_convert_user_image_unknown_source_type_raises():
    """Unknown image source types still raise an error."""
    content = [
        MockBlock(
            type="image",
            source={"type": "s3", "bucket": "b", "key": "k"},
        ),
    ]
    messages = [MockMessage("user", content)]
    with pytest.raises(OpenAIConversionError, match="Unsupported image source type"):
        AnthropicToOpenAIConverter.convert_messages(messages)


def test_convert_user_text_only_remains_flat_string():
    """Text-only messages keep the flat string format (backward compat)."""
    content = [
        MockBlock(type="text", text="Hello"),
        MockBlock(type="text", text="World"),
    ]
    messages = [MockMessage("user", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 1
    assert result[0]["content"] == "Hello\nWorld"
    assert isinstance(result[0]["content"], str)


def test_convert_user_image_in_injection_path():
    """Images in _convert_user_message_with_injection convert instead of raising."""
    from core.anthropic.conversion import _PendingAfterTools

    content = [
        MockBlock(type="text", text="See this:"),
        MockBlock(
            type="image",
            source={"type": "url", "url": "https://example.com/x.png"},
        ),
        MockBlock(type="tool_result", tool_use_id="call_z", content="ok"),
    ]
    pending = _PendingAfterTools(
        remaining_tool_ids={"call_z"},
        deferred_blocks=[MockBlock(type="text", text="Deferred")],
    )
    pieces = AnthropicToOpenAIConverter._convert_user_message_with_injection(
        content, pending
    )
    msgs = pieces["messages"]
    # First message: text + image as multipart
    assert msgs[0]["role"] == "user"
    assert isinstance(msgs[0]["content"], list)
    assert msgs[0]["content"][0] == {"type": "text", "text": "See this:"}
    assert msgs[0]["content"][1]["type"] == "image_url"
    # Second message: tool result
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_z"


def test_convert_assistant_image_block_still_raises():
    """Assistant-role image blocks remain unsupported."""
    content = [MockBlock(type="image", source={"type": "url", "url": "https://x"})]
    messages = [MockMessage("assistant", content)]
    with pytest.raises(OpenAIConversionError, match="Assistant image"):
        AnthropicToOpenAIConverter.convert_messages(messages)


def test_convert_assistant_text_after_tool_use_splits_for_openai_chat():
    """Post-tool_use assistant text is replayed as a second assistant turn (issue 206)."""
    content = [
        MockBlock(type="tool_use", id="call_z", name="Read", input={}),
        MockBlock(type="text", text="After tool"),
    ]
    messages = [MockMessage("assistant", content)]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[0]["tool_calls"][0]["id"] == "call_z"
    assert result[1] == {"role": "assistant", "content": "After tool"}


def test_convert_assistant_text_after_tool_use_inserts_after_tool_results():
    messages = [
        MockMessage(
            "assistant",
            [
                MockBlock(type="tool_use", id="call_z", name="Read", input={}),
                MockBlock(type="text", text="Post-tool commentary"),
            ],
        ),
        MockMessage(
            "user",
            [
                MockBlock(
                    type="tool_result",
                    tool_use_id="call_z",
                    content="file contents",
                )
            ],
        ),
    ]
    result = AnthropicToOpenAIConverter.convert_messages(messages)
    assert result[0]["role"] == "assistant" and "tool_calls" in result[0]
    assert result[1]["role"] == "tool" and result[1]["tool_call_id"] == "call_z"
    assert result[2] == {"role": "assistant", "content": "Post-tool commentary"}


def test_openai_build_accepts_declared_native_top_level_hints() -> None:
    """OpenAI conversion ignores known non-OpenAI hints (e.g. context_management) without 400."""
    req = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "h"}],
            "context_management": {"edits": []},
            "output_config": {"foo": 1},
            "mcp_servers": [{"type": "url", "url": "https://x.com"}],
        }
    )
    body = build_base_request_body(req, default_max_tokens=100)
    assert "context_management" not in body
    assert "output_config" not in body
    assert "mcp_servers" not in body
    assert body["model"] == "m"


def test_openai_build_rejects_unknown_top_level_extras() -> None:
    """Truly unknown keys must still be rejected (not dropped silently)."""
    req = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "h"}],
            "experimental_client_only_passthrough": True,
        }
    )
    with pytest.raises(OpenAIConversionError, match="top-level request fields"):
        build_base_request_body(req)


@pytest.mark.parametrize(
    "content",
    [
        [MockBlock(type="server_tool_use", id="1", name="web_search", input={})],
        [MockBlock(type="web_search_tool_result", tool_use_id="1", content=[])],
        [
            MockBlock(
                type="web_fetch_tool_result",
                tool_use_id="1",
                content={"type": "web_fetch_result", "url": "https://a.com/x"},
            )
        ],
    ],
)
def test_convert_assistant_server_tool_blocks_raise(content) -> None:
    messages = [MockMessage("assistant", content)]
    with pytest.raises(OpenAIConversionError, match="server tool"):
        AnthropicToOpenAIConverter.convert_messages(messages)
