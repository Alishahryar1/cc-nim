"""Tests for the CommandCode native provider request mapping."""

from unittest.mock import MagicMock

from providers.commandcode.request import _map_messages, build_request_body


def test_map_messages_text_only():
    """Test mapping standard text messages."""
    anthropic_msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
    ]

    cc_msgs = _map_messages(anthropic_msgs)

    assert len(cc_msgs) == 2
    assert cc_msgs[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
    }
    assert cc_msgs[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
    }


def test_map_messages_tool_use_and_result():
    """Test mapping tool-use and tool-result, specifically role transformation."""
    anthropic_msgs = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "bash",
                    "input": {"command": "ls"},
                }
            ],
        },
        {
            "role": "user",  # Anthropic uses role: user for tool results
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "file.txt",
                }
            ],
        },
    ]

    cc_msgs = _map_messages(anthropic_msgs)

    assert len(cc_msgs) == 2

    # Assistant tool call
    assert cc_msgs[0]["role"] == "assistant"
    assert len(cc_msgs[0]["content"]) == 1
    assert cc_msgs[0]["content"][0]["type"] == "tool-call"
    assert cc_msgs[0]["content"][0]["toolCallId"] == "toolu_123"
    assert cc_msgs[0]["content"][0]["toolName"] == "bash"
    assert cc_msgs[0]["content"][0]["input"] == {"command": "ls"}

    # User tool result MUST be mapped to role: tool
    assert cc_msgs[1]["role"] == "tool"
    assert len(cc_msgs[1]["content"]) == 1
    assert cc_msgs[1]["content"][0]["type"] == "tool-result"
    assert cc_msgs[1]["content"][0]["toolCallId"] == "toolu_123"
    assert (
        cc_msgs[1]["content"][0]["toolName"] == "bash"
    )  # Propagated from previous msg
    assert cc_msgs[1]["content"][0]["output"] == {"type": "text", "value": "file.txt"}


def test_map_messages_multiple_tool_results():
    """Test that multiple tool results in one Anthropic message map correctly to a single tool message."""
    anthropic_msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "cmd1", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "cmd2", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "res1"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "res2"},
            ],
        },
    ]

    cc_msgs = _map_messages(anthropic_msgs)

    assert cc_msgs[1]["role"] == "tool"
    assert len(cc_msgs[1]["content"]) == 2
    assert cc_msgs[1]["content"][0]["toolName"] == "cmd1"
    assert cc_msgs[1]["content"][0]["output"]["value"] == "res1"
    assert cc_msgs[1]["content"][1]["toolName"] == "cmd2"
    assert cc_msgs[1]["content"][1]["output"]["value"] == "res2"


def test_build_request_body_structure():
    """Test that the overarching wrapper body complies with CommandCode API expectations."""
    raw_req = MagicMock()
    raw_req.model = "deepseek-v4-pro"
    raw_req.system = "You are a helpful assistant."
    raw_req.max_tokens = 1024
    raw_req.temperature = 0.5
    raw_req.messages = [{"role": "user", "content": "Hi"}]
    raw_req.tools = [{"name": "Bash", "description": "Run command", "input_schema": {}}]

    cc_body = build_request_body(raw_req, thinking_enabled=False)

    assert "config" in cc_body
    assert "threadId" in cc_body
    assert "params" in cc_body

    params = cc_body["params"]
    assert params["model"] == "deepseek-v4-pro"
    assert params["system"] == "You are a helpful assistant."
    assert params["maxTokens"] == 1024
    assert params["temperature"] == 0.5
    assert params["stream"] is True
    assert len(params["tools"]) == 1
    assert params["tools"][0]["name"] == "Bash"
    assert len(params["messages"]) == 1
    assert params["messages"][0]["role"] == "user"
