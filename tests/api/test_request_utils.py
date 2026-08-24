"""Tests for API request detection and token counting helpers."""

from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.api.command_utils import extract_command_prefix
from free_claude_code.api.detection import (
    is_prefix_detection_request,
    is_quota_check_request,
    is_title_generation_request,
)
from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
)
from free_claude_code.core.inference import (
    Base64MediaSource,
    FunctionTool,
    ImageContent,
    InferenceItem,
    InferenceRequest,
    InstructionItem,
    InstructionOrigin,
    InstructionPlacement,
    MessageItem,
    MessageRole,
    ReasoningItem,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolResultItem,
    get_inference_token_count,
)


class TestQuotaCheckRequest:
    """Tests for is_quota_check_request function."""

    def test_quota_check_simple_string(self):
        """Test quota check with simple string content."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_quota_check_case_insensitive(self):
        """Test quota check is case insensitive."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my QUOTA"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_quota_check_list_content(self):
        """Test quota check with list content blocks."""
        block = MagicMock()
        block.text = "Check my quota"

        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = [block]

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is True

    def test_not_quota_check_wrong_max_tokens(self):
        """Test not quota check when max_tokens != 1."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 100
        req.messages = [msg]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_multiple_messages(self):
        """Test not quota check when multiple messages."""
        msg1 = MagicMock(spec=Message)
        msg1.role = "user"
        msg1.content = "Check my quota"

        msg2 = MagicMock(spec=Message)
        msg2.role = "assistant"
        msg2.content = "Hello"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg1, msg2]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_wrong_role(self):
        """Test not quota check when role is not user."""
        msg = MagicMock(spec=Message)
        msg.role = "assistant"
        msg.content = "Check my quota"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is False

    def test_not_quota_check_no_quota_keyword(self):
        """Test not quota check when content doesn't contain quota."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Hello world"

        req = MagicMock(spec=MessagesRequest)
        req.max_tokens = 1
        req.messages = [msg]

        assert is_quota_check_request(req) is False


class TestTitleGenerationRequest:
    """Tests for is_title_generation_request function."""

    def _title_gen_system(self) -> list[MagicMock]:
        block = MagicMock()
        block.text = (
            "Generate a concise, sentence-case title (3-7 words) that captures the "
            "main topic or goal of this coding session. Return JSON with a single "
            '"title" field.'
        )
        return [block]

    def test_title_generation_detected_via_system(self):
        """Title gen detected by session title system prompt (sentence-case / JSON)."""
        req = MagicMock(spec=MessagesRequest)
        req.system = self._title_gen_system()
        req.tools = None
        req.messages = []

        assert is_title_generation_request(req) is True

    def test_title_generation_not_detected_with_tools(self):
        """Not detected when tools are present (main conversation, not title gen)."""
        req = MagicMock(spec=MessagesRequest)
        req.system = self._title_gen_system()
        req.tools = [MagicMock()]
        req.messages = []

        assert is_title_generation_request(req) is False

    def test_title_generation_not_detected_no_system(self):
        """Not detected when system is absent."""
        req = MagicMock(spec=MessagesRequest)
        req.system = None
        req.tools = None
        req.messages = []

        assert is_title_generation_request(req) is False

    def test_title_generation_not_detected_unrelated_system(self):
        """Not detected when system prompt has no topic/title keywords."""
        block = MagicMock()
        block.text = "You are a helpful assistant."
        req = MagicMock(spec=MessagesRequest)
        req.system = [block]
        req.tools = None
        req.messages = []

        assert is_title_generation_request(req) is False

    def test_title_generation_return_json_coding_session_branch(self):
        """JSON title field + session wording matches without sentence-case phrase."""
        block = MagicMock()
        block.text = 'Return JSON with a single "title" field for this coding session.'
        req = MagicMock(spec=MessagesRequest)
        req.system = [block]
        req.tools = None
        req.messages = []

        assert is_title_generation_request(req) is True


class TestExtractCommandPrefix:
    """Tests for extract_command_prefix function."""

    def test_simple_command(self):
        """Test extraction of simple command."""
        assert extract_command_prefix("git status") == "git status"
        assert extract_command_prefix("ls -la") == "ls"

    def test_two_word_commands(self):
        """Test extraction of two-word commands."""
        assert extract_command_prefix("git commit -m 'message'") == "git commit"
        assert extract_command_prefix("npm install package") == "npm install"
        assert extract_command_prefix("docker run image") == "docker run"
        assert extract_command_prefix("kubectl get pods") == "kubectl get"

    def test_two_word_command_with_options(self):
        """Test two-word command with options only returns first word."""
        assert extract_command_prefix("git -v") == "git"
        assert extract_command_prefix("npm --version") == "npm"

    def test_with_env_vars(self):
        """Test command with environment variables."""
        assert extract_command_prefix("DEBUG=1 python script.py") == "DEBUG=1 python"
        assert (
            extract_command_prefix("API_KEY=secret node app.js")
            == "API_KEY=secret node"
        )

    def test_single_word_commands(self):
        """Test single word commands."""
        assert extract_command_prefix("ls") == "ls"
        assert extract_command_prefix("python") == "python"
        assert extract_command_prefix("make") == "make"

    def test_command_injection_detected(self):
        """Test detection of command injection attempts."""
        assert extract_command_prefix("`whoami`") == "command_injection_detected"
        assert extract_command_prefix("$(whoami)") == "command_injection_detected"
        assert (
            extract_command_prefix("echo $(cat /etc/passwd)")
            == "command_injection_detected"
        )

    def test_empty_command(self):
        """Test handling of empty commands."""
        assert extract_command_prefix("") == "none"
        assert extract_command_prefix("   ") == "none"

    def test_complex_git_command(self):
        """Test complex git command extraction."""
        assert extract_command_prefix("git log --oneline --graph") == "git log"
        assert (
            extract_command_prefix("git checkout -b feature-branch") == "git checkout"
        )

    def test_cargo_command(self):
        """Test cargo command extraction."""
        assert extract_command_prefix("cargo build") == "cargo build"
        assert extract_command_prefix("cargo test") == "cargo test"
        assert extract_command_prefix("cargo --version") == "cargo"


class TestPrefixDetectionRequest:
    """Tests for is_prefix_detection_request function."""

    def test_prefix_detection_with_policy_spec(self):
        """Test prefix detection with policy spec and command."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "<policy_spec>policy</policy_spec> Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is True
        assert command == "git status"

    def test_prefix_detection_case_sensitive(self):
        """Test prefix detection is case sensitive for Command:."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "<policy_spec>policy</policy_spec> command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_no_policy_spec(self):
        """Test not prefix detection without policy_spec."""
        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = "Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_multiple_messages(self):
        """Test not prefix detection with multiple messages."""
        msg1 = MagicMock(spec=Message)
        msg1.role = "user"
        msg1.content = "<policy_spec>policy</policy_spec> Command: git status"

        msg2 = MagicMock(spec=Message)
        msg2.role = "assistant"
        msg2.content = "OK"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg1, msg2]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_not_prefix_detection_wrong_role(self):
        """Test not prefix detection when message is not from user."""
        msg = MagicMock(spec=Message)
        msg.role = "assistant"
        msg.content = "<policy_spec>policy</policy_spec> Command: git status"

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is False
        assert command == ""

    def test_prefix_detection_list_content(self):
        """Test prefix detection with list content blocks."""
        block = MagicMock()
        block.text = "<policy_spec>policy</policy_spec> Command: ls -la"

        msg = MagicMock(spec=Message)
        msg.role = "user"
        msg.content = [block]

        req = MagicMock(spec=MessagesRequest)
        req.messages = [msg]

        is_prefix, command = is_prefix_detection_request(req)
        assert is_prefix is True
        assert command == "ls -la"


def _message(turn_id: str, text: str) -> MessageItem:
    return MessageItem(
        turn_id=turn_id,
        role=MessageRole.USER,
        content=(TextContent(text),),
    )


def _token_request(
    *items: InferenceItem,
    tools: tuple[FunctionTool, ...] = (),
) -> InferenceRequest:
    return InferenceRequest(model="provider/model", items=items, tools=tools)


class TestGetInferenceTokenCount:
    """Canonical token accounting uses the same transcript as execution."""

    def test_empty_request_has_nonzero_floor(self) -> None:
        assert get_inference_token_count(_token_request()) == 1

    def test_plain_and_special_token_text_are_counted(self) -> None:
        ordinary = get_inference_token_count(_token_request(_message("turn_0", "Hi")))
        special = get_inference_token_count(
            _token_request(_message("turn_0", "<|endoftext|>"))
        )

        assert ordinary > 0
        assert special > 0

    def test_top_level_instruction_contributes_content_and_overhead(self) -> None:
        message = _message("turn_0", "Hello")
        without_instruction = get_inference_token_count(_token_request(message))
        with_instruction = get_inference_token_count(
            _token_request(
                InstructionItem(
                    text="You are a helpful assistant.",
                    origin=InstructionOrigin.SYSTEM,
                    placement=InstructionPlacement.TOP_LEVEL,
                ),
                message,
            )
        )

        assert with_instruction >= without_instruction + 4

    def test_transcript_instruction_contributes_to_same_turn(self) -> None:
        message = _message("turn_0", "Hello")
        instruction = InstructionItem(
            text="New instructions",
            origin=InstructionOrigin.SYSTEM,
            placement=InstructionPlacement.TRANSCRIPT,
            turn_id="turn_1",
        )

        assert get_inference_token_count(
            _token_request(message, instruction)
        ) > get_inference_token_count(_token_request(message))

    def test_reasoning_tool_call_and_tool_result_are_counted(self) -> None:
        request = _token_request(
            _message("turn_0", "Use the search tool"),
            ReasoningItem(turn_id="turn_1", reasoning="Need current information"),
            ToolCallItem(
                turn_id="turn_1",
                call_id="call_abc123",
                kind=ToolCallKind.FUNCTION,
                name="search",
                input={"query": "test"},
            ),
            ToolResultItem(
                turn_id="turn_2",
                call_id="call_abc123",
                content={"result": "data"},
            ),
        )

        assert get_inference_token_count(request) > 0

    def test_tool_definition_contributes_to_count(self) -> None:
        message = _message("turn_0", "Use the search tool")
        without_tool = get_inference_token_count(_token_request(message))
        with_tool = get_inference_token_count(
            _token_request(
                message,
                tools=(
                    FunctionTool(
                        name="search",
                        description="Search for information",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ),
            )
        )

        assert with_tool > without_tool

    def test_multiple_messages_include_per_turn_overhead(self) -> None:
        first = _message("turn_0", "Hi")
        second = _message("turn_1", "Hello")

        assert get_inference_token_count(
            _token_request(first, second)
        ) > get_inference_token_count(_token_request(first))

    def test_base64_image_uses_media_floor(self) -> None:
        request = _token_request(
            MessageItem(
                turn_id="turn_0",
                role=MessageRole.USER,
                content=(
                    ImageContent(
                        Base64MediaSource(
                            media_type="image/png",
                            data="x" * 3_000,
                        )
                    ),
                ),
            )
        )

        assert get_inference_token_count(request) >= 85

    def test_known_payload_structural_overhead(self) -> None:
        system_text = "You are a helpful assistant."
        user_text = "Hello, how are you?"
        request = _token_request(
            InstructionItem(
                text=system_text,
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TOP_LEVEL,
            ),
            _message("turn_0", user_text),
        )
        with patch(
            "free_claude_code.core.inference.tokens.estimate_text_tokens",
            side_effect=len,
        ):
            count = get_inference_token_count(request)

        assert count == len(system_text) + len(user_text) + 4 + 4


# --- Parametrized Edge Case Tests ---


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git status", "git status"),
        ("ls -la", "ls"),
        ("git commit -m 'msg'", "git commit"),
        ("npm install pkg", "npm install"),
        ("ls", "ls"),
        ("python", "python"),
        ("", "none"),
        ("   ", "none"),
        ("`whoami`", "command_injection_detected"),
        ("$(whoami)", "command_injection_detected"),
        ("echo $(cat /etc/passwd)", "command_injection_detected"),
        ("git -v", "git"),
        ("DEBUG=1 python script.py", "DEBUG=1 python"),
        ("cargo build", "cargo build"),
        ("cargo --version", "cargo"),
    ],
    ids=[
        "git_status",
        "ls_with_flag",
        "git_commit",
        "npm_install",
        "bare_ls",
        "bare_python",
        "empty",
        "whitespace",
        "injection_backtick",
        "injection_dollar",
        "injection_echo",
        "git_flag",
        "env_var",
        "cargo_build",
        "cargo_flag",
    ],
)
def test_extract_command_prefix_parametrized(command, expected):
    """Parametrized command prefix extraction."""
    assert extract_command_prefix(command) == expected


def test_extract_command_prefix_unterminated_quote():
    """Unterminated quote falls back to simple split (shlex.split ValueError)."""
    result = extract_command_prefix("git commit -m 'unterminated")
    # Should fall back to command.split()[0] = "git"
    assert result == "git"


def test_extract_command_prefix_pipe():
    """Piped commands - shlex handles pipe character."""
    result = extract_command_prefix("cat file.txt | grep pattern")
    assert result in ("cat", "cat file.txt")


@pytest.mark.parametrize(
    "content,max_tokens,role,expected",
    [
        ("Check my quota", 1, "user", True),
        ("Check my QUOTA", 1, "user", True),
        ("Hello world", 1, "user", False),
        ("Check my quota", 100, "user", False),
        ("Check my quota", 1, "assistant", False),
    ],
    ids=["basic", "case_insensitive", "no_keyword", "wrong_max_tokens", "wrong_role"],
)
def test_quota_check_parametrized(content, max_tokens, role, expected):
    """Parametrized quota check request detection."""
    msg = MagicMock(spec=Message)
    msg.role = role
    msg.content = content

    req = MagicMock(spec=MessagesRequest)
    req.max_tokens = max_tokens
    req.messages = [msg]

    assert is_quota_check_request(req) is expected


def test_quota_check_empty_messages():
    """Quota check with empty message list should not crash."""
    req = MagicMock(spec=MessagesRequest)
    req.max_tokens = 1
    req.messages = []
    assert is_quota_check_request(req) is False
