"""Request utility functions for API route handlers.

This module contains optimization functions, quota detection, title generation detection,
prefix detection, and token counting utilities.
"""

import json
import logging
from typing import List, Optional, Tuple, Union

import tiktoken

from .models import MessagesRequest

logger = logging.getLogger(__name__)
ENCODER = tiktoken.get_encoding("cl100k_base")


def is_quota_check_request(request_data: MessagesRequest) -> bool:
    """Check if this is a quota probe request.

    Quota checks are typically simple requests with max_tokens=1
    and a single message containing the word "quota".

    Args:
        request_data: The incoming request data

    Returns:
        True if this is a quota probe request
    """
    if (
        request_data.max_tokens == 1
        and len(request_data.messages) == 1
        and request_data.messages[0].role == "user"
    ):
        content = request_data.messages[0].content
        # Check string content
        if isinstance(content, str) and "quota" in content.lower():
            return True
        # Check list content
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "text") and "quota" in block.text.lower():
                    return True
    return False


def is_title_generation_request(request_data: MessagesRequest) -> bool:
    """Check if this is a conversation title generation request.

    Title generation requests typically contain the phrase
    "write a 5-10 word title" in the user's message.

    Args:
        request_data: The incoming request data

    Returns:
        True if this is a title generation request
    """
    if len(request_data.messages) > 0 and request_data.messages[-1].role == "user":
        content = request_data.messages[-1].content
        # Check string content
        target_phrase = "write a 5-10 word title"
        if isinstance(content, str) and target_phrase in content.lower():
            return True
        # Check list content
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "text") and target_phrase in block.text.lower():
                    return True
    return False


def extract_command_prefix(command: str) -> str:
    """Extract the command prefix for fast prefix detection.

    Parses a shell command safely, handling environment variables and
    command injection attempts. Returns the command prefix suitable
    for quick identification.

    Args:
        command: The command string to analyze

    Returns:
        Command prefix (e.g., "git", "git commit", "npm install")
        or "none" if no valid command found
    """
    import shlex

    # Quick check for command injection patterns
    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command)
        if not parts:
            return "none"

        # Handle environment variable prefixes (e.g., KEY=value command)
        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if "=" in part and not part.startswith("-"):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git",
            "npm",
            "docker",
            "kubectl",
            "cargo",
            "go",
            "pip",
            "yarn",
        }

        # For compound commands, include the subcommand (e.g., "git commit")
        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        return first_word if not env_prefix else " ".join(env_prefix) + " " + first_word

    except ValueError:
        # Fall back to simple split if shlex fails
        return command.split()[0] if command.split() else "none"


def is_prefix_detection_request(request_data: MessagesRequest) -> Tuple[bool, str]:
    """Check if this is a fast prefix detection request.

    Prefix detection requests contain a policy_spec block and
    a Command: section for extracting shell command prefixes.

    Args:
        request_data: The incoming request data

    Returns:
        Tuple of (is_prefix_request, command_string)
    """
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, ""

    msg = request_data.messages[0]
    content = ""
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            if hasattr(block, "text"):
                content += block.text

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass

    return False, ""


def get_token_count(
    messages: List,
    system: Optional[Union[str, List]] = None,
    tools: Optional[List] = None,
) -> int:
    """Estimate token count for a request.

    Uses tiktoken cl100k_base encoding to estimate token usage.
    Includes system prompt, messages, tools, and per-message overhead.

    Args:
        messages: List of message objects with content
        system: Optional system prompt (str or list of blocks)
        tools: Optional list of tool definitions

    Returns:
        Estimated total token count
    """
    total_tokens = 0

    # Count system prompt tokens
    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += len(ENCODER.encode(block.text))

    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = getattr(block, "type", None)

                if b_type == "text":
                    total_tokens += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    total_tokens += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    total_tokens += len(ENCODER.encode(name))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += 10  # Tool use overhead
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += 5  # Tool result overhead

    # Count tool definition tokens
    if tools:
        for tool in tools:
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    # Add per-message overhead
    total_tokens += len(messages) * 3
    if tools:
        total_tokens += len(tools) * 5

    return max(1, total_tokens)
