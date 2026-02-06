"""Request utility functions for API route handlers.

This module contains optimization functions, quota detection, title generation detection,
prefix detection, and token counting utilities.
"""

import json
import logging
from typing import List, Optional, Tuple, Union

import tiktoken

from .models.anthropic import MessagesRequest
from utils.text import extract_text_from_content

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
        text = extract_text_from_content(request_data.messages[0].content)
        if "quota" in text.lower():
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
        text = extract_text_from_content(request_data.messages[-1].content)
        if "write a 5-10 word title" in text.lower():
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

    content = extract_text_from_content(request_data.messages[0].content)

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass

    return False, ""


def is_suggestion_mode_request(request_data: MessagesRequest) -> bool:
    """Check if this is a suggestion mode request.

    Suggestion mode requests contain "[SUGGESTION MODE:" in the user's message,
    used for auto-suggesting what the user might type next.

    Args:
        request_data: The incoming request data

    Returns:
        True if this is a suggestion mode request
    """
    for msg in request_data.messages:
        if msg.role == "user":
            text = extract_text_from_content(msg.content)
            if "[SUGGESTION MODE:" in text:
                return True
    return False


def is_filepath_extraction_request(
    request_data: MessagesRequest,
) -> Tuple[bool, str, str]:
    """Check if this is a filepath extraction request.

    Filepath extraction requests have a single user message with
    "Command:" and "Output:" sections, asking to extract file paths
    from command output.

    Args:
        request_data: The incoming request data

    Returns:
        Tuple of (is_filepath_request, command, output)
    """
    # Must be single message, no tools
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, "", ""
    if request_data.tools:
        return False, "", ""

    content = extract_text_from_content(request_data.messages[0].content)

    # Must have Command: and Output: markers
    if "Command:" not in content or "Output:" not in content:
        return False, "", ""

    # Must ask for filepath extraction
    if "filepaths" not in content.lower() and "<filepaths>" not in content.lower():
        return False, "", ""

    try:
        # Extract command and output
        cmd_start = content.find("Command:") + len("Command:")
        output_marker = content.find("Output:", cmd_start)
        if output_marker == -1:
            return False, "", ""

        command = content[cmd_start:output_marker].strip()
        output = content[output_marker + len("Output:") :].strip()

        # Clean up output - stop at next section marker if present
        for marker in ["<", "\n\n"]:
            if marker in output:
                output = output.split(marker)[0].strip()

        return True, command, output
    except Exception:
        return False, "", ""


def extract_filepaths_from_command(command: str, output: str) -> str:
    """Extract file paths from a command locally without API call.

    Determines if the command reads file contents and extracts paths accordingly.
    Commands like ls/dir/find just list files, so return empty.
    Commands like cat/head/tail actually read contents, so extract the file path.

    Args:
        command: The shell command that was executed
        output: The command's output

    Returns:
        Filepath extraction result in <filepaths> format
    """
    import shlex

    # Commands that just list files (don't read contents)
    listing_commands = {
        "ls",
        "dir",
        "find",
        "tree",
        "pwd",
        "cd",
        "mkdir",
        "rmdir",
        "rm",
    }

    # Commands that read file contents
    reading_commands = {"cat", "head", "tail", "less", "more", "bat", "type"}

    try:
        parts = shlex.split(command)
        if not parts:
            return "<filepaths>\n</filepaths>"

        # Get base command (handle paths like /bin/cat)
        base_cmd = parts[0].split("/")[-1].split("\\")[-1].lower()

        # Listing commands - return empty
        if base_cmd in listing_commands:
            return "<filepaths>\n</filepaths>"

        # Reading commands - extract file arguments
        if base_cmd in reading_commands:
            filepaths = []
            for part in parts[1:]:
                # Skip flags
                if part.startswith("-"):
                    continue
                # This is likely a file path
                filepaths.append(part)

            if filepaths:
                paths_str = "\n".join(filepaths)
                return f"<filepaths>\n{paths_str}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        # grep with file argument
        if base_cmd == "grep":
            filepaths = []
            skip_next = False
            for i, part in enumerate(parts[1:], 1):
                if skip_next:
                    skip_next = False
                    continue
                # Skip flags and their arguments
                if part.startswith("-"):
                    # Flags that take an argument
                    if part in {"-e", "-f", "-m", "-A", "-B", "-C"}:
                        skip_next = True
                    continue
                # First non-flag is pattern, rest are files
                if i > 1:  # Skip the pattern
                    filepaths.append(part)

            if filepaths:
                paths_str = "\n".join(filepaths)
                return f"<filepaths>\n{paths_str}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        # Default - return empty for unknown commands
        return "<filepaths>\n</filepaths>"

    except Exception:
        return "<filepaths>\n</filepaths>"


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
