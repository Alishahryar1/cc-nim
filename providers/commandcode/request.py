"""Native Command Code request builder for /alpha/generate."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from core.anthropic.native_messages_request import dump_raw_messages_request


def _map_system(system: Any) -> str:
    """Map Anthropic system blocks to a single string."""
    if not system:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(system)


def _map_tool(tool: dict) -> dict:
    """Map Anthropic tool format to Command Code tool format."""
    # Anthropic tools look like: {"name": "...", "description": "...", "input_schema": {...}}
    # Command code uses the same format, so we can just pass it through.
    return tool


def _map_messages(messages: list[dict]) -> list[dict]:
    """Map Anthropic messages to Command Code messages."""
    cc_messages = []
    # Anthropic tool_result doesn't carry the tool name, but command code expects it.
    # We maintain a map of tool_use_id to tool_name from assistant messages.
    tool_names = {}

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            cc_messages.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )
            continue

        if not isinstance(content, list):
            cc_messages.append(
                {"role": role, "content": [{"type": "text", "text": str(content)}]}
            )
            continue

        cc_content = []
        for part in content:
            if not isinstance(part, dict):
                continue

            p_type = part.get("type")
            if p_type == "text":
                cc_content.append({"type": "text", "text": part.get("text", "")})
            elif p_type == "tool_use":
                # Anthropic tool_use: id, name, input
                tool_id = part.get("id")
                tool_name = part.get("name")
                if tool_id and tool_name:
                    tool_names[tool_id] = tool_name

                cc_content.append(
                    {
                        "type": "tool-call",
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "input": part.get("input", {}),
                    }
                )
            elif p_type == "tool_result":
                # Anthropic tool_result: tool_use_id, content
                tool_id = part.get("tool_use_id")
                tool_name = tool_names.get(tool_id, "unknown")

                part_content = part.get("content", "")
                val_str = ""
                if isinstance(part_content, str):
                    val_str = part_content
                elif isinstance(part_content, list):
                    # Join text blocks
                    texts = [
                        p.get("text", "")
                        for p in part_content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    val_str = "".join(texts)
                else:
                    val_str = str(part_content)

                out_type = "error-text" if part.get("is_error") else "text"

                cc_content.append(
                    {
                        "type": "tool-result",
                        "toolCallId": tool_id,
                        "toolName": tool_name,
                        "output": {"type": out_type, "value": val_str},
                    }
                )

        # If any part of this message is a tool-result, the whole message role must be "tool"
        if any(p.get("type") == "tool-result" for p in cc_content):
            role = "tool"

        cc_messages.append({"role": role, "content": cc_content})

    return cc_messages


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for Command Code /alpha/generate."""
    raw = dump_raw_messages_request(request_data)

    model = raw.get("model", "")
    system = _map_system(raw.get("system", ""))
    messages = _map_messages(raw.get("messages", []))
    tools = [_map_tool(t) for t in raw.get("tools", [])] if raw.get("tools") else []

    max_tokens = raw.get("max_tokens", 64000)
    temperature = raw.get("temperature", 0.3)

    cc_body = {
        "config": {
            "workingDir": ".",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "environment": "cli",
            "structure": [],
            "isGitRepo": False,
            "currentBranch": "",
            "mainBranch": "main",
            "gitStatus": "",
            "recentCommits": [],
        },
        "memory": "",
        "taste": "",
        "skills": "",
        "params": {
            "model": model,
            "messages": messages,
            "tools": tools,
            "system": system,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        "threadId": str(uuid.uuid4()),
    }

    logger.debug(
        "COMMANDCODE_REQUEST: build done model={} msgs={} tools={}",
        model,
        len(messages),
        len(tools),
    )
    return cc_body
