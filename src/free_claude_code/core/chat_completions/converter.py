"""Convert OpenAI Chat Completions requests to Anthropic Messages format."""

import json
import uuid
from typing import Any


class ChatCompletionsToAnthropicConverter:
    """Convert OpenAI Chat Completions format to Anthropic Messages format."""

    @staticmethod
    def to_anthropic_payload(request: Any) -> dict[str, Any]:
        """Convert a ChatCompletionsRequest to an Anthropic Messages payload."""
        messages: list[dict[str, Any]] = []
        system_parts: list[str] = []

        for msg in request.messages:
            role = msg.role
            content = msg.content

            if role == "system":
                # Anthropic uses a top-level `system` field
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    system_parts.extend(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                continue

            if role == "user":
                if isinstance(content, str):
                    messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Convert multipart content
                    anthropic_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                anthropic_content.append(
                                    {
                                        "type": "text",
                                        "text": part.get("text", ""),
                                    }
                                )
                            elif part.get("type") == "image_url":
                                image_url = part.get("image_url", {})
                                url = image_url.get("url", "")
                                if url.startswith("data:"):
                                    # base64 image
                                    media_type = url.split(";")[0].split(":")[1]
                                    data = url.split(",", 1)[1]
                                    anthropic_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data,
                                            },
                                        }
                                    )
                                else:
                                    anthropic_content.append(
                                        {
                                            "type": "image",
                                            "source": {"type": "url", "url": url},
                                        }
                                    )
                    if anthropic_content:
                        messages.append({"role": "user", "content": anthropic_content})
                    else:
                        messages.append({"role": "user", "content": str(content)})
                else:
                    messages.append({"role": "user", "content": str(content)})

            elif role == "assistant":
                if isinstance(content, str):
                    messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # Already Anthropic-style content blocks
                    messages.append({"role": "assistant", "content": content})
                elif msg.tool_calls:
                    # Assistant with tool calls
                    content_blocks: list[dict[str, Any]] = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg.tool_calls:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError, TypeError:
                                args = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": func.get("name", ""),
                                "input": args,
                            }
                        )
                    messages.append({"role": "assistant", "content": content_blocks})
                else:
                    messages.append({"role": "assistant", "content": content or ""})

            elif role == "tool":
                # Tool result message
                tool_use_id = msg.tool_call_id or ""
                tool_content = (
                    content if isinstance(content, str) else json.dumps(content)
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_content,
                            }
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,  # Always stream for SSE conversion
        }

        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            if isinstance(request.stop, list):
                payload["stop_sequences"] = request.stop
            else:
                payload["stop_sequences"] = [request.stop]

        # Convert tools
        if request.tools:
            anthropic_tools = []
            for tool in request.tools:
                func = tool.function
                anthropic_tools.append(
                    {
                        "name": func.name,
                        "description": func.description or "",
                        "input_schema": func.parameters
                        or {"type": "object", "properties": {}},
                    }
                )
            payload["tools"] = anthropic_tools

        # Convert tool_choice
        if request.tool_choice is not None:
            tc = request.tool_choice
            if isinstance(tc, str):
                if tc == "auto":
                    payload["tool_choice"] = {"type": "auto"}
                elif tc == "none":
                    pass  # No tool_choice means no tools
                elif tc == "required":
                    payload["tool_choice"] = {"type": "any"}
            elif isinstance(tc, dict):
                tc_type = tc.get("type")
                if tc_type == "function":
                    payload["tool_choice"] = {
                        "type": "tool",
                        "name": tc.get("function", {}).get("name", ""),
                    }
                elif tc_type == "auto":
                    payload["tool_choice"] = {"type": "auto"}

        return payload
