"""Message and tool format converters."""

import json
from typing import Any, Dict, List, Optional


def get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: Any) -> Optional[str]:
    """Get block type from object or dict."""
    return get_block_attr(block, "type")


class AnthropicToOpenAIConverter:
    """Converts Anthropic message format to OpenAI format."""

    @staticmethod
    def convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert a list of Anthropic messages to OpenAI format."""
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_assistant_message(content)
                    )
                elif role == "user":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_user_message(content)
                    )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    @staticmethod
    def _convert_assistant_message(content: List[Any]) -> List[Dict[str, Any]]:
        """Convert assistant message blocks."""
        text_parts = []
        tool_calls = []
        reasoning_parts = []

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                reasoning_parts.append(get_block_attr(block, "thinking", ""))
            elif block_type == "tool_use":
                tool_input = get_block_attr(block, "input", {})
                tool_calls.append(
                    {
                        "id": get_block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": get_block_attr(block, "name"),
                            "arguments": json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input),
                        },
                    }
                )

        # Merge everything into content for NIM/Mistral compatibility
        # Anthropic 'thinking' blocks are converted to <thought> tags
        actual_content = []
        if reasoning_parts:
            # Join reasoning parts and handle as a separate block
            reasoning_str = "\n".join(reasoning_parts)
            actual_content.append(f"<think>\n{reasoning_str}\n</think>")

        if text_parts:
            actual_content.append("\n".join(text_parts))

        content_str = "\n\n".join(actual_content)

        # Ensure content is never an empty string for assistant messages
        # NIM (especially Mistral models) requires non-empty content if there are no tool calls
        if not content_str and not tool_calls:
            content_str = " "

        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls

        return [msg]

    @staticmethod
    def _convert_user_message(content: List[Any]) -> List[Dict[str, Any]]:
        """Convert user message blocks (including tool results)."""
        result = []
        text_parts = []

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "tool_result":
                tool_content = get_block_attr(block, "content", "")
                if isinstance(tool_content, list):
                    tool_content = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in tool_content
                    )
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": get_block_attr(block, "tool_use_id"),
                        "content": str(tool_content) if tool_content else "",
                    }
                )

        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})

        return result

    @staticmethod
    def convert_tools(tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert Anthropic tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_system_prompt(system: Any) -> Optional[Dict[str, str]]:
        """Convert Anthropic system prompt to OpenAI format."""
        if isinstance(system, str):
            return {"role": "system", "content": system}
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if get_block_type(block) == "text":
                    text_parts.append(get_block_attr(block, "text", ""))
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None
