"""DeepSeek Anthropic-to-OpenAI chat request policy."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic import serialize_tool_result_content
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="DEEPSEEK",
    include_extra_body=True,
)

_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}


def build_deepseek_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a DeepSeek Chat Completions body from an Anthropic request."""
    logger.debug(
        "DEEPSEEK_REQUEST: chat build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(data["messages"])
    _validate_deepseek_request_dict(data)
    _downgrade_forced_tool_choice(data)

    has_tool_history = _has_tool_history(data)
    has_replayable_tool_thinking = _has_replayable_tool_thinking(data)
    unsafe_tool_followup = has_tool_history and not has_replayable_tool_thinking
    effective_thinking_enabled = thinking_enabled and not unsafe_tool_followup
    if thinking_enabled:
        if unsafe_tool_followup:
            logger.debug(
                "DEEPSEEK_REQUEST: disabling thinking for tool follow-up without "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
            _remove_deepseek_thinking_hints(data)
        elif has_tool_history:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for tool follow-up with "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
        elif data.get("tools") or data.get("tool_choice"):
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for initial tool request "
                "model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )

    if "messages" in data:
        data["messages"] = _normalize_tool_result_content(
            sanitize_deepseek_messages_for_openai(
                data["messages"],
                thinking_enabled=effective_thinking_enabled,
            )
        )

    sanitized_request = _request_from_dict(request_data, data)
    body = build_openai_chat_request_body(
        sanitized_request,
        thinking_enabled=effective_thinking_enabled,
        policy=_REQUEST_POLICY,
        postprocessors=(_apply_deepseek_chat_extras,),
    )
    if "max_tokens" not in body or body.get("max_tokens") is None:
        body["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def sanitize_deepseek_messages_for_openai(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant content before converting to DeepSeek Chat Completions."""
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        if not thinking_enabled:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
        else:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict) and block.get("type") == "redacted_thinking"
                )
            ]
        new_msg = dict(message)
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _strip_unsupported_attachment_blocks(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    stripped: list[Any] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0

    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        message_dropped_attachment = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                    top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                    message_dropped_attachment = True
                    continue
                if btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        filtered_inner: list[Any] = []
                        for sub in inner:
                            if (
                                isinstance(sub, dict)
                                and sub.get("type") in _STRIPPABLE_MESSAGE_BLOCK_TYPES
                            ):
                                sub_type = sub["type"]
                                nested_dropped[sub_type] = (
                                    nested_dropped.get(sub_type, 0) + 1
                                )
                                continue
                            filtered_inner.append(sub)
                        if not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_block = dict(block)
                        new_block["content"] = filtered_inner
                        new_content.append(new_block)
                        continue
            new_content.append(block)
        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_BLOCK]
            placeholder_replacements += 1
        new_msg = dict(message)
        new_msg["content"] = new_content
        stripped.append(new_msg)

    if top_level_dropped or nested_dropped:
        logger.warning(
            "DEEPSEEK_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "DeepSeek has no vision/document support; the model will not see this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped


def _is_server_listed_tool(tool: Mapping[str, Any]) -> bool:
    name = (tool.get("name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.get("type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(blocks: Any, *, where: str) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result" and "content" in block:
            _walk_block_list_for_unsupported(
                block["content"], where=f"{where} (tool_result content)"
            )


def _validate_deepseek_request_dict(data: dict[str, Any]) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError("DeepSeek does not support mcp_servers on requests.")

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            _walk_block_list_for_unsupported(content, where=f"messages[{i}].content")

    system = data.get("system")
    if isinstance(system, list):
        _walk_block_list_for_unsupported(system, where="system")


def _has_tool_history_blocks(message: Mapping[str, Any]) -> bool:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if role == "assistant" and btype == "tool_use":
            return True
        if role == "user" and btype == "tool_result":
            return True
    return False


def _has_replayable_thinking_before_tool_use(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False

    has_thinking = isinstance(message.get("reasoning_content"), str)
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking" and isinstance(block.get("thinking"), str):
            has_thinking = True
            continue
        if btype == "tool_use":
            return has_thinking
    return False


def _has_tool_history(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_tool_history_blocks(message):
            return True
    return False


def _has_replayable_tool_thinking(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_replayable_thinking_before_tool_use(
            message
        ):
            return True
    return False


def _remove_deepseek_thinking_hints(data: dict[str, Any]) -> None:
    output_config = data.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        cleaned_output_config = dict(output_config)
        cleaned_output_config.pop("effort", None)
        if cleaned_output_config:
            data["output_config"] = cleaned_output_config
        else:
            data.pop("output_config", None)

    context_management = data.get("context_management")
    if not isinstance(context_management, dict):
        return
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return
    filtered_edits = [
        edit
        for edit in edits
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("type"), str)
            and edit["type"].startswith("clear_thinking_")
        )
    ]
    if len(filtered_edits) == len(edits):
        return
    cleaned_context_management = dict(context_management)
    if filtered_edits:
        cleaned_context_management["edits"] = filtered_edits
        data["context_management"] = cleaned_context_management
    else:
        cleaned_context_management.pop("edits", None)
        if cleaned_context_management:
            data["context_management"] = cleaned_context_management
        else:
            data.pop("context_management", None)


def _normalize_tool_result_content(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                normalized_block = dict(block)
                normalized_block["content"] = serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


def _downgrade_forced_tool_choice(data: dict[str, Any]) -> None:
    tool_choice = data.get("tool_choice")
    if not isinstance(tool_choice, dict):
        return
    if tool_choice.get("type") != "tool" or not isinstance(
        tool_choice.get("name"), str
    ):
        return
    logger.debug(
        "DEEPSEEK_REQUEST: downgrading forced tool_choice to auto for unsupported "
        "native request shape tool={}",
        tool_choice["name"],
    )
    data["tool_choice"] = {"type": "auto"}


def _request_from_dict(request_data: Any, data: dict[str, Any]) -> Any:
    validator = getattr(type(request_data), "model_validate", None)
    if callable(validator):
        return validator(data)
    return SimpleNamespace(**data)


def _apply_deepseek_chat_extras(
    body: dict[str, Any], _request_data: Any, thinking_enabled: bool
) -> None:
    if not thinking_enabled or body.get("model") == "deepseek-reasoner":
        return
    extra_body = body.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        extra_body.setdefault("thinking", {"type": "enabled"})
