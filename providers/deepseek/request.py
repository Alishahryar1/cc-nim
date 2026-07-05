"""Request builder and DeepSeek native Anthropic compatibility sanitizer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError

# Block types not supported on DeepSeek partial Anthropic-compatible API.
_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)

# Block types silently stripped for DeepSeek since the content is typically
# also provided via tool_result (e.g. Claude Code attaches PDFs as document
# blocks alongside a Read tool_result containing the text).
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}


def _strip_unsupported_attachment_blocks(messages: Any) -> Any:
    """Remove image/document blocks that DeepSeek cannot process.

    Claude Code sends PDFs as ``document`` blocks alongside a Read ``tool_result``
    that already contains the extracted text. Stripping preserves the request
    instead of failing with an unsupported block error.
    """
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
    """True for Anthropic web_search / web_fetch-style tool definitions (listed tools)."""
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


def _validate_deepseek_native_request_dict(data: dict[str, Any]) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError(
            "DeepSeek native does not support mcp_servers on requests."
        )

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek native does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        c = message.get("content")
        if isinstance(c, list):
            _walk_block_list_for_unsupported(c, where=f"messages[{i}].content")
        if isinstance(c, str) and "<think>" in c:
            # Unusual, but block encoded redacted content — treat as unsafe for DeepSeek.
            pass

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

    has_thinking = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking" and isinstance(block.get("thinking"), str):
            has_thinking = bool(block["thinking"])
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


def _has_prior_assistant_without_thinking(data: dict[str, Any]) -> bool:
    """True if there is a prior assistant message with content but no thinking block.

    DeepSeek requires that thinking blocks from prior turns are replayed in the
    conversation history.  Claude Code 2.1+ does not include thinking blocks
    when replaying history, causing DeepSeek to reject the request with:
      "The content[].thinking in the thinking mode must be passed back to the API."
    When detected, placeholder thinking blocks are injected into the affected turns.
    """
    messages = data.get("messages") or []
    # Only relevant when there is more than one message (multi-turn conversation).
    if len(messages) < 2:
        return False
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            continue
        has_thinking = any(
            isinstance(block, dict) and block.get("type") == "thinking"
            for block in content
        )
        if not has_thinking:
            return True
    return False


def _remove_deepseek_thinking_hints(data: dict[str, Any]) -> None:
    """Remove request hints that can keep DeepSeek in thinking mode after fallback."""
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


def _tool_ids_from_content(content: list) -> frozenset[str]:
    """Extract tool_use IDs from an assistant message content list."""
    return frozenset(
        b["id"]
        for b in content
        if isinstance(b, dict)
        and b.get("type") == "tool_use"
        and isinstance(b.get("id"), str)
        and b["id"]
    )


def _inject_placeholder_thinking_blocks(messages: Any) -> Any:
    """Inject real or placeholder thinking blocks for assistant turns without thinking.

    DeepSeek-v4-pro requires thinking blocks in ALL prior assistant turns that contain
    tool_use blocks.  Claude Code 2.1+ strips thinking from conversation history, so we
    inject them here.  When the thinking cache has the real content (stored server-side
    during the original stream), we use it; otherwise we fall back to a short placeholder
    so DeepSeek still accepts the request.
    """
    from core.thinking_cache import ThinkingCache

    if not isinstance(messages, list):
        return messages

    result: list[Any] = []
    injected = 0
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        if msg.get("role") != "assistant":
            result.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        has_thinking = any(
            isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
            for b in content
        )
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)

        if has_tool_use and not has_thinking:
            tool_ids = _tool_ids_from_content(content)
            cached = ThinkingCache.lookup(tool_ids) if tool_ids else None
            thinking_text = cached if cached else "(prior reasoning not available)"
            placeholder = {"type": "thinking", "thinking": thinking_text}
            new_msg = dict(msg)
            new_msg["content"] = [placeholder] + list(content)
            result.append(new_msg)
            injected += 1
        else:
            result.append(msg)

    if injected:
        logger.debug(
            "DEEPSEEK_REQUEST: injected {} thinking block(s) for prior tool-use turns",
            injected,
        )
    return result


def sanitize_deepseek_messages_for_native(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant content for DeepSeek: unsigned ``thinking`` is allowed; no ``redacted_thinking``."""  # noqa: E501
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


def _serialize_tool_result_content(content: Any) -> str:
    """Serialize tool_result content to string for DeepSeek API.

    DeepSeek's Anthropic-compatible API expects tool_result.content to be a string,
    not an array of content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_tool_result_content(messages: Any) -> Any:
    """Normalize tool_result content to strings for DeepSeek API compatibility."""
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

        # Process content blocks
        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                # Normalize tool_result content to string
                normalized_block = dict(block)
                normalized_block["content"] = _serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


def _strip_reasoning_content_when_native(messages: Any) -> Any:
    """``reasoning_content`` is OpenAI-helper metadata; not part of native Anthropic body."""
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        msg = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(msg)
    return out


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a DeepSeek ``/v1/messages`` JSON body (Anthropic format)."""
    logger.debug(
        "DEEPSEEK_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(data["messages"])
    _validate_deepseek_native_request_dict(data)
    data.pop("extra_body", None)

    has_tool_history = _has_tool_history(data)
    has_replayable_tool_thinking = _has_replayable_tool_thinking(data)
    # DeepSeek-v4-pro is an always-on reasoning model: it requires thinking blocks in all
    # prior assistant turns with tool_use.  Claude Code 2.1+ drops thinking blocks from
    # history (no valid Anthropic signature).  Instead of disabling thinking (which still
    # fails because DeepSeek requires prior thinking regardless of the current request
    # parameter), inject placeholder thinking blocks into incomplete history turns.
    missing_prior_thinking = thinking_enabled and _has_prior_assistant_without_thinking(data)
    needs_placeholder_injection = thinking_enabled and (
        missing_prior_thinking or (has_tool_history and not has_replayable_tool_thinking)
    )
    effective_thinking_enabled = thinking_enabled

    if thinking_enabled:
        if needs_placeholder_injection:
            logger.debug(
                "DEEPSEEK_REQUEST: injecting placeholder thinking for incomplete history "
                "model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
            if "messages" in data:
                data["messages"] = _inject_placeholder_thinking_blocks(data["messages"])
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

    thinking_cfg = data.pop("thinking", None)
    if effective_thinking_enabled:
        thinking_payload: dict[str, Any] = {"type": "enabled"}
        if isinstance(thinking_cfg, dict):
            budget_tokens = thinking_cfg.get("budget_tokens")
            if isinstance(budget_tokens, int):
                thinking_payload["budget_tokens"] = budget_tokens
        data["thinking"] = thinking_payload

    if "messages" in data:
        data["messages"] = _strip_reasoning_content_when_native(
            _normalize_tool_result_content(
                sanitize_deepseek_messages_for_native(
                    data["messages"],
                    thinking_enabled=effective_thinking_enabled,
                )
            )
        )
    if "max_tokens" not in data or data.get("max_tokens") is None:
        data["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    data["stream"] = True

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        data.get("model"),
        len(data.get("messages", [])),
        len(data.get("tools", [])),
    )
    return data
