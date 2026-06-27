"""Request builder for Vertex AI Generative Language API (streamGenerateContent)."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from config.paths import config_dir_path
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict[str, Any]:
    """Build a Generative Language API request body from an Anthropic request."""
    logger.debug(
        "VERTEX_AI_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        openai_body = build_base_request_body(
            request_data,
            reasoning_replay=(
                ReasoningReplayMode.THINK_TAGS
                if thinking_enabled
                else ReasoningReplayMode.DISABLED
            ),
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    system_text, messages = _extract_system(openai_body.get("messages", []))
    contents = _openai_messages_to_contents(messages)

    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    if openai_body.get("tools"):
        body["tools"] = [
            {
                "functionDeclarations": [
                    _convert_function_declaration(tool) for tool in openai_body["tools"]
                ]
            }
        ]

    tool_choice = openai_body.get("tool_choice")
    if tool_choice and (tool_config := _convert_tool_choice(tool_choice)):
        body["toolConfig"] = tool_config

    generation_config = _build_generation_config(openai_body)
    if thinking_enabled:
        thinking = getattr(request_data, "thinking", None)
        budget = None
        if thinking:
            budget = (
                thinking.get("budget_tokens")
                if isinstance(thinking, dict)
                else getattr(thinking, "budget_tokens", None)
            )
        t_config: dict[str, Any] = {}
        if budget is not None:
            t_config["thinkingBudget"] = budget
        else:
            t_config["thinkingBudget"] = 2048
        generation_config["thinkingConfig"] = t_config

    if generation_config:
        body["generationConfig"] = generation_config

    logger.debug(
        "VERTEX_AI_REQUEST: conversion done model={} msgs={} tools={}",
        openai_body.get("model"),
        len(contents),
        len(openai_body.get("tools", [])),
    )
    return body


def _extract_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    remaining: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
        else:
            remaining.append(msg)
    return "\n\n".join(system_parts), remaining


def _openai_messages_to_contents(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_contents: list[dict[str, Any]] = []
    tool_id_map: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")
            tool_name = tool_id_map.get(tool_call_id, "tool")
            response = _parse_json_or_text(msg.get("content", ""))
            raw_contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_name,
                                "response": response,
                            }
                        }
                    ],
                }
            )
            continue

        content = msg.get("content", "")
        parts = _content_to_parts(content)

        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            # Drop empty text parts for assistant turns containing tool calls
            parts = [p for p in parts if p.get("text", "").strip()]
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name") or "tool"
                args = fn.get("arguments", "") or "{}"
                parsed_args = _parse_json_or_text(args)

                # Prepend the associated thought signature if cached to avoid 400 rejection
                sig = lookup_thought_signature(name, parsed_args)
                part_dict: dict[str, Any] = {
                    "functionCall": {
                        "name": name,
                        "args": parsed_args,
                    }
                }
                if sig:
                    part_dict["thoughtSignature"] = sig

                parts.append(part_dict)
                if tc.get("id"):
                    tool_id_map[str(tc["id"])] = name

        if not parts:
            parts = [{"text": ""}]

        raw_contents.append({"role": _map_role(role), "parts": parts})

    # Filter out empty turns to prevent bloated context and infinite loops from interrupted tools
    filtered_contents = [item for item in raw_contents if not _is_empty_turn(item)]
    if not filtered_contents:
        filtered_contents = raw_contents

    contents: list[dict[str, Any]] = []
    for item in filtered_contents:
        if not contents:
            contents.append(item)
            continue

        last_item = contents[-1]
        if last_item["role"] == item["role"]:
            last_has_fn_res = any("functionResponse" in p for p in last_item["parts"])
            curr_has_fn_res = any("functionResponse" in p for p in item["parts"])
            if last_has_fn_res == curr_has_fn_res:
                last_item["parts"].extend(item["parts"])
            else:
                contents.append(item)
        else:
            contents.append(item)

    return contents


def _is_empty_turn(item: dict[str, Any]) -> bool:
    parts = item.get("parts", [])
    if not parts:
        return True
    for p in parts:
        if "functionCall" in p or "functionResponse" in p:
            return False
        if p.get("text", "").strip():
            return False
    return True


def _clean_text(text: str) -> str:
    if "[Tool use interrupted]" in text:
        text = text.replace("[Tool use interrupted]", "")
    if "(no content)" in text:
        text = text.replace("(no content)", "")
    return text.strip()


def _content_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        cleaned = _clean_text(content)
        return [{"text": cleaned}] if cleaned else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                cleaned = _clean_text(str(item.get("text", "")))
                if cleaned:
                    parts.append({"text": cleaned})
            else:
                raise InvalidRequestError(
                    "Vertex AI Generative Language API only supports text content."
                )
        return parts
    if content is None:
        return []
    cleaned = _clean_text(str(content))
    return [{"text": cleaned}] if cleaned else []


def _sanitize_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    sanitized = {}
    allowed_keys = {
        "type",
        "properties",
        "required",
        "description",
        "items",
        "enum",
        "nullable",
        "format",
    }
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "properties" and isinstance(v, dict):
                sanitized[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                sanitized[k] = _sanitize_schema(v)
            else:
                sanitized[k] = v
    return sanitized


def _convert_function_declaration(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        return {
            "name": "tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        }
    fn = tool.get("function", {})
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": fn.get("name") or "tool",
        "description": fn.get("description") or "",
        "parameters": _sanitize_schema(params),
    }


def _convert_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if isinstance(tool_choice, str):
        return {
            "functionCallingConfig": {
                "mode": _tool_choice_mode(tool_choice),
            }
        }
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none", "required"}:
            return {"functionCallingConfig": {"mode": _tool_choice_mode(choice_type)}}
        if choice_type in {"function", "tool"}:
            name = (tool_choice.get("function", {}) or {}).get(
                "name"
            ) or tool_choice.get("name")
            f_config: dict[str, Any] = {
                "mode": "ANY",
            }
            if name:
                f_config["allowedFunctionNames"] = [str(name)]
            config: dict[str, Any] = {"functionCallingConfig": f_config}
            return config
    return None


def _tool_choice_mode(choice: str) -> str:
    if choice == "none":
        return "NONE"
    if choice == "required":
        return "ANY"
    return "AUTO"


def _build_generation_config(openai_body: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    max_tokens = openai_body.get("max_tokens")
    if isinstance(max_tokens, int):
        config["maxOutputTokens"] = max_tokens
    temperature = openai_body.get("temperature")
    if isinstance(temperature, (int, float)):
        config["temperature"] = float(temperature)
    top_p = openai_body.get("top_p")
    if isinstance(top_p, (int, float)):
        config["topP"] = float(top_p)
    stop_sequences = openai_body.get("stop")
    if isinstance(stop_sequences, list) and stop_sequences:
        config["stopSequences"] = stop_sequences
    return config


def _parse_json_or_text(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"content": raw}
            if isinstance(parsed, dict):
                return parsed
        return {"content": raw}
    if value is None:
        return {"content": ""}
    return {"content": str(value)}


def _map_role(role: Any) -> str:
    if role == "assistant":
        return "model"
    return "user"


def _get_cache_file_path() -> Path:
    cache_dir = config_dir_path() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "vertex_ai_signatures.json"


_IN_MEMORY_CACHES: dict[Path, dict[str, str]] = {}


def _get_cache() -> dict[str, str]:
    path = _get_cache_file_path()
    if path in _IN_MEMORY_CACHES:
        return _IN_MEMORY_CACHES[path]

    if not path.exists():
        _IN_MEMORY_CACHES[path] = {}
        return _IN_MEMORY_CACHES[path]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _IN_MEMORY_CACHES[path] = data
                return _IN_MEMORY_CACHES[path]
    except Exception as e:
        logger.warning("Failed to load vertex thought signature cache: {}", e)
    _IN_MEMORY_CACHES[path] = {}
    return _IN_MEMORY_CACHES[path]


_SAVE_LOCK = threading.Lock()


def _save_cache_to_disk(cache_copy: dict[str, str]) -> None:
    path = _get_cache_file_path()
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4()}.tmp")
    try:
        with _SAVE_LOCK:
            disk_cache: dict[str, str] = {}
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            disk_cache = data
                except Exception:
                    pass

            merged = {**disk_cache, **cache_copy}
            if len(merged) > 1000:
                excess = len(merged) - 1000
                keys_to_remove = list(merged.keys())[:excess]
                for k in keys_to_remove:
                    merged.pop(k, None)

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)
    except Exception as e:
        logger.warning("Failed to save vertex thought signature cache: {}", e)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


_PENDING_WRITES: set[asyncio.Future[None]] = set()


def _save_cache_async(cache_copy: dict[str, str]) -> asyncio.Future[None] | None:
    try:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, _save_cache_to_disk, cache_copy)
        _PENDING_WRITES.add(fut)
        fut.add_done_callback(_PENDING_WRITES.discard)
        return fut
    except RuntimeError:
        _save_cache_to_disk(cache_copy)
        return None


async def flush_thought_signatures() -> None:
    """Await all pending signature cache disk writes."""
    if _PENDING_WRITES:
        pending = list(_PENDING_WRITES)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _make_cache_key(name: str, args: dict[str, Any]) -> str:
    try:
        serialized = json.dumps(args, sort_keys=True)
    except Exception:
        serialized = str(args)
    return f"{name}:{serialized}"


def save_thought_signature(name: str, args: dict[str, Any], signature: str) -> None:
    key = _make_cache_key(name, args)
    cache = _get_cache()
    if key not in cache and len(cache) >= 1000:
        first_key = next(iter(cache))
        cache.pop(first_key, None)
    cache[key] = signature
    _save_cache_async(cache.copy())


def lookup_thought_signature(name: str, args: dict[str, Any]) -> str | None:
    key = _make_cache_key(name, args)
    cache = _get_cache()
    return cache.get(key)
