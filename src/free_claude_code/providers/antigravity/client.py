"""Google Antigravity CLI Provider implementation."""

import asyncio
import json
import logging
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import serialize_tool_result_content
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.antigravity.auth import (
    ANTIGRAVITY_CLIENT_NAME,
    ANTIGRAVITY_DEFAULT_BASE_URL,
    ANTIGRAVITY_GOOG_API_CLIENT,
    ANTIGRAVITY_USER_AGENT,
    DEFAULT_FALLBACK_PROJECT_ID,
    AntigravityAuth,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)
from free_claude_code.providers.model_listing import model_infos_from_ids

logger = logging.getLogger(__name__)


def _normalize_model_name(model: str) -> str:
    """Normalize model identifier by stripping prefixes."""
    m = str(model)
    m = m.removeprefix("models/").removeprefix("antigravity/")
    return m


def _extract_error_message(raw_text: str) -> str:
    """Extract a clean, human-readable message from raw API error responses."""
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                status = err.get("status")
                details = err.get("details", [])
                reason = None
                if isinstance(details, list):
                    for detail in details:
                        if isinstance(detail, dict) and "reason" in detail:
                            reason = detail["reason"]
                            break
                parts = []
                if reason:
                    parts.append(f"[{reason}]")
                elif status:
                    parts.append(f"[{status}]")
                if msg:
                    parts.append(msg)
                if parts:
                    return " ".join(parts)
            elif isinstance(err, str) and err:
                return err

            if "message" in data and isinstance(data["message"], str):
                return data["message"]
    except json.JSONDecodeError, TypeError:
        pass

    return text


def _raise_mapped_http_error(status_code: int, message: str) -> None:
    """Map HTTP status code to ProviderError exception."""
    clean_message = _extract_error_message(message) or message
    if status_code in (401, 403):
        raise AuthenticationError(clean_message)
    if status_code == 400:
        raise InvalidRequestError(clean_message)
    if status_code == 429:
        raise RateLimitError(clean_message)
    if status_code in (502, 503, 504):
        raise OverloadedError(clean_message)
    raise APIError(clean_message, status_code=status_code)


UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$comment",
    "propertyNames",
    "const",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
}


def _clean_gemini_schema(schema: Any) -> Any:
    """Recursively strip JSON Schema draft keywords unsupported by Google Gemini API."""
    if isinstance(schema, dict):
        cleaned: dict[str, Any] = {}
        for k, v in schema.items():
            if k in UNSUPPORTED_GEMINI_SCHEMA_KEYS:
                continue
            cleaned[k] = _clean_gemini_schema(v)
        return cleaned
    if isinstance(schema, list):
        return [_clean_gemini_schema(item) for item in schema]
    return schema


def _convert_anthropic_tools_to_gemini(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to Gemini functionDeclarations format."""
    declarations = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("input_schema", {})
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "")
            parameters = getattr(tool, "input_schema", {})

        if not name:
            continue

        decl = {
            "name": name,
            "description": description,
            "parameters": _clean_gemini_schema(parameters),
        }
        declarations.append(decl)

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]


def _convert_anthropic_messages_to_gemini(
    messages: list[Any],
    system: str | list[Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Convert Anthropic messages and system prompt to Gemini contents and systemInstruction."""
    contents: list[dict[str, Any]] = []
    system_text_parts: list[str] = []

    if system:
        if isinstance(system, str):
            system_text_parts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    system_text_parts.append(getattr(block, "text", ""))

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")

    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        if isinstance(content, list):
            for block in content:
                b_type = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )
                if b_type == "tool_use":
                    t_id = (
                        (block.get("id") or block.get("tool_use_id"))
                        if isinstance(block, dict)
                        else (
                            getattr(block, "id", None)
                            or getattr(block, "tool_use_id", "")
                        )
                    )
                    t_name = (
                        block.get("name")
                        if isinstance(block, dict)
                        else getattr(block, "name", "")
                    )
                    if t_id and t_name:
                        tool_name_map[str(t_id)] = t_name

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")

        if role == "system":
            if isinstance(content, str):
                system_text_parts.append(content)
            continue

        gemini_role = "model" if role in ("assistant", "model") else "user"
        parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append({"text": block})
                    continue

                b_type = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )

                if b_type == "text":
                    txt = (
                        block.get("text")
                        if isinstance(block, dict)
                        else getattr(block, "text", "")
                    )
                    if txt:
                        parts.append({"text": txt})
                elif b_type == "thinking":
                    th = (
                        block.get("thinking")
                        if isinstance(block, dict)
                        else getattr(block, "thinking", "")
                    )
                    if th:
                        parts.append({"thought": True, "text": th})
                elif b_type == "tool_use":
                    name = (
                        block.get("name")
                        if isinstance(block, dict)
                        else getattr(block, "name", "")
                    )
                    inp = (
                        block.get("input", {})
                        if isinstance(block, dict)
                        else getattr(block, "input", {})
                    )
                    parts.append(
                        {
                            "functionCall": {
                                "name": name,
                                "args": inp if isinstance(inp, dict) else {},
                            },
                        }
                    )
                elif b_type in ("image", "image_url"):
                    source = (
                        block.get("source", {})
                        if isinstance(block, dict)
                        else getattr(block, "source", {})
                    )
                    source_type = (
                        source.get("type")
                        if isinstance(source, dict)
                        else getattr(source, "type", None)
                    )
                    media_type = None
                    data = None

                    if source_type == "base64":
                        media_type = (
                            source.get("media_type")
                            if isinstance(source, dict)
                            else getattr(source, "media_type", "image/jpeg")
                        )
                        data = (
                            source.get("data")
                            if isinstance(source, dict)
                            else getattr(source, "data", "")
                        )
                    elif source_type == "url" or b_type == "image_url":
                        url = (
                            source.get("url")
                            if isinstance(source, dict)
                            else getattr(source, "url", "")
                        )
                        if not url:
                            img_obj = (
                                block.get("image_url", {})
                                if isinstance(block, dict)
                                else getattr(block, "image_url", {})
                            )
                            url = (
                                img_obj.get("url", "")
                                if isinstance(img_obj, dict)
                                else getattr(img_obj, "url", "")
                            )
                        if url and url.startswith("data:"):
                            try:
                                header, data = url.split(",", 1)
                                media_type = (
                                    header.split(";")[0].removeprefix("data:")
                                    or "image/jpeg"
                                )
                            except Exception:
                                pass

                    if data:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": media_type or "image/jpeg",
                                    "data": data,
                                }
                            }
                        )
                elif b_type == "tool_result":
                    t_id = (
                        block.get("tool_use_id")
                        if isinstance(block, dict)
                        else getattr(block, "tool_use_id", "tool")
                    )
                    t_name = (
                        block.get("name")
                        if isinstance(block, dict)
                        else getattr(block, "name", None)
                    )
                    func_name = t_name or tool_name_map.get(t_id, t_id)
                    res_content = (
                        block.get("content")
                        if isinstance(block, dict)
                        else getattr(block, "content", "")
                    )
                    is_err = (
                        block.get("is_error", False)
                        if isinstance(block, dict)
                        else getattr(block, "is_error", False)
                    )
                    serialized_content = serialize_tool_result_content(res_content)
                    if is_err:
                        err_text = serialized_content or "Tool execution failed"
                        response_dict = {
                            "error": err_text,
                            "output": err_text,
                        }
                    else:
                        response_dict = {
                            "output": serialized_content,
                        }
                    parts.append(
                        {
                            "functionResponse": {
                                "name": func_name,
                                "response": response_dict,
                            }
                        }
                    )

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    system_instruction = None
    if system_text_parts:
        system_instruction = {"parts": [{"text": "\n\n".join(system_text_parts)}]}

    return contents, system_instruction


class AntigravityProvider(BaseProvider):
    """Google Antigravity CLI provider implementing CodeAssist Gemini API with CLI fingerprinting."""

    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: AntigravityAuth | None = None,
        *,
        admission: Any | None = None,
    ):
        super().__init__(config, admission=admission)
        self._base_url = (config.base_url or ANTIGRAVITY_DEFAULT_BASE_URL).rstrip("/")
        self._auth = auth_manager or AntigravityAuth(base_url=self._base_url)
        self._client = httpx.AsyncClient(
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    def _is_thinking_enabled(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> bool:
        if thinking_enabled is not None:
            return thinking_enabled
        thinking = request.thinking
        if isinstance(thinking, dict):
            return thinking.get("type") == "enabled"
        return getattr(thinking, "type", None) == "enabled"

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: Any = None,
    ) -> None:
        """Validate request before opening an SSE stream."""
        pass

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    async def _fetch_model_ids_via_cli(self) -> frozenset[str]:
        """Fallback to fetching model IDs via agy CLI when direct API auth is unavailable."""
        fetched_ids: set[str] = set()
        agy_bin = shutil.which("agy") or "/home/omni/.local/bin/agy"
        if not os.path.exists(agy_bin):
            return frozenset()

        try:
            proc = await asyncio.create_subprocess_exec(
                agy_bin,
                "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            out = stdout.decode("utf-8", errors="ignore")
            for line in out.splitlines():
                line = line.strip()
                if (
                    line
                    and not line.startswith("⠋")
                    and not line.startswith("Fetching")
                ):
                    parts = line.split()
                    if parts:
                        norm = _normalize_model_name(parts[0])
                        fetched_ids.add(norm)
                        fetched_ids.add(f"antigravity/{norm}")
        except Exception as exc:
            logger.debug("CLI fallback model listing failed: %s", exc)

        return frozenset(fetched_ids)

    async def list_model_ids(self) -> frozenset[str]:
        """Fetch supported Antigravity CLI model IDs dynamically via Google Cloud Code Assist API or CLI fallback."""
        fetched_ids: set[str] = set()
        try:
            access_token = await self._auth.get_access_token_async()
            project_id = await self._auth.get_project_id_async()
            headers = self._build_request_headers(access_token)

            try:
                res = await self._client.post(
                    f"{self._base_url}/v1internal:retrieveUserQuota",
                    headers=headers,
                    json={"project": project_id},
                    timeout=5.0,
                )
                if res.status_code == 200:
                    data = res.json()
                    buckets = data.get("quotaBuckets", []) or data.get("buckets", [])
                    for bucket in buckets:
                        if isinstance(bucket, dict):
                            model_id = (
                                bucket.get("modelId")
                                or bucket.get("model")
                                or bucket.get("name")
                            )
                            if model_id:
                                norm = _normalize_model_name(str(model_id))
                                fetched_ids.add(norm)
                                fetched_ids.add(f"antigravity/{norm}")
                    if fetched_ids:
                        return frozenset(fetched_ids)
            except Exception as exc:
                logger.debug("POST retrieveUserQuota failed: %s", exc)

            try:
                load_body = {
                    "metadata": {
                        "ideType": "ANTIGRAVITY",
                        "platform": "PLATFORM_UNSPECIFIED",
                    }
                }
                res_load = await self._client.post(
                    f"{self._base_url}/v1internal:loadCodeAssist",
                    headers=headers,
                    json=load_body,
                    timeout=5.0,
                )
                if res_load.status_code == 200:
                    data = res_load.json()
                    models = data.get("models", []) or data.get("allowedModels", [])
                    for item in models:
                        if isinstance(item, str):
                            norm = _normalize_model_name(item)
                            fetched_ids.add(norm)
                            fetched_ids.add(f"antigravity/{norm}")
                        elif isinstance(item, dict) and "name" in item:
                            norm = _normalize_model_name(item["name"])
                            fetched_ids.add(norm)
                            fetched_ids.add(f"antigravity/{norm}")
            except Exception as exc:
                logger.debug("POST loadCodeAssist failed: %s", exc)
                raise
        except Exception as exc:
            logger.debug(
                "Antigravity direct HTTP auth/fetch failed: %s",
                exc,
            )

        if fetched_ids:
            return frozenset(fetched_ids)

        return frozenset()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return advertised model information."""
        return model_infos_from_ids(await self.list_model_ids())

    def _build_request_headers(self, access_token: str) -> dict[str, str]:
        """Construct exact Antigravity CLI v1.1.11 HTTP headers."""
        return {
            "User-Agent": ANTIGRAVITY_USER_AGENT,
            "X-Client-Name": ANTIGRAVITY_CLIENT_NAME,
            "X-Goog-Api-Client": ANTIGRAVITY_GOOG_API_CLIENT,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_request_body(
        self,
        request: MessagesRequest,
        model_name: str = "gemini-2.5-pro",
        project_id: str = DEFAULT_FALLBACK_PROJECT_ID,
        thinking_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Convert Anthropic request parameters into Gemini/CodeAssist request body with CLI metadata."""
        messages = request.messages
        system = request.system
        tools = request.tools

        contents, system_instruction = _convert_anthropic_messages_to_gemini(
            messages, system=system
        )

        gen_config: dict[str, Any] = {}
        max_tokens = request.max_tokens
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens

        temp = request.temperature
        if temp is not None:
            gen_config["temperature"] = temp

        top_p = request.top_p
        if top_p is not None:
            gen_config["topP"] = top_p

        stop_seqs = request.stop_sequences
        if stop_seqs:
            gen_config["stopSequences"] = stop_seqs

        is_thinking = self._is_thinking_enabled(request, thinking_enabled)
        if is_thinking:
            gen_config["thinkingConfig"] = {"includeThoughts": True}

        req_dict: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            req_dict["systemInstruction"] = system_instruction

        if tools:
            gemini_tools = _convert_anthropic_tools_to_gemini(tools)
            if gemini_tools:
                req_dict["tools"] = gemini_tools

        if gen_config:
            req_dict["generationConfig"] = gen_config

        body: dict[str, Any] = {
            "model": model_name,
            "request": req_dict,
        }

        if project_id:
            body["project"] = project_id

        return body

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format using Antigravity direct HTTP REST API."""
        model_raw = request.model or "gemini-2.5-pro"
        model_name = _normalize_model_name(model_raw)

        ledger = AnthropicStreamLedger(
            message_id=request_id,
            model=model_name,
            input_tokens=input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )

        sent_any_event = False
        try:
            access_token = await self._auth.get_access_token_async()
            project_id = await self._auth.get_project_id_async()

            headers = self._build_request_headers(access_token)
            body = self._build_request_body(
                request,
                model_name=model_name,
                project_id=project_id,
                thinking_enabled=thinking_enabled,
            )

            url = f"{self._base_url}/v1internal:streamGenerateContent?alt=sse"

            async with self._client.stream(
                "POST", url, headers=headers, json=body
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    error_text = response.text
                    clean_msg = (
                        _extract_error_message(error_text)
                        or f"HTTP {response.status_code}"
                    )
                    _raise_mapped_http_error(response.status_code, clean_msg)

                yield ledger.message_start()
                sent_any_event = True

                active_tool_index: int | None = None
                tool_call_count = 0

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line.removeprefix("data:").strip()
                    if not data_str or data_str == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    chunk_data = (
                        chunk.get("response", chunk)
                        if isinstance(chunk, dict)
                        else chunk
                    )
                    candidates = (
                        chunk_data.get("candidates", [])
                        if isinstance(chunk_data, dict)
                        else []
                    )
                    if not candidates:
                        continue

                    candidate = candidates[0]
                    content_obj = candidate.get("content", {})
                    parts = content_obj.get("parts", [])

                    for part in parts:
                        is_thought = part.get("thought") is True or isinstance(
                            part.get("thought"), str
                        )
                        if is_thought:
                            t_text = (
                                part["thought"]
                                if isinstance(part.get("thought"), str)
                                else part.get("text", "")
                            )
                            if t_text:
                                for ev in ledger.ensure_thinking_block():
                                    yield ev
                                yield ledger.emit_thinking_delta(t_text)

                        elif "text" in part:
                            text_delta = part["text"]
                            if text_delta:
                                for ev in ledger.ensure_text_block():
                                    yield ev
                                yield ledger.emit_text_delta(text_delta)

                        if "functionCall" in part:
                            fn_call = part["functionCall"]
                            fn_name = fn_call.get("name", "tool")
                            fn_args = fn_call.get("args", {})
                            args_str = (
                                json.dumps(fn_args)
                                if isinstance(fn_args, dict)
                                else str(fn_args)
                            )

                            tool_call_count += 1
                            tool_id = f"call_{uuid.uuid4().hex[:8]}"

                            if active_tool_index is not None:
                                yield ledger.stop_tool_block(active_tool_index)

                            tool_idx = ledger.blocks.allocate_index()
                            yield ledger.start_tool_block(tool_idx, tool_id, fn_name)
                            active_tool_index = tool_idx

                            yield ledger.emit_tool_delta(active_tool_index, args_str)
                            yield ledger.stop_tool_block(active_tool_index)
                            active_tool_index = None

                    finish_reason = candidate.get("finishReason")
                    if finish_reason:
                        stop_reason = "end_turn"
                        if finish_reason == "MAX_TOKENS":
                            stop_reason = "max_tokens"
                        elif tool_call_count > 0:
                            stop_reason = "tool_use"
                        ledger.stop_reason = stop_reason

                has_emitted_tool = ledger.has_emitted_tool_block()
                has_content_blocks = (
                    ledger.blocks.text_index != -1
                    or ledger.blocks.thinking_index != -1
                    or has_emitted_tool
                )
                if not has_content_blocks or (
                    not has_emitted_tool
                    and ledger.blocks.text_index != -1
                    and not ledger.accumulated_text
                ):
                    for ev in ledger.ensure_text_block():
                        yield ev
                    yield ledger.emit_text_delta(" ")

                for ev in ledger.close_all_blocks():
                    yield ev

                stop_r = ledger.stop_reason or "end_turn"
                yield ledger.message_delta(stop_r, 1)
                yield ledger.message_stop()

        except Exception as exc:
            raw_msg = getattr(exc, "message", str(exc))
            error_message = _extract_error_message(raw_msg) or str(exc)
            logger.error("Antigravity streaming error: %s", error_message)
            if not sent_any_event:
                yield ledger.message_start()
                for ev in ledger.ensure_text_block():
                    yield ev
                yield ledger.emit_text_delta(error_message)
                for ev in ledger.close_all_blocks():
                    yield ev
                yield ledger.message_delta("end_turn", 1)
                yield ledger.message_stop()
            else:
                for ev in ledger.close_all_blocks():
                    yield ev
                yield ledger.emit_text_delta(f"\n[Error: {error_message}]")
                yield ledger.message_delta("end_turn", 1)
                yield ledger.message_stop()
