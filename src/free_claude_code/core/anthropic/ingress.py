"""Anthropic Messages ingress conversion to canonical inference semantics."""

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from free_claude_code.core.inference import (
    Base64MediaSource,
    CacheControl,
    CacheControlType,
    CacheTTL,
    ClientReasoningIntent,
    DocumentContent,
    FileMediaSource,
    FunctionTool,
    ImageContent,
    InferenceItem,
    InferenceRequest,
    InstructionItem,
    InstructionOrigin,
    InstructionPlacement,
    MessageItem,
    MessageRole,
    OpenAIChatExtension,
    ReasoningItem,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolChoice,
    ToolChoiceMode,
    ToolResultItem,
    UrlMediaSource,
)
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort
from free_claude_code.core.replay_envelope import (
    ReplayEnvelopeError,
    decode_replay_envelope,
)

from .models import (
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockRedactedThinking,
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ContentBlockWebFetchToolResult,
    ContentBlockWebSearchToolResult,
    MessagesRequest,
    SystemContent,
    TokenCountRequest,
    Tool,
)

_NOOP_CONTEXT_EDIT = {"type": "clear_thinking_20251015", "keep": "all"}
_ALLOWED_BLOCK_EXTRAS = frozenset({"cache_control"})


class AnthropicIngressError(ValueError):
    """An Anthropic request cannot be represented without semantic loss."""


def messages_to_inference_request(request: MessagesRequest) -> InferenceRequest:
    """Validate and convert one Messages request exactly once."""

    validate_messages_field_policy(request)
    items = [*_system_items(request.system), *_message_items(request.messages)]
    tools = tuple(
        _function_tool(tool, index=index)
        for index, tool in enumerate(request.tools or ())
    )
    tool_choice, parallel_tool_calls = _tool_choice(request.tool_choice)
    extensions = (
        (OpenAIChatExtension(_json_object(request.extra_body, path="extra_body")),)
        if request.extra_body
        else ()
    )
    metadata = (
        _json_object(request.metadata, path="metadata")
        if request.metadata is not None
        else None
    )
    return InferenceRequest(
        model=request.model,
        items=tuple(items),
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=tuple(request.stop_sequences or ()),
        reasoning=_reasoning_intent(request),
        metadata=metadata,
        extensions=extensions,
    )


def token_count_to_inference_request(request: TokenCountRequest) -> InferenceRequest:
    """Convert the token-count product through the same canonical transcript."""

    _validate_token_count_top_level(request)
    items = [*_system_items(request.system), *_message_items(request.messages)]
    tools = tuple(
        _function_tool(tool, index=index)
        for index, tool in enumerate(request.tools or ())
    )
    tool_choice, parallel_tool_calls = _tool_choice(request.tool_choice)
    return InferenceRequest(
        model=request.model,
        items=tuple(items),
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning=_reasoning_intent(request),
    )


def validate_messages_field_policy(request: MessagesRequest) -> None:
    """Validate protocol-wide policy before local Messages short circuits."""

    _validate_top_level(request)
    _validate_system(request.system)
    _validate_messages(request.messages)
    for index, tool in enumerate(request.tools or ()):
        _validate_wire_tool(tool, path=f"tools[{index}]")
    _tool_choice(request.tool_choice)


def _validate_top_level(request: MessagesRequest) -> None:
    _reject_model_extras(request, path="request")
    _validate_thinking(request.thinking)
    _validate_common_controls(
        context_management=request.context_management,
        output_config=request.output_config,
        mcp_servers=request.mcp_servers,
    )


def _validate_token_count_top_level(request: TokenCountRequest) -> None:
    _reject_model_extras(request, path="request")
    _validate_thinking(request.thinking)
    _validate_common_controls(
        context_management=request.context_management,
        output_config=request.output_config,
        mcp_servers=request.mcp_servers,
    )
    _validate_system(request.system)
    _validate_messages(request.messages)
    for index, tool in enumerate(request.tools or ()):
        _validate_tool(tool, path=f"tools[{index}]")


def _validate_common_controls(
    *,
    context_management: dict[str, object] | None,
    output_config: dict[str, object] | None,
    mcp_servers: list[dict[str, object]] | None,
) -> None:
    if not _is_noop_context_management(context_management):
        raise AnthropicIngressError(
            "context_management contains unsupported active semantics"
        )
    if output_config:
        unsupported = sorted(str(key) for key in output_config if key != "effort")
        if unsupported:
            raise AnthropicIngressError(
                f"output_config.{unsupported[0]} is not supported"
            )
        _reasoning_effort(output_config.get("effort"), path="output_config.effort")
    if mcp_servers:
        raise AnthropicIngressError("mcp_servers must be omitted or empty")


def _is_noop_context_management(value: Mapping[str, object] | None) -> bool:
    if not value:
        return True
    if set(value) != {"edits"}:
        return False
    edits = value.get("edits")
    return isinstance(edits, list) and all(edit == _NOOP_CONTEXT_EDIT for edit in edits)


def _validate_thinking(value: BaseModel | None) -> None:
    if value is None:
        return
    extras = _block_extras(value)
    unknown = set(extras) - {"display"}
    if unknown:
        raise AnthropicIngressError(
            f"thinking.{sorted(str(key) for key in unknown)[0]} is not supported"
        )
    if "display" in extras and extras["display"] != "omitted":
        raise AnthropicIngressError(
            f"thinking.display has unsupported value {extras['display']!r}"
        )
    thinking_type = getattr(value, "type", None)
    if thinking_type not in {None, "adaptive", "disabled", "enabled"}:
        raise AnthropicIngressError(
            f"thinking.type has unsupported value {thinking_type!r}"
        )


def _system_items(value: str | list[SystemContent] | None) -> list[InstructionItem]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            InstructionItem(
                text=value,
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TOP_LEVEL,
            )
        ]
    return [
        InstructionItem(
            text=block.text,
            origin=InstructionOrigin.SYSTEM,
            placement=InstructionPlacement.TOP_LEVEL,
            cache_control=_cache_control(block, path=f"system[{index}]"),
        )
        for index, block in enumerate(value)
    ]


def _message_items(messages: Sequence[object]) -> list[InferenceItem]:
    items: list[InferenceItem] = []
    for index, raw_message in enumerate(messages):
        role = getattr(raw_message, "role", None)
        content = getattr(raw_message, "content", None)
        reasoning_content = getattr(raw_message, "reasoning_content", None)
        turn_id = f"turn_{index}"
        path = f"messages[{index}]"
        if role == "system":
            items.extend(_inline_instruction_items(content, turn_id=turn_id, path=path))
            continue
        if role not in {"user", "assistant"}:
            raise AnthropicIngressError(f"{path}.role is unsupported")
        message_role = MessageRole(role)
        if role == "assistant" and isinstance(reasoning_content, str):
            items.append(ReasoningItem(turn_id=turn_id, reasoning=reasoning_content))
        if isinstance(content, str):
            items.append(
                MessageItem(
                    turn_id=turn_id,
                    role=message_role,
                    content=(TextContent(content),),
                )
            )
            continue
        if not isinstance(content, list):
            raise AnthropicIngressError(f"{path}.content must be text or a list")
        for block_index, block in enumerate(content):
            block_path = f"{path}.content[{block_index}]"
            if isinstance(block, ContentBlockText):
                items.append(
                    MessageItem(
                        turn_id=turn_id,
                        role=message_role,
                        content=(
                            TextContent(
                                block.text,
                                cache_control=_cache_control(block, path=block_path),
                            ),
                        ),
                    )
                )
            elif isinstance(block, ContentBlockImage):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} assistant images are not supported"
                    )
                items.append(
                    MessageItem(
                        turn_id=turn_id,
                        role=message_role,
                        content=(
                            ImageContent(
                                _image_source(
                                    block.source, path=f"{block_path}.source"
                                ),
                                cache_control=_cache_control(block, path=block_path),
                            ),
                        ),
                    )
                )
            elif isinstance(block, ContentBlockDocument):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} assistant documents are not supported"
                    )
                items.append(
                    MessageItem(
                        turn_id=turn_id,
                        role=message_role,
                        content=(
                            DocumentContent(
                                _document_source(
                                    block.source,
                                    path=f"{block_path}.source",
                                ),
                                cache_control=_cache_control(block, path=block_path),
                            ),
                        ),
                    )
                )
            elif isinstance(block, ContentBlockThinking):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} thinking requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                artifacts = _reasoning_carrier(
                    block.signature,
                    kind=ReplayArtifactKind.THINKING_SIGNATURE,
                    path=f"{block_path}.signature",
                )
                items.append(
                    ReasoningItem(
                        turn_id=turn_id,
                        reasoning=block.thinking,
                        artifacts=artifacts,
                    )
                )
            elif isinstance(block, ContentBlockRedactedThinking):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} redacted_thinking requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                items.append(
                    ReasoningItem(
                        turn_id=turn_id,
                        reasoning="",
                        artifacts=_reasoning_carrier(
                            block.data,
                            kind=ReplayArtifactKind.REDACTED_THINKING,
                            path=f"{block_path}.data",
                        ),
                    )
                )
            elif isinstance(block, ContentBlockToolUse):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} tool_use requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                items.append(
                    ToolCallItem(
                        turn_id=turn_id,
                        call_id=_non_empty(block.id, path=f"{block_path}.id"),
                        kind=ToolCallKind.FUNCTION,
                        name=_non_empty(block.name, path=f"{block_path}.name"),
                        input=_json_object(block.input, path=f"{block_path}.input"),
                        artifacts=_tool_artifacts(block, path=block_path),
                    )
                )
            elif isinstance(block, ContentBlockToolResult):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} tool_result requires a user message"
                    )
                extras = _block_extras(block)
                unknown = set(extras) - {"cache_control", "is_error"}
                if unknown:
                    raise AnthropicIngressError(
                        f"{block_path}.{sorted(unknown)[0]} is not supported"
                    )
                is_error = extras.get("is_error", False)
                if not isinstance(is_error, bool):
                    raise AnthropicIngressError(
                        f"{block_path}.is_error must be boolean"
                    )
                items.append(
                    ToolResultItem(
                        turn_id=turn_id,
                        call_id=_non_empty(
                            block.tool_use_id,
                            path=f"{block_path}.tool_use_id",
                        ),
                        content=_json_value(
                            block.content, path=f"{block_path}.content"
                        ),
                        is_error=is_error,
                    )
                )
            else:
                block_type = getattr(block, "type", type(block).__name__)
                raise AnthropicIngressError(
                    f"{block_path} type {block_type!r} is not supported for provider execution"
                )
    return items


def _inline_instruction_items(
    content: object,
    *,
    turn_id: str,
    path: str,
) -> list[InstructionItem]:
    if isinstance(content, str):
        return [
            InstructionItem(
                text=content,
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TRANSCRIPT,
                turn_id=turn_id,
            )
        ]
    if not isinstance(content, list):
        raise AnthropicIngressError(f"{path}.content must contain only text")
    items: list[InstructionItem] = []
    for index, block in enumerate(content):
        block_path = f"{path}.content[{index}]"
        if not isinstance(block, ContentBlockText):
            raise AnthropicIngressError(f"{block_path} must be a text block")
        items.append(
            InstructionItem(
                text=block.text,
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TRANSCRIPT,
                cache_control=_cache_control(block, path=block_path),
                turn_id=turn_id,
            )
        )
    return items


def _function_tool(tool: Tool, *, index: int) -> FunctionTool:
    path = f"tools[{index}]"
    _validate_tool(tool, path=path)
    schema = tool.input_schema or {"type": "object", "properties": {}}
    extras = _block_extras(tool)
    strict = extras.get("strict", False)
    if not isinstance(strict, bool):
        raise AnthropicIngressError(f"{path}.strict must be boolean")
    return FunctionTool(
        name=_non_empty(tool.name, path=f"{path}.name"),
        description=tool.description,
        input_schema=_json_object(schema, path=f"{path}.input_schema"),
        strict=strict,
        cache_control=_cache_control(tool, path=path),
    )


def _validate_tool(tool: Tool, *, path: str) -> None:
    if tool.type is not None:
        raise AnthropicIngressError(
            f"{path}.type {tool.type!r} is provider-managed and cannot be forwarded"
        )
    unknown = set(_block_extras(tool)) - {"cache_control", "strict"}
    if unknown:
        raise AnthropicIngressError(f"{path}.{sorted(unknown)[0]} is not supported")
    _cache_control(tool, path=path)


def _validate_wire_tool(tool: Tool, *, path: str) -> None:
    """Validate a tool before local server-tool routing is known."""

    if tool.type is None:
        _validate_tool(tool, path=path)
        return
    if tool.type.startswith(("web_search_", "web_fetch_")):
        return
    raise AnthropicIngressError(
        f"{path}.type {tool.type!r} is provider-managed and cannot be forwarded"
    )


def _tool_choice(
    value: dict[str, object] | None,
) -> tuple[ToolChoice | None, bool | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise AnthropicIngressError("tool_choice must be an object")
    choice_type = value.get("type")
    allowed = {"type", "name", "disable_parallel_tool_use"}
    unknown = set(value) - allowed
    if unknown:
        raise AnthropicIngressError(
            f"tool_choice.{sorted(str(key) for key in unknown)[0]} is not supported"
        )
    disable_parallel = value.get("disable_parallel_tool_use")
    if disable_parallel is not None and not isinstance(disable_parallel, bool):
        raise AnthropicIngressError(
            "tool_choice.disable_parallel_tool_use must be boolean"
        )
    parallel = None if disable_parallel is None else not disable_parallel
    if choice_type == "auto":
        return ToolChoice(ToolChoiceMode.AUTO), parallel
    if choice_type in {"any", "required"}:
        return ToolChoice(ToolChoiceMode.REQUIRED), parallel
    if choice_type == "none":
        return ToolChoice(ToolChoiceMode.NONE), parallel
    if choice_type == "tool":
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise AnthropicIngressError("tool_choice.name must be a non-empty string")
        return (
            ToolChoice(
                ToolChoiceMode.SPECIFIC,
                kind=ToolCallKind.FUNCTION,
                name=name,
            ),
            parallel,
        )
    raise AnthropicIngressError(f"tool_choice.type {choice_type!r} is not supported")


def _reasoning_intent(
    request: MessagesRequest | TokenCountRequest,
) -> ClientReasoningIntent:
    thinking = request.thinking
    budget = thinking.budget_tokens if thinking is not None else None
    if budget is not None and (
        not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
    ):
        raise AnthropicIngressError("thinking.budget_tokens must be a positive integer")
    if thinking is None:
        control = ReasoningControl.DEFAULT
    elif thinking.type == "disabled" or (
        "enabled" in thinking.model_fields_set and thinking.enabled is False
    ):
        control = ReasoningControl.OFF
    elif (
        thinking.type in {"adaptive", "enabled"}
        or ("enabled" in thinking.model_fields_set and thinking.enabled is True)
        or budget is not None
    ):
        control = ReasoningControl.ON
    else:
        control = ReasoningControl.DEFAULT
    output_config = request.output_config or {}
    effort = _reasoning_effort(
        output_config.get("effort"),
        path="output_config.effort",
    )
    if output_config.get("effort") == "none":
        control = ReasoningControl.OFF
    if control is ReasoningControl.OFF:
        budget = None
    if budget is not None:
        control = ReasoningControl.ON
    return ClientReasoningIntent(control=control, effort=effort, budget_tokens=budget)


def _reasoning_effort(value: object, *, path: str) -> ReasoningEffort | None:
    if value is None or value == "none":
        return None
    if not isinstance(value, str):
        raise AnthropicIngressError(f"{path} must be a string")
    try:
        return ReasoningEffort(value.strip().lower())
    except ValueError as exc:
        raise AnthropicIngressError(f"{path} has unsupported value {value!r}") from exc


def _image_source(value: object, *, path: str) -> UrlMediaSource | Base64MediaSource:
    source = _mapping(value, path=path)
    source_type = source.get("type")
    if source_type == "url":
        _require_exact_keys(source, allowed={"type", "url"}, path=path)
        return UrlMediaSource(_required_mapping_str(source, "url", path=path))
    if source_type == "base64":
        _require_exact_keys(
            source,
            allowed={"type", "media_type", "data"},
            path=path,
        )
        return Base64MediaSource(
            media_type=_required_mapping_str(source, "media_type", path=path),
            data=_required_mapping_str(source, "data", path=path),
        )
    raise AnthropicIngressError(
        f"{path}.type must be 'url' or 'base64', got {source_type!r}"
    )


def _document_source(value: object, *, path: str):
    source = _mapping(value, path=path)
    source_type = source.get("type")
    if source_type == "url":
        _require_exact_keys(source, allowed={"type", "url"}, path=path)
        return UrlMediaSource(_required_mapping_str(source, "url", path=path))
    if source_type == "base64":
        _require_exact_keys(
            source,
            allowed={"type", "media_type", "data"},
            path=path,
        )
        return Base64MediaSource(
            media_type=_required_mapping_str(source, "media_type", path=path),
            data=_required_mapping_str(source, "data", path=path),
        )
    if source_type in {"file", "file_id"}:
        _require_exact_keys(source, allowed={"type", "file_id"}, path=path)
        return FileMediaSource(_required_mapping_str(source, "file_id", path=path))
    raise AnthropicIngressError(
        f"{path}.type must be 'url', 'base64', or 'file', got {source_type!r}"
    )


def _validate_system(value: str | list[SystemContent] | None) -> None:
    if isinstance(value, list):
        for index, block in enumerate(value):
            _cache_control(block, path=f"system[{index}]")


def _validate_messages(messages: Sequence[object]) -> None:
    for index, raw_message in enumerate(messages):
        role = getattr(raw_message, "role", None)
        content = getattr(raw_message, "content", None)
        path = f"messages[{index}]"
        if isinstance(raw_message, BaseModel):
            _reject_model_extras(raw_message, path=path)
        if role == "system":
            _validate_inline_instruction_content(content, path=path)
            continue
        if role not in {"user", "assistant"}:
            raise AnthropicIngressError(f"{path}.role is unsupported")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise AnthropicIngressError(f"{path}.content must be text or a list")
        for block_index, block in enumerate(content):
            block_path = f"{path}.content[{block_index}]"
            if isinstance(block, ContentBlockText):
                _cache_control(block, path=block_path)
            elif isinstance(block, ContentBlockImage):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} assistant images are not supported"
                    )
                _cache_control(block, path=block_path)
                _image_source(block.source, path=f"{block_path}.source")
            elif isinstance(block, ContentBlockDocument):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} assistant documents are not supported"
                    )
                _cache_control(block, path=block_path)
                _document_source(block.source, path=f"{block_path}.source")
            elif isinstance(block, ContentBlockThinking):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} thinking requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                _reasoning_carrier(
                    block.signature,
                    kind=ReplayArtifactKind.THINKING_SIGNATURE,
                    path=f"{block_path}.signature",
                )
            elif isinstance(block, ContentBlockRedactedThinking):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} redacted_thinking requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                _reasoning_carrier(
                    block.data,
                    kind=ReplayArtifactKind.REDACTED_THINKING,
                    path=f"{block_path}.data",
                )
            elif isinstance(block, ContentBlockToolUse):
                if role != "assistant":
                    raise AnthropicIngressError(
                        f"{block_path} tool_use requires an assistant message"
                    )
                _cache_control(block, path=block_path)
                _non_empty(block.id, path=f"{block_path}.id")
                _non_empty(block.name, path=f"{block_path}.name")
                _json_object(block.input, path=f"{block_path}.input")
                _tool_artifacts(block, path=block_path)
            elif isinstance(block, ContentBlockToolResult):
                if role != "user":
                    raise AnthropicIngressError(
                        f"{block_path} tool_result requires a user message"
                    )
                extras = _block_extras(block)
                unknown = set(extras) - {"cache_control", "is_error"}
                if unknown:
                    raise AnthropicIngressError(
                        f"{block_path}.{sorted(unknown)[0]} is not supported"
                    )
                is_error = extras.get("is_error", False)
                if not isinstance(is_error, bool):
                    raise AnthropicIngressError(
                        f"{block_path}.is_error must be boolean"
                    )
                _non_empty(
                    block.tool_use_id,
                    path=f"{block_path}.tool_use_id",
                )
                _json_value(block.content, path=f"{block_path}.content")
            elif isinstance(
                block,
                ContentBlockServerToolUse
                | ContentBlockWebSearchToolResult
                | ContentBlockWebFetchToolResult,
            ):
                continue
            else:
                block_type = getattr(block, "type", type(block).__name__)
                raise AnthropicIngressError(
                    f"{block_path} type {block_type!r} is not supported"
                )


def _validate_inline_instruction_content(content: object, *, path: str) -> None:
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise AnthropicIngressError(f"{path}.content must contain only text")
    for index, block in enumerate(content):
        block_path = f"{path}.content[{index}]"
        if not isinstance(block, ContentBlockText):
            raise AnthropicIngressError(f"{block_path} must be a text block")
        _cache_control(block, path=block_path)


def _cache_control(block: BaseModel, *, path: str) -> CacheControl | None:
    extras = _block_extras(block)
    allowed = set(_ALLOWED_BLOCK_EXTRAS)
    if isinstance(block, ContentBlockToolUse):
        allowed.add("extra_content")
    if isinstance(block, Tool):
        allowed.add("strict")
    unknown = set(extras) - allowed
    if unknown:
        raise AnthropicIngressError(f"{path}.{sorted(unknown)[0]} is not supported")
    value = extras.get("cache_control")
    if value is None:
        return None
    mapping = _mapping(value, path=f"{path}.cache_control")
    if set(mapping) - {"type", "ttl"}:
        key = sorted(str(key) for key in set(mapping) - {"type", "ttl"})[0]
        raise AnthropicIngressError(f"{path}.cache_control.{key} is not supported")
    if mapping.get("type") != "ephemeral":
        raise AnthropicIngressError(f"{path}.cache_control.type must be 'ephemeral'")
    raw_ttl = mapping.get("ttl")
    if raw_ttl is None:
        ttl = None
    else:
        try:
            ttl = CacheTTL(raw_ttl)
        except (TypeError, ValueError) as exc:
            raise AnthropicIngressError(
                f"{path}.cache_control.ttl must be '5m' or '1h'"
            ) from exc
    return CacheControl(CacheControlType.EPHEMERAL, ttl)


def _reasoning_carrier(
    value: str | None,
    *,
    kind: ReplayArtifactKind,
    path: str,
) -> tuple[ReplayArtifact, ...]:
    if not value:
        return ()
    try:
        decoded = decode_replay_envelope(
            value,
            attachment=ReplayAttachment.REASONING,
        )
    except ReplayEnvelopeError as exc:
        raise AnthropicIngressError(f"{path}: {exc}") from exc
    if decoded is not None:
        return decoded
    return (
        ReplayArtifact(
            origin=ReplayArtifactOrigin.ANTHROPIC,
            kind=kind,
            attachment=ReplayAttachment.REASONING,
            payload=value,
        ),
    )


def _tool_artifacts(
    block: ContentBlockToolUse,
    *,
    path: str,
) -> tuple[ReplayArtifact, ...]:
    extra_content = _block_extras(block).get("extra_content")
    if extra_content is None:
        return ()
    mapping = _mapping(extra_content, path=f"{path}.extra_content")
    envelope = mapping.get("fcc_replay")
    if envelope is not None:
        if set(mapping) != {"fcc_replay"} or not isinstance(envelope, str):
            raise AnthropicIngressError(
                f"{path}.extra_content.fcc_replay must be the only non-empty string carrier"
            )
        try:
            decoded = decode_replay_envelope(
                envelope,
                attachment=ReplayAttachment.TOOL_CALL,
            )
        except ReplayEnvelopeError as exc:
            raise AnthropicIngressError(
                f"{path}.extra_content.fcc_replay: {exc}"
            ) from exc
        if decoded is None:
            raise AnthropicIngressError(
                f"{path}.extra_content.fcc_replay must contain an FCC replay envelope"
            )
        return decoded
    return (
        ReplayArtifact(
            origin=ReplayArtifactOrigin.OPENAI_COMPATIBLE,
            kind=ReplayArtifactKind.TOOL_EXTRA_CONTENT,
            attachment=ReplayAttachment.TOOL_CALL,
            payload=_json_object(mapping, path=f"{path}.extra_content"),
        ),
    )


def _block_extras(block: BaseModel) -> Mapping[str, object]:
    return cast(Mapping[str, object], block.model_extra or {})


def _reject_model_extras(model: BaseModel, *, path: str) -> None:
    extras = model.model_extra or {}
    if extras:
        key = sorted(str(key) for key in extras)[0]
        raise AnthropicIngressError(f"{path}.{key} is not supported")


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AnthropicIngressError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _require_exact_keys(
    mapping: Mapping[str, object],
    *,
    allowed: set[str],
    path: str,
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise AnthropicIngressError(
            f"{path}.{sorted(str(key) for key in unknown)[0]} is not supported"
        )


def _required_mapping_str(mapping: Mapping[str, object], key: str, *, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AnthropicIngressError(f"{path}.{key} must be a non-empty string")
    return value


def _non_empty(value: str, *, path: str) -> str:
    if not value.strip():
        raise AnthropicIngressError(f"{path} must be a non-empty string")
    return value


def _json_object(value: object, *, path: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, path=path)
    return {
        str(key): _json_value(item, path=f"{path}.{key}")
        for key, item in mapping.items()
    }


def _json_value(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="python"), path=path)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise AnthropicIngressError(f"{path} object keys must be strings")
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise AnthropicIngressError(f"{path} must contain JSON-compatible values")
