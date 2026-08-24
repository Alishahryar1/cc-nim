"""OpenAI Responses ingress conversion to canonical inference semantics."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from free_claude_code.core.inference import (
    ClientReasoningIntent,
    CustomTool,
    CustomToolFormat,
    CustomToolFormatType,
    FunctionTool,
    InferenceItem,
    InferenceRequest,
    InstructionItem,
    InstructionOrigin,
    InstructionPlacement,
    MessageContent,
    MessageItem,
    MessageRole,
    ReasoningItem,
    RefusalContent,
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
)
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort
from free_claude_code.core.replay_envelope import (
    ReplayEnvelopeError,
    decode_replay_envelope,
)
from free_claude_code.core.trace import trace_event

from .errors import ResponsesConversionError
from .ids import new_call_id
from .models import OpenAIResponsesRequest, ResponsesPresentationSnapshot

_AMBIENT_HOSTED_TOOL_TYPES = frozenset(
    {"web_search", "image_generation", "tool_search"}
)


@dataclass(frozen=True, slots=True)
class ResponsesIngressResult:
    """Canonical request plus the minimal validated public echo state."""

    request: InferenceRequest
    presentation: ResponsesPresentationSnapshot


@dataclass(slots=True)
class _PendingReasoning:
    text_parts: list[str] = field(default_factory=list)
    artifacts: list[ReplayArtifact] = field(default_factory=list)

    def add(self, item: Mapping[str, object], *, path: str) -> None:
        self.text_parts.extend(_reasoning_text_parts(item, path=path))
        encrypted = item.get("encrypted_content")
        if encrypted is None:
            return
        if not isinstance(encrypted, str):
            raise ResponsesConversionError(f"{path}.encrypted_content must be a string")
        try:
            decoded = decode_replay_envelope(
                encrypted,
                attachment=ReplayAttachment.REASONING,
            )
        except ReplayEnvelopeError as exc:
            raise ResponsesConversionError(f"{path}.encrypted_content: {exc}") from exc
        if decoded is not None:
            self.artifacts.extend(decoded)
        elif encrypted:
            self.artifacts.append(
                ReplayArtifact(
                    origin=ReplayArtifactOrigin.OPENAI,
                    kind=ReplayArtifactKind.ENCRYPTED_REASONING,
                    attachment=ReplayAttachment.REASONING,
                    payload=encrypted,
                )
            )

    @property
    def pending(self) -> bool:
        return bool(self.text_parts or self.artifacts)

    def take(self, *, turn_id: str) -> ReasoningItem | None:
        if not self.pending:
            return None
        item = ReasoningItem(
            turn_id=turn_id,
            reasoning="\n".join(self.text_parts),
            artifacts=tuple(self.artifacts),
        )
        self.text_parts.clear()
        self.artifacts.clear()
        return item


@dataclass(slots=True)
class _TranscriptBuilder:
    items: list[InferenceItem] = field(default_factory=list)
    system: list[InstructionItem] = field(default_factory=list)
    pending_reasoning: _PendingReasoning = field(default_factory=_PendingReasoning)
    quarantined_call_ids: set[str] = field(default_factory=set)
    _next_turn: int = 0
    _last_assistant_tool_turn: str | None = None
    _last_user_tool_turn: str | None = None

    def new_turn(self) -> str:
        turn_id = f"turn_{self._next_turn}"
        self._next_turn += 1
        return turn_id

    def break_tool_groups(self) -> None:
        self._last_assistant_tool_turn = None
        self._last_user_tool_turn = None

    def flush_reasoning(self) -> None:
        if self.pending_reasoning.pending:
            item = self.pending_reasoning.take(turn_id=self.new_turn())
            if item is not None:
                self.items.append(item)
        self.break_tool_groups()

    def assistant_tool_turn(self) -> str:
        if self._last_assistant_tool_turn is None:
            self._last_assistant_tool_turn = self.new_turn()
        self._last_user_tool_turn = None
        return self._last_assistant_tool_turn

    def user_tool_turn(self) -> str:
        if self._last_user_tool_turn is None:
            self._last_user_tool_turn = self.new_turn()
        self._last_assistant_tool_turn = None
        return self._last_user_tool_turn


def responses_to_inference_request(
    wire: OpenAIResponsesRequest,
) -> ResponsesIngressResult:
    """Validate and convert one public Responses request exactly once."""

    _validate_top_level(wire)
    builder = _TranscriptBuilder()
    if wire.instructions is not None:
        builder.system.append(
            InstructionItem(
                text=wire.instructions,
                origin=InstructionOrigin.SYSTEM,
                placement=InstructionPlacement.TOP_LEVEL,
            )
        )
    for index, item in enumerate(_input_items(wire.input)):
        _append_item(item, builder=builder, path=f"input[{index}]")
    builder.flush_reasoning()
    if not builder.items:
        raise ResponsesConversionError("Responses request input must contain a message")

    tool_choice = _tool_choice(wire.tool_choice)
    tools = _tools(wire.tools, raw_choice=wire.tool_choice)
    if (
        tool_choice is not None
        and tool_choice.mode is ToolChoiceMode.REQUIRED
        and not tools
    ):
        raise ResponsesConversionError(
            "tool_choice 'required' needs at least one executable function or custom tool"
        )
    metadata = (
        _json_object(wire.metadata, path="metadata")
        if wire.metadata is not None
        else None
    )
    request = InferenceRequest(
        model=_required_str(wire.model, path="model"),
        items=(*builder.system, *builder.items),
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=wire.parallel_tool_calls,
        max_output_tokens=wire.max_output_tokens,
        temperature=wire.temperature,
        top_p=wire.top_p,
        reasoning=_reasoning_intent(wire.reasoning),
        metadata=metadata,
    )
    return ResponsesIngressResult(
        request=request,
        presentation=ResponsesPresentationSnapshot(
            model=request.model,
            parallel_tool_calls=(
                True if wire.parallel_tool_calls is None else wire.parallel_tool_calls
            ),
            tool_choice=_presentation_tool_choice(wire.tool_choice),
            temperature=wire.temperature,
            top_p=wire.top_p,
            max_output_tokens=wire.max_output_tokens,
        ),
    )


def validate_responses_field_policy(wire: OpenAIResponsesRequest) -> None:
    """Apply protocol-wide rejection before routing or provider I/O."""

    _validate_top_level(wire)
    _tool_choice(wire.tool_choice)
    _tools(wire.tools, raw_choice=wire.tool_choice)


def _validate_top_level(wire: OpenAIResponsesRequest) -> None:
    if wire.model_extra:
        key = sorted(str(key) for key in wire.model_extra)[0]
        raise ResponsesConversionError(f"request.{key} is not supported")
    if wire.stream is False:
        raise ResponsesConversionError(
            "stream=false is not supported; omit stream or set stream=true"
        )
    if wire.store is True:
        raise ResponsesConversionError("store=true is not supported")
    if wire.previous_response_id is not None:
        raise ResponsesConversionError("previous_response_id must be null or omitted")
    _validate_include(wire.include)
    _validate_prompt_cache_key(wire.prompt_cache_key)
    _reasoning_intent(wire.reasoning)


def _validate_include(value: object) -> None:
    if value is None:
        return
    if value != ["reasoning.encrypted_content"]:
        raise ResponsesConversionError(
            "include must be exactly ['reasoning.encrypted_content'] when provided"
        )


def _validate_prompt_cache_key(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ResponsesConversionError(
            "prompt_cache_key must be a non-empty string when provided"
        )


def _reasoning_intent(value: Mapping[str, object] | None) -> ClientReasoningIntent:
    if value is None:
        return ClientReasoningIntent()
    unknown = set(value) - {"effort", "summary"}
    if unknown:
        raise ResponsesConversionError(
            f"reasoning.{sorted(str(key) for key in unknown)[0]} is not supported"
        )
    summary = value.get("summary")
    if summary is not None and summary != "auto":
        raise ResponsesConversionError("reasoning.summary must be null or 'auto'")
    raw_effort = value.get("effort")
    if raw_effort is None:
        return ClientReasoningIntent()
    if not isinstance(raw_effort, str) or not raw_effort.strip():
        raise ResponsesConversionError("reasoning.effort must be a non-empty string")
    normalized = raw_effort.strip().lower()
    if normalized == "none":
        return ClientReasoningIntent(control=ReasoningControl.OFF)
    try:
        effort = ReasoningEffort(normalized)
    except ValueError as exc:
        raise ResponsesConversionError(
            f"reasoning.effort has unsupported value {raw_effort!r}"
        ) from exc
    return ClientReasoningIntent(control=ReasoningControl.ON, effort=effort)


def _input_items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _append_item(
    value: object,
    *,
    builder: _TranscriptBuilder,
    path: str,
) -> None:
    if isinstance(value, str):
        builder.flush_reasoning()
        turn_id = builder.new_turn()
        builder.items.append(
            MessageItem(turn_id, MessageRole.USER, (TextContent(value),))
        )
        return
    item = _mapping(value, path=path)
    item_type = item.get("type")
    if item_type in {None, "message"} or "role" in item:
        role = _required_str(item.get("role", "user"), path=f"{path}.role")
        _append_message(
            role,
            item.get("content", ""),
            builder=builder,
            path=path,
        )
        return
    if item_type == "reasoning":
        builder.pending_reasoning.add(item, path=path)
        return
    if item_type in {"function_call", "custom_tool_call"}:
        _append_tool_call(item, item_type=item_type, builder=builder, path=path)
        return
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        _append_tool_result(item, item_type=item_type, builder=builder, path=path)
        return
    if item_type in {"input_text", "output_text", "text"}:
        builder.flush_reasoning()
        turn_id = builder.new_turn()
        builder.items.append(
            MessageItem(
                turn_id,
                MessageRole.USER,
                (TextContent(_text_from_part(item, path=path)),),
            )
        )
        return
    raise ResponsesConversionError(f"{path}.type {item_type!r} is not supported")


def _append_message(
    role: str,
    content: object,
    *,
    builder: _TranscriptBuilder,
    path: str,
) -> None:
    if role in {"system", "developer"}:
        builder.flush_reasoning()
        text = _instruction_text(content, path=f"{path}.content")
        if text:
            turn_id = builder.new_turn()
            builder.items.append(
                InstructionItem(
                    text=text,
                    origin=(
                        InstructionOrigin.DEVELOPER
                        if role == "developer"
                        else InstructionOrigin.SYSTEM
                    ),
                    placement=InstructionPlacement.TRANSCRIPT,
                    turn_id=turn_id,
                )
            )
        builder.break_tool_groups()
        return
    if role not in {"user", "assistant"}:
        raise ResponsesConversionError(f"{path}.role {role!r} is not supported")
    if role == "user":
        builder.flush_reasoning()
    turn_id = builder.new_turn()
    if role == "assistant" and (
        reasoning := builder.pending_reasoning.take(turn_id=turn_id)
    ):
        builder.items.append(reasoning)
    parts = _message_content(content, path=f"{path}.content")
    builder.items.append(MessageItem(turn_id, MessageRole(role), parts))
    builder.break_tool_groups()


def _append_tool_call(
    item: Mapping[str, object],
    *,
    item_type: object,
    builder: _TranscriptBuilder,
    path: str,
) -> None:
    call_id = _call_id(item)
    namespace = _optional_str(item.get("namespace"), path=f"{path}.namespace")
    name = _required_str(item.get("name"), path=f"{path}.name")
    if item_type == "function_call":
        try:
            tool_input = _function_arguments(
                item.get("arguments"), path=f"{path}.arguments"
            )
        except ResponsesConversionError as exc:
            builder.quarantined_call_ids.add(call_id)
            trace_event(
                stage="responses",
                event="responses.input.function_call_quarantined",
                source="openai_responses",
                call_id=call_id,
                error_type=type(exc).__name__,
            )
            return
        kind = ToolCallKind.FUNCTION
    else:
        tool_input = _custom_input(item.get("input"), path=f"{path}.input")
        kind = ToolCallKind.CUSTOM
    turn_id = builder.assistant_tool_turn()
    if reasoning := builder.pending_reasoning.take(turn_id=turn_id):
        builder.items.append(reasoning)
    builder.items.append(
        ToolCallItem(
            turn_id=turn_id,
            call_id=call_id,
            kind=kind,
            name=name,
            namespace=namespace,
            input=tool_input,
        )
    )


def _append_tool_result(
    item: Mapping[str, object],
    *,
    item_type: object,
    builder: _TranscriptBuilder,
    path: str,
) -> None:
    call_id = _call_id(item)
    if item_type == "function_call_output" and call_id in builder.quarantined_call_ids:
        return
    if builder.pending_reasoning.pending:
        turn_id = builder._last_assistant_tool_turn or builder.new_turn()
        if reasoning := builder.pending_reasoning.take(turn_id=turn_id):
            builder.items.append(reasoning)
    turn_id = builder.user_tool_turn()
    builder.items.append(
        ToolResultItem(
            turn_id=turn_id,
            call_id=call_id,
            content=_json_value(item.get("output", ""), path=f"{path}.output"),
        )
    )


def _message_content(value: object, *, path: str) -> tuple[MessageContent, ...]:
    if isinstance(value, str):
        return (TextContent(value),)
    if isinstance(value, Mapping):
        return (_content_part(_mapping(value, path=path), path=path),)
    if not isinstance(value, list):
        raise ResponsesConversionError(f"{path} must be text or a list")
    parts: list[MessageContent] = []
    for index, raw_part in enumerate(value):
        part_path = f"{path}[{index}]"
        if isinstance(raw_part, str):
            parts.append(TextContent(raw_part))
        else:
            parts.append(
                _content_part(_mapping(raw_part, path=part_path), path=part_path)
            )
    return tuple(parts)


def _content_part(part: Mapping[str, object], *, path: str) -> MessageContent:
    part_type = part.get("type")
    if part_type in {"input_text", "output_text", "text"}:
        return TextContent(_text_from_part(part, path=path))
    if part_type == "refusal":
        return RefusalContent(
            _required_str(part.get("refusal"), path=f"{path}.refusal")
        )
    raise ResponsesConversionError(f"{path}.type {part_type!r} is not supported")


def _instruction_text(value: object, *, path: str) -> str:
    parts = _message_content(value, path=path)
    text: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, TextContent):
            raise ResponsesConversionError(
                f"{path}[{index}] instruction content must be text"
            )
        text.append(part.text)
    return "\n\n".join(text)


def _reasoning_text_parts(item: Mapping[str, object], *, path: str) -> list[str]:
    content = _typed_text_list(
        item.get("content"),
        item_type="reasoning_text",
        path=f"{path}.content",
    )
    if content:
        return content
    return _typed_text_list(
        item.get("summary"),
        item_type="summary_text",
        path=f"{path}.summary",
    )


def _typed_text_list(value: object, *, item_type: str, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ResponsesConversionError(f"{path} must be a list")
    parts: list[str] = []
    for index, raw_part in enumerate(value):
        part = _mapping(raw_part, path=f"{path}[{index}]")
        if part.get("type") != item_type:
            raise ResponsesConversionError(
                f"{path}[{index}].type must be {item_type!r}"
            )
        parts.append(_required_str(part.get("text"), path=f"{path}[{index}].text"))
    return parts


def _tools(
    value: list[dict[str, object]] | None,
    *,
    raw_choice: object,
) -> tuple[FunctionTool | CustomTool, ...]:
    if value is None:
        return ()
    tools: list[FunctionTool | CustomTool] = []
    ambient_allowed = raw_choice is None or raw_choice == "auto"
    for index, raw_tool in enumerate(value):
        path = f"tools[{index}]"
        tool = _mapping(raw_tool, path=path)
        tool_type = tool.get("type")
        if tool_type in _AMBIENT_HOSTED_TOOL_TYPES:
            if ambient_allowed:
                continue
            if raw_choice == "none":
                continue
            continue
        if tool_type == "function":
            tools.append(_function_tool(tool, namespace=None, path=path))
        elif tool_type == "custom":
            tools.append(_custom_tool(tool, namespace=None, path=path))
        elif tool_type == "namespace":
            tools.extend(_namespace_tools(tool, path=path))
        else:
            raise ResponsesConversionError(
                f"{path}.type {tool_type!r} is not supported"
            )
    return tuple(tools)


def _function_tool(
    tool: Mapping[str, object],
    *,
    namespace: str | None,
    path: str,
) -> FunctionTool:
    nested = tool.get("function")
    source = _mapping(nested, path=f"{path}.function") if nested is not None else tool
    name = _required_str(source.get("name"), path=f"{path}.name")
    raw_schema = source.get("parameters", {"type": "object", "properties": {}})
    schema = _json_object(raw_schema, path=f"{path}.parameters")
    strict = source.get("strict", False)
    if not isinstance(strict, bool):
        raise ResponsesConversionError(f"{path}.strict must be boolean")
    return FunctionTool(
        name=name,
        description=_optional_str(
            source.get("description"), path=f"{path}.description"
        ),
        input_schema=schema,
        strict=strict,
        namespace=namespace,
    )


def _custom_tool(
    tool: Mapping[str, object],
    *,
    namespace: str | None,
    path: str,
) -> CustomTool:
    nested = tool.get("custom")
    source = _mapping(nested, path=f"{path}.custom") if nested is not None else tool
    name = _required_str(source.get("name"), path=f"{path}.name")
    raw_format = source.get("format")
    if raw_format is None:
        custom_format = CustomToolFormat(CustomToolFormatType.TEXT)
    else:
        format_value = _mapping(raw_format, path=f"{path}.format")
        format_type = format_value.get("type")
        if format_type == "text":
            custom_format = CustomToolFormat(CustomToolFormatType.TEXT)
        elif format_type == "grammar":
            custom_format = CustomToolFormat(
                CustomToolFormatType.GRAMMAR,
                syntax=_optional_str(
                    format_value.get("syntax"),
                    path=f"{path}.format.syntax",
                ),
                definition=_optional_str(
                    format_value.get("definition"),
                    path=f"{path}.format.definition",
                ),
            )
        else:
            raise ResponsesConversionError(
                f"{path}.format.type {format_type!r} is not supported"
            )
    return CustomTool(
        name=name,
        description=_optional_str(
            source.get("description"), path=f"{path}.description"
        ),
        format=custom_format,
        namespace=namespace,
    )


def _namespace_tools(
    tool: Mapping[str, object], *, path: str
) -> list[FunctionTool | CustomTool]:
    namespace = _required_str(tool.get("name"), path=f"{path}.name")
    nested = tool.get("tools")
    if not isinstance(nested, list):
        raise ResponsesConversionError(f"{path}.tools must be a list")
    result: list[FunctionTool | CustomTool] = []
    for index, raw_tool in enumerate(nested):
        nested_path = f"{path}.tools[{index}]"
        nested_tool = _mapping(raw_tool, path=nested_path)
        if nested_tool.get("type") == "function":
            result.append(
                _function_tool(nested_tool, namespace=namespace, path=nested_path)
            )
        elif nested_tool.get("type") == "custom":
            result.append(
                _custom_tool(nested_tool, namespace=namespace, path=nested_path)
            )
        else:
            raise ResponsesConversionError(
                f"{nested_path}.type {nested_tool.get('type')!r} is not supported"
            )
    return result


def _tool_choice(value: object) -> ToolChoice | None:
    if value is None or value == "auto":
        return None if value is None else ToolChoice(ToolChoiceMode.AUTO)
    if value == "none":
        return ToolChoice(ToolChoiceMode.NONE)
    if value == "required":
        return ToolChoice(ToolChoiceMode.REQUIRED)
    choice = _mapping(value, path="tool_choice")
    choice_type = choice.get("type")
    if choice_type in _AMBIENT_HOSTED_TOOL_TYPES:
        raise ResponsesConversionError(
            f"tool_choice.type {choice_type!r} cannot be selected explicitly"
        )
    if choice_type in {"auto"}:
        return ToolChoice(ToolChoiceMode.AUTO)
    if choice_type in {"any", "required"}:
        return ToolChoice(ToolChoiceMode.REQUIRED)
    if choice_type in {"function", "custom", "tool"}:
        nested = choice.get(str(choice_type))
        source = (
            _mapping(nested, path=f"tool_choice.{choice_type}")
            if nested is not None
            else choice
        )
        name = _required_str(source.get("name"), path="tool_choice.name")
        namespace = _optional_str(
            source.get("namespace", choice.get("namespace")),
            path="tool_choice.namespace",
        )
        return ToolChoice(
            ToolChoiceMode.SPECIFIC,
            kind=(
                ToolCallKind.CUSTOM
                if choice_type == "custom"
                else ToolCallKind.FUNCTION
            ),
            name=name,
            namespace=namespace,
        )
    raise ResponsesConversionError(f"tool_choice.type {choice_type!r} is not supported")


def _presentation_tool_choice(value: object) -> object:
    if value is None:
        return "auto"
    return _json_value(value, path="tool_choice")


def _function_arguments(value: object, *, path: str) -> dict[str, JsonValue]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return _json_object(value, path=path)
    if not isinstance(value, str):
        raise ResponsesConversionError(f"{path} must be JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ResponsesConversionError(f"{path} is invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ResponsesConversionError(f"{path} must decode to an object")
    return _json_object(parsed, path=path)


def _custom_input(value: object, *, path: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(
        _json_value(value, path=path),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _call_id(item: Mapping[str, object]) -> str:
    for key in ("call_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return new_call_id()


def _text_from_part(part: Mapping[str, object], *, path: str) -> str:
    for key in ("text", "input_text", "output_text"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    raise ResponsesConversionError(f"{path} requires a text value")


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ResponsesConversionError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _required_str(value: object, *, path: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ResponsesConversionError(f"{path} must be a non-empty string")


def _optional_str(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResponsesConversionError(f"{path} must be a string")
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
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ResponsesConversionError(f"{path} object keys must be strings")
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ResponsesConversionError(f"{path} must contain JSON-compatible values")
