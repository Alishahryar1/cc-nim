"""Strict Anthropic web server-tool request planning."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import (
    Message,
    MessagesRequest,
    Tool,
    get_block_type,
)

from .egress import WebDomainPolicy, WebDomainRule, parse_web_domain_rule
from .parsers import collect_urls

type WebServerToolKind = Literal["web_search", "web_fetch"]
type WebToolChoice = Literal["auto", "any", "tool", "none"]

WEB_SEARCH_TYPE = "web_search_20250305"
WEB_FETCH_TYPE = "web_fetch_20250910"
HIDDEN_WEB_SEARCH_NAME = "fcc_web_search"
HIDDEN_WEB_FETCH_NAME = "fcc_web_fetch"

_SUPPORTED_TYPES: dict[str, tuple[WebServerToolKind, str]] = {
    WEB_SEARCH_TYPE: ("web_search", HIDDEN_WEB_SEARCH_NAME),
    WEB_FETCH_TYPE: ("web_fetch", HIDDEN_WEB_FETCH_NAME),
}
_SERVER_TYPE_PREFIXES = ("web_search_", "web_fetch_")
_SERVER_HISTORY_TYPES = {
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
}
_TOOL_OPTION_FIELDS = {"max_uses", "allowed_domains", "blocked_domains"}
_CHOICE_FIELDS = {"type", "name", "disable_parallel_tool_use"}


@dataclass(frozen=True, slots=True)
class WebServerToolSpec:
    """Validated definition for one supported Anthropic server tool."""

    kind: WebServerToolKind
    protocol_type: str
    hidden_name: str
    max_uses: int | None
    domains: WebDomainPolicy


@dataclass(frozen=True, slots=True)
class PendingWebServerCall:
    """One validated public server call awaiting local execution."""

    public_id: str
    spec: WebServerToolSpec
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class _NormalizedTranscript:
    messages: tuple[Message, ...]
    pending: tuple[PendingWebServerCall, ...]
    provenance: frozenset[str]


@dataclass(frozen=True, slots=True)
class WebServerToolPlan:
    """One request's validated server-tool translation."""

    request: MessagesRequest
    specs: tuple[WebServerToolSpec, ...]
    choice: WebToolChoice
    forced_kind: WebServerToolKind | None
    has_history: bool
    pending: tuple[PendingWebServerCall, ...]
    provenance: frozenset[str]

    @property
    def active(self) -> bool:
        return self.choice != "none" or self.has_history

    def spec_for_hidden_name(self, name: str) -> WebServerToolSpec | None:
        return next((spec for spec in self.specs if spec.hidden_name == name), None)

    def spec_for_kind(self, kind: WebServerToolKind) -> WebServerToolSpec | None:
        return next((spec for spec in self.specs if spec.kind == kind), None)


def plan_web_server_tools(
    request: MessagesRequest,
    *,
    enabled: bool,
) -> WebServerToolPlan | None:
    """Validate and translate supported server tools, or return ``None``."""
    raw_tools = request.tools or []
    server_tools: list[Tool] = []
    ordinary_tools: list[Tool] = []
    for tool in raw_tools:
        tool_type = tool.type
        if tool_type in _SUPPORTED_TYPES:
            server_tools.append(tool)
            continue
        if isinstance(tool_type, str) and (
            tool_type in {"web_search", "web_fetch"}
            or tool_type.startswith(_SERVER_TYPE_PREFIXES)
        ):
            raise InvalidRequestError(
                f"Unsupported Anthropic server tool type {tool_type!r}. FCC supports "
                f"{WEB_SEARCH_TYPE!r} and {WEB_FETCH_TYPE!r}."
            )
        ordinary_tools.append(tool)

    has_history = has_supported_server_tool_history(request)
    if not server_tools:
        if has_history:
            raise InvalidRequestError(
                "Server-tool history requires the matching supported server tool "
                "definition in the current request."
            )
        return None
    if not enabled:
        raise InvalidRequestError(
            "Anthropic web server tools are disabled. Enable web access in FCC or "
            "remove the server tool from the request."
        )
    if ordinary_tools:
        raise InvalidRequestError(
            "Mixing Anthropic server tools with ordinary client tools is not "
            "supported in one request."
        )

    specs = tuple(_parse_server_tool(tool) for tool in server_tools)
    kinds = [spec.kind for spec in specs]
    if len(set(kinds)) != len(kinds):
        raise InvalidRequestError(
            "Each Anthropic web server tool may be declared once."
        )

    choice, forced_kind, translated_choice = _translate_tool_choice(request, specs)
    normalized = _normalize_transcript(request, specs)
    if choice == "none" and normalized.pending:
        raise InvalidRequestError(
            "tool_choice none cannot resume a pending server-tool call."
        )
    translated_tools = (
        None if choice == "none" else [_hidden_tool(spec) for spec in specs]
    )
    translated_request = request.model_copy(
        update={
            "messages": list(normalized.messages),
            "tools": translated_tools,
            "tool_choice": translated_choice,
        },
        deep=True,
    )
    return WebServerToolPlan(
        request=translated_request,
        specs=specs,
        choice=choice,
        forced_kind=forced_kind,
        has_history=has_history,
        pending=normalized.pending,
        provenance=normalized.provenance,
    )


def has_supported_server_tool_history(request: MessagesRequest) -> bool:
    """Return whether the transcript contains public server-tool blocks."""
    return any(
        get_block_type(block) in _SERVER_HISTORY_TYPES
        for message in request.messages
        if isinstance(message.content, list)
        for block in message.content
    )


def _parse_server_tool(tool: Tool) -> WebServerToolSpec:
    tool_type = tool.type
    if tool_type not in _SUPPORTED_TYPES:
        raise InvalidRequestError("Unsupported Anthropic server tool definition.")
    kind, hidden_name = _SUPPORTED_TYPES[tool_type]
    if tool.name != kind:
        raise InvalidRequestError(
            f"Server tool type {tool_type!r} must use the canonical name {kind!r}."
        )
    if tool.description is not None or tool.input_schema is not None:
        raise InvalidRequestError(
            f"Server tool {kind!r} must not define description or input_schema."
        )

    options = tool.model_extra or {}
    unsupported = sorted(set(options) - _TOOL_OPTION_FIELDS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise InvalidRequestError(
            f"Server tool {kind!r} contains unsupported option(s): {joined}."
        )

    raw_max_uses = options.get("max_uses")
    if raw_max_uses is not None and (
        isinstance(raw_max_uses, bool)
        or not isinstance(raw_max_uses, int)
        or raw_max_uses <= 0
    ):
        raise InvalidRequestError(f"{kind}.max_uses must be a positive integer.")

    allowed = _parse_domains(
        options.get("allowed_domains"),
        field=f"{kind}.allowed_domains",
        allow_path=kind == "web_search",
    )
    blocked = _parse_domains(
        options.get("blocked_domains"),
        field=f"{kind}.blocked_domains",
        allow_path=kind == "web_search",
    )
    if allowed and blocked:
        raise InvalidRequestError(
            f"{kind} may define allowed_domains or blocked_domains, not both."
        )
    return WebServerToolSpec(
        kind=kind,
        protocol_type=tool_type,
        hidden_name=hidden_name,
        max_uses=raw_max_uses,
        domains=WebDomainPolicy(allowed=allowed, blocked=blocked),
    )


def _parse_domains(
    value: object,
    *,
    field: str,
    allow_path: bool,
) -> tuple[WebDomainRule, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise InvalidRequestError(f"{field} must be a non-empty list of domains.")
    rules: list[WebDomainRule] = []
    for entry in value:
        if not isinstance(entry, str):
            raise InvalidRequestError(f"{field} entries must be strings.")
        try:
            rules.append(parse_web_domain_rule(entry, allow_path=allow_path))
        except ValueError as exc:
            raise InvalidRequestError(
                f"Invalid {field} entry {entry!r}: {exc}"
            ) from exc
    if len(set(rules)) != len(rules):
        raise InvalidRequestError(f"{field} must not contain duplicates.")
    return tuple(rules)


def _translate_tool_choice(
    request: MessagesRequest,
    specs: tuple[WebServerToolSpec, ...],
) -> tuple[WebToolChoice, WebServerToolKind | None, dict[str, object] | None]:
    raw = request.tool_choice
    if raw is None:
        return "auto", None, None
    unsupported = sorted(set(raw) - _CHOICE_FIELDS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise InvalidRequestError(
            f"tool_choice contains unsupported field(s): {joined}."
        )

    disable_parallel = raw.get("disable_parallel_tool_use", False)
    if not isinstance(disable_parallel, bool):
        raise InvalidRequestError(
            "tool_choice.disable_parallel_tool_use must be boolean."
        )
    if disable_parallel:
        raise InvalidRequestError(
            "tool_choice.disable_parallel_tool_use=true is not supported for local "
            "Anthropic server tools."
        )

    choice_type = raw.get("type")
    if choice_type not in {"auto", "any", "tool", "none"}:
        raise InvalidRequestError(
            "tool_choice.type must be auto, any, tool, or none for server tools."
        )
    if choice_type != "tool":
        if "name" in raw:
            raise InvalidRequestError(
                "tool_choice.name is valid only when tool_choice.type is 'tool'."
            )
        return (
            choice_type,
            None,
            None if choice_type == "none" else {"type": choice_type},
        )

    name = raw.get("name")
    if name not in {"web_search", "web_fetch"}:
        raise InvalidRequestError(
            "A forced server tool must name web_search or web_fetch."
        )
    kind: WebServerToolKind = name
    spec = next((item for item in specs if item.kind == kind), None)
    if spec is None:
        raise InvalidRequestError(
            f"tool_choice forces {kind!r}, but that server tool is not declared."
        )
    return "tool", kind, {"type": "tool", "name": spec.hidden_name}


def _hidden_tool(spec: WebServerToolSpec) -> Tool:
    field = "query" if spec.kind == "web_search" else "url"
    return Tool(
        name=spec.hidden_name,
        description=(
            "Search the public web for current information."
            if spec.kind == "web_search"
            else "Fetch one public URL already present in the conversation."
        ),
        input_schema={
            "type": "object",
            "properties": {field: {"type": "string"}},
            "required": [field],
            "additionalProperties": False,
        },
    )


def _normalize_transcript(
    request: MessagesRequest,
    specs: tuple[WebServerToolSpec, ...],
) -> _NormalizedTranscript:
    normalized: list[Message] = []
    provenance: set[str] = set()
    calls: dict[str, PendingWebServerCall] = {}
    completed: set[str] = set()
    pending_message_indexes: dict[str, int] = {}

    for message_index, message in enumerate(request.messages):
        if message.role == "user":
            provenance.update(collect_urls(message.content))
        if not isinstance(message.content, list):
            normalized.append(message)
            continue
        raw_blocks = [_block_dict(block) for block in message.content]
        if any(
            block.get("type") == "tool_use"
            and isinstance(name := block.get("name"), str)
            and _spec_for_hidden_name(specs, name)
            for block in raw_blocks
        ):
            raise InvalidRequestError(
                "Ordinary tool history uses a name reserved for FCC web server tools."
            )
        has_server_blocks = any(
            block.get("type")
            in {"server_tool_use", "web_search_tool_result", "web_fetch_tool_result"}
            for block in raw_blocks
        )
        if has_server_blocks and message.role != "assistant":
            raise InvalidRequestError(
                "Anthropic server-tool history blocks must use the assistant role."
            )
        if not has_server_blocks:
            normalized.append(message)
            continue

        assistant_blocks: list[dict[str, object]] = []
        result_blocks: list[dict[str, object]] = []
        first_assistant_segment = True

        for block in raw_blocks:
            block_type = block.get("type")
            if block_type in {"web_search_tool_result", "web_fetch_tool_result"}:
                first_assistant_segment = _flush_assistant_segment(
                    normalized,
                    assistant_blocks,
                    reasoning_content=message.reasoning_content,
                    first_segment=first_assistant_segment,
                )
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str) or tool_use_id not in calls:
                    raise InvalidRequestError(
                        "Server-tool history contains a result without a matching call."
                    )
                call = calls[tool_use_id]
                expected_result_type = f"{call.spec.kind}_tool_result"
                if block_type != expected_result_type:
                    raise InvalidRequestError(
                        "Server-tool history pairs a call with the wrong result type."
                    )
                if tool_use_id in completed:
                    raise InvalidRequestError(
                        "Server-tool history contains a duplicate result."
                    )
                content = block.get("content")
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                )
                provenance.update(collect_urls(content))
                completed.add(tool_use_id)
                continue

            _flush_result_segment(normalized, result_blocks)
            if block_type == "server_tool_use":
                tool_use_id = block.get("id")
                name = block.get("name")
                tool_input = block.get("input")
                if (
                    not isinstance(tool_use_id, str)
                    or not tool_use_id
                    or tool_use_id in calls
                    or not isinstance(name, str)
                    or not isinstance(tool_input, dict)
                ):
                    raise InvalidRequestError("Malformed server-tool history call.")
                spec = _spec_for_kind(specs, name)
                if spec is None:
                    raise InvalidRequestError(
                        "Server-tool history references a tool not declared in this "
                        "request."
                    )
                arguments = _object_dict(tool_input)
                calls[tool_use_id] = PendingWebServerCall(
                    public_id=tool_use_id,
                    spec=spec,
                    arguments=arguments,
                )
                pending_message_indexes[tool_use_id] = message_index
                assistant_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": spec.hidden_name,
                        "input": arguments,
                    }
                )
            else:
                assistant_blocks.append(block)
        _flush_assistant_segment(
            normalized,
            assistant_blocks,
            reasoning_content=message.reasoning_content,
            first_segment=first_assistant_segment,
        )
        _flush_result_segment(normalized, result_blocks)

    pending = tuple(call for call_id, call in calls.items() if call_id not in completed)
    if pending and any(
        pending_message_indexes[call.public_id] != len(request.messages) - 1
        for call in pending
    ):
        raise InvalidRequestError(
            "Pending server-tool calls must be at the end of the transcript."
        )
    if pending and _has_pending_ordinary_tool_call(normalized, pending):
        raise InvalidRequestError(
            "A transcript cannot resume pending server and client tools together."
        )
    return _NormalizedTranscript(
        messages=tuple(normalized),
        pending=pending,
        provenance=frozenset(provenance),
    )


def _flush_assistant_segment(
    normalized: list[Message],
    blocks: list[dict[str, object]],
    *,
    reasoning_content: str | None,
    first_segment: bool,
) -> bool:
    if not blocks:
        return first_segment
    normalized.append(
        Message.model_validate(
            {
                "role": "assistant",
                "content": list(blocks),
                "reasoning_content": reasoning_content if first_segment else None,
            }
        )
    )
    blocks.clear()
    return False


def _flush_result_segment(
    normalized: list[Message],
    blocks: list[dict[str, object]],
) -> None:
    if not blocks:
        return
    normalized.append(Message.model_validate({"role": "user", "content": list(blocks)}))
    blocks.clear()


def _has_pending_ordinary_tool_call(
    messages: list[Message],
    pending_server: tuple[PendingWebServerCall, ...],
) -> bool:
    server_ids = {call.public_id for call in pending_server}
    calls: set[str] = set()
    results: set[str] = set()
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for raw in message.content:
            block = _block_dict(raw)
            if block.get("type") == "tool_use":
                tool_id = block.get("id")
                if isinstance(tool_id, str) and tool_id not in server_ids:
                    calls.add(tool_id)
            elif block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                if isinstance(tool_id, str):
                    results.add(tool_id)
    return bool(calls - results)


def _spec_for_hidden_name(
    specs: tuple[WebServerToolSpec, ...], name: str
) -> WebServerToolSpec | None:
    return next((spec for spec in specs if spec.hidden_name == name), None)


def _spec_for_kind(
    specs: tuple[WebServerToolSpec, ...], kind: str
) -> WebServerToolSpec | None:
    return next((spec for spec in specs if spec.kind == kind), None)


def _block_dict(block: object) -> dict[str, object]:
    if isinstance(block, BaseModel):
        return _object_dict(block.model_dump(exclude_none=True))
    if isinstance(block, dict):
        return _object_dict(block)
    raise InvalidRequestError("Messages contain a malformed content block.")


def _object_dict(value: Mapping[object, object]) -> dict[str, object]:
    return {str(key): item for key, item in value.items()}
