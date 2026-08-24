"""OpenAI-chat tool-call assembly helpers."""

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from free_claude_code.core.inference import (
    CustomTool,
    InferenceEvent,
    InferenceRequest,
    InferenceStreamLedger,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
    ToolCallKind,
    ToolCallState,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.openai_chat.recovery import (
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from free_claude_code.providers.openai_compat import (
    OpenAIToolIdentity,
    OpenAIToolNameCodec,
)

RecordToolExtraContent = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class _CollectedToolCall:
    index: int
    tool_id: str | None = None
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)
    extra_content: dict[str, Any] | None = None


class OpenAIToolCallCollector:
    """Collect and validate one buffered OpenAI tool-call response."""

    def __init__(self) -> None:
        self._calls: dict[int, _CollectedToolCall] = {}

    @property
    def has_calls(self) -> bool:
        return bool(self._calls)

    def add(self, tool_call: Any) -> None:
        """Add one SDK tool-call delta without emitting downstream state."""
        raw_index = getattr(tool_call, "index", 0)
        index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else 0
        state = self._calls.setdefault(index, _CollectedToolCall(index=index))

        tool_id = getattr(tool_call, "id", None)
        if tool_id:
            state.tool_id = str(tool_id)

        function = getattr(tool_call, "function", None)
        incoming_name = getattr(function, "name", None)
        if isinstance(incoming_name, str) and incoming_name:
            state.name = _merge_tool_name(state.name, incoming_name)

        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, str) and arguments:
            state.argument_parts.append(arguments)

        extra_content = tool_call_extra_content(tool_call)
        if extra_content:
            state.extra_content = extra_content

    def completed_calls(
        self,
        request: InferenceRequest,
        *,
        tool_names: OpenAIToolNameCodec | None = None,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], ...] | None:
        """Return complete schema-valid calls, or None when output is incomplete."""
        schemas = tool_schemas_by_name(request)
        request_identities = {
            OpenAIToolIdentity(
                ToolCallKind.CUSTOM
                if isinstance(tool, CustomTool)
                else ToolCallKind.FUNCTION,
                tool.name,
                tool.namespace,
            )
            for tool in request.tools
        }
        completed: list[dict[str, Any]] = []
        for index in sorted(self._calls):
            state = self._calls[index]
            wire_name = state.name.strip()
            identity = (
                tool_names.decode_identity(wire_name)
                if tool_names is not None
                else OpenAIToolIdentity(ToolCallKind.FUNCTION, wire_name)
            )
            name = identity.name
            if not name or identity not in request_identities:
                return None
            arguments = "".join(state.argument_parts)
            aliases = (
                tool_argument_aliases.get(wire_name, {})
                if tool_argument_aliases is not None
                else {}
            )
            if aliases:
                restored = restore_tool_argument_aliases(arguments, aliases)
                if restored is None:
                    return None
                arguments = restored
            if parse_complete_tool_input(arguments, name, schemas) is None:
                return None

            call: dict[str, Any] = {
                "index": index,
                "id": state.tool_id,
                "function": {
                    "name": wire_name,
                    "arguments": arguments,
                },
            }
            if state.extra_content:
                call["extra_content"] = state.extra_content
            completed.append(call)
        return tuple(completed)


def iter_heuristic_tool_use_events(
    ledger: InferenceStreamLedger,
    tool_use: dict[str, Any],
    *,
    tool_names: OpenAIToolNameCodec | None = None,
) -> Iterator[InferenceEvent]:
    """Emit canonical events for one heuristic tool-use block."""
    name = tool_use.get("name")
    identity = OpenAIToolIdentity(ToolCallKind.FUNCTION, str(name or ""))
    if tool_names is not None and isinstance(name, str):
        identity = tool_names.decode_identity(name)
        if identity.name != name:
            tool_use = {**tool_use, "name": identity.name}
    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
        task_input = tool_use["input"]
        if task_input.get("run_in_background") is not False:
            task_input["run_in_background"] = False
    yield from ledger.close_content_blocks()
    tool_index = len(ledger.blocks.tool_states)
    yield ledger.start_tool_block(
        tool_index,
        str(tool_use["id"]),
        str(tool_use["name"]),
        kind=identity.kind,
        namespace=identity.namespace,
    )
    yield ledger.emit_tool_delta(tool_index, json.dumps(tool_use["input"]))
    yield ledger.stop_tool_block(tool_index)


def tool_call_extra_content(tool_call: Any) -> dict[str, Any] | None:
    """Return provider-specific extra tool-call metadata from OpenAI objects."""
    if isinstance(tool_call, dict):
        value = tool_call.get("extra_content")
        return value if isinstance(value, dict) else None

    value = getattr(tool_call, "extra_content", None)
    if isinstance(value, dict):
        return value

    model_extra = getattr(tool_call, "model_extra", None)
    if isinstance(model_extra, dict):
        value = model_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    pydantic_extra = getattr(tool_call, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict):
        value = pydantic_extra.get("extra_content")
        if isinstance(value, dict):
            return value

    return None


def has_generated_output(ledger: InferenceStreamLedger) -> bool:
    """Return whether one canonical assistant block has been generated."""
    return ledger.has_generated_output()


def started_tool_states(
    ledger: InferenceStreamLedger,
) -> list[tuple[int, ToolCallState]]:
    """Return started tool states in stream order."""
    return [
        (tool_index, state)
        for tool_index, state in ledger.blocks.tool_states.items()
        if state.started
    ]


def all_emitted_tools_complete(
    ledger: InferenceStreamLedger, request: InferenceRequest
) -> bool:
    """Return whether every emitted tool block has schema-valid input."""
    schemas = tool_schemas_by_name(request)
    tool_blocks = ledger.tool_blocks()
    if not tool_blocks:
        return False
    return all(
        block.call_id
        and block.name
        and parse_complete_tool_input(block.content, block.name, schemas) is not None
        for block in tool_blocks
    )


class OpenAIToolCallAssembler:
    """Assemble OpenAI tool-call deltas into canonical tool events."""

    def __init__(
        self,
        *,
        replay_scope: ReplayCompatibilityScope,
        record_extra_content: RecordToolExtraContent | None = None,
    ) -> None:
        self._replay_scope = replay_scope
        self._record_extra_content = record_extra_content
        self._task_arg_buffers: dict[int, str] = {}
        self._task_args_emitted: set[int] = set()

    def process_tool_call(
        self,
        tc: dict[str, Any],
        ledger: InferenceStreamLedger,
        *,
        tool_names: OpenAIToolNameCodec | None = None,
        tool_name_buffers: dict[int, str] | None = None,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[InferenceEvent]:
        """Process one tool-call delta and yield canonical events."""
        raw_index = tc.get("index", 0)
        tc_index = raw_index if isinstance(raw_index, int) else 0
        if tc_index < 0:
            tc_index = len(ledger.blocks.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        arguments = fn_delta.get("arguments", "") or ""

        if tc.get("id") is not None:
            ledger.blocks.set_stream_tool_id(tc_index, tc.get("id"))

        raw_extra_content = tc.get("extra_content")
        extra_content = (
            raw_extra_content
            if isinstance(raw_extra_content, dict) and raw_extra_content
            else None
        )
        if extra_content:
            ledger.set_tool_artifacts(
                tc_index,
                _tool_replay_artifacts(extra_content, scope=self._replay_scope),
            )

        if isinstance(incoming_name, str) and incoming_name:
            resolved_identity = _decode_streamed_tool_identity(
                incoming_name,
                tool_index=tc_index,
                tool_names=tool_names,
                buffers=tool_name_buffers,
            )
            if resolved_identity is not None:
                ledger.blocks.register_tool_identity(
                    tc_index,
                    resolved_identity.name,
                    kind=resolved_identity.kind,
                    namespace=resolved_identity.namespace,
                )

        state = ledger.blocks.tool_states.get(tc_index)
        resolved_id = (state.call_id if state and state.call_id else None) or tc.get(
            "id"
        )
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                start_artifacts = state.artifacts if state else ()
                if extra_content:
                    self._record_tool_call_extra_content(tool_id, extra_content)
                yield ledger.start_tool_block(
                    tc_index,
                    tool_id,
                    display_name,
                    kind=state.kind if state else ToolCallKind.FUNCTION,
                    namespace=state.namespace if state else None,
                    artifacts=start_artifacts,
                )
                state = ledger.blocks.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(
                        ledger,
                        tc_index,
                        pre,
                        tool_names=tool_names,
                        tool_argument_aliases=tool_argument_aliases,
                        tool_argument_alias_buffers=tool_argument_alias_buffers,
                    )

        state = ledger.blocks.tool_states.get(tc_index)
        if state is not None and state.call_id and extra_content:
            self._record_tool_call_extra_content(state.call_id, extra_content)
        if not arguments:
            return
        if state is None or not state.started:
            state = ledger.blocks.ensure_tool_state(tc_index)
            if not (resolved_name or "").strip():
                state.pre_start_args += arguments
                return

        yield from self._emit_tool_arg_delta(
            ledger,
            tc_index,
            arguments,
            tool_names=tool_names,
            tool_argument_aliases=tool_argument_aliases,
            tool_argument_alias_buffers=tool_argument_alias_buffers,
        )

    def flush_task_arg_buffers(
        self, ledger: InferenceStreamLedger
    ) -> Iterator[InferenceEvent]:
        """Emit buffered Task args as a single JSON delta."""
        for tool_index, buffered in list(self._task_arg_buffers.items()):
            if not buffered or tool_index in self._task_args_emitted:
                continue
            output = "{}"
            try:
                parsed = json.loads(buffered)
                if isinstance(parsed, dict):
                    _normalize_task_run_in_background(parsed)
                    output = json.dumps(parsed)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                state = ledger.blocks.tool_states.get(tool_index)
                digest = hashlib.sha256(
                    buffered.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
                logger.warning(
                    "Task args invalid JSON (id={} len={} buffer_sha256_prefix={}): {}",
                    state.call_id if state is not None and state.call_id else "unknown",
                    len(buffered),
                    digest,
                    exc,
                )
            self._task_args_emitted.add(tool_index)
            self._task_arg_buffers.pop(tool_index, None)
            yield ledger.emit_tool_delta(tool_index, output)

    def buffered_task_args(self, tool_index: int) -> str:
        """Return provider-owned Task arguments that have not been emitted yet."""
        return self._task_arg_buffers.get(tool_index, "")

    def flush_tool_name_buffers(
        self,
        ledger: InferenceStreamLedger,
        *,
        tool_names: OpenAIToolNameCodec,
        tool_name_buffers: dict[int, str],
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[InferenceEvent]:
        """Resolve names held only because they also prefix a generated alias."""
        for tool_index, name in list(tool_name_buffers.items()):
            tool_name_buffers.pop(tool_index, None)
            yield from self.process_tool_call(
                {
                    "index": tool_index,
                    "function": {"name": name, "arguments": ""},
                },
                ledger,
                tool_names=tool_names,
                tool_argument_aliases=tool_argument_aliases,
                tool_argument_alias_buffers=tool_argument_alias_buffers,
            )

    def flush_tool_argument_alias_buffers(
        self,
        ledger: InferenceStreamLedger,
        tool_names: OpenAIToolNameCodec,
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[InferenceEvent]:
        """Emit remaining aliased args without losing malformed JSON."""
        for tool_index, buffered_args in list(tool_argument_alias_buffers.items()):
            if not buffered_args:
                tool_argument_alias_buffers.pop(tool_index, None)
                continue
            state = ledger.blocks.tool_states.get(tool_index)
            if state is None or state.name == "Task":
                continue
            aliases = tool_argument_aliases_for_identity(
                tool_argument_aliases,
                tool_names=tool_names,
                kind=state.kind,
                name=state.name,
                namespace=state.namespace,
            )
            if not aliases:
                continue
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            yield ledger.emit_tool_delta(
                tool_index,
                restored if restored is not None else buffered_args,
            )
            tool_argument_alias_buffers.pop(tool_index, None)

    def _emit_tool_arg_delta(
        self,
        ledger: InferenceStreamLedger,
        tc_index: int,
        args: str,
        *,
        tool_names: OpenAIToolNameCodec | None,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[InferenceEvent]:
        """Emit one argument fragment for a started tool block."""
        if not args:
            return
        state = ledger.blocks.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            if tc_index in self._task_args_emitted:
                return
            buffered = self._task_arg_buffers.get(tc_index, "") + args
            self._task_arg_buffers[tc_index] = buffered
            try:
                parsed = json.loads(buffered)
            except json.JSONDecodeError, TypeError, ValueError:
                return
            if not isinstance(parsed, dict):
                return
            _normalize_task_run_in_background(parsed)
            self._task_args_emitted.add(tc_index)
            self._task_arg_buffers.pop(tc_index, None)
            yield ledger.emit_tool_delta(tc_index, json.dumps(parsed))
            return
        aliases = (
            tool_argument_aliases_for_identity(
                tool_argument_aliases,
                tool_names=tool_names,
                kind=state.kind,
                name=state.name,
                namespace=state.namespace,
            )
            if tool_argument_aliases
            else {}
        )
        if aliases:
            if tool_argument_alias_buffers is None:
                restored = self._restore_aliased_tool_arguments(args, aliases)
                if restored is not None:
                    yield ledger.emit_tool_delta(tc_index, restored)
                return

            buffered_args = tool_argument_alias_buffers.get(tc_index, "") + args
            restored = self._restore_aliased_tool_arguments(buffered_args, aliases)
            if restored is None:
                tool_argument_alias_buffers[tc_index] = buffered_args
                return
            tool_argument_alias_buffers.pop(tc_index, None)
            yield ledger.emit_tool_delta(tc_index, restored)
            return
        yield ledger.emit_tool_delta(tc_index, args)

    def _restore_aliased_tool_arguments(
        self, argument_json: str, aliases: dict[str, str]
    ) -> str | None:
        return restore_tool_argument_aliases(argument_json, aliases)

    def _record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        if self._record_extra_content is not None:
            self._record_extra_content(tool_call_id, extra_content)


def tool_argument_aliases_for_identity(
    aliases: dict[str, dict[str, str]],
    *,
    tool_names: OpenAIToolNameCodec | None,
    kind: ToolCallKind,
    name: str,
    namespace: str | None,
) -> dict[str, str]:
    """Resolve provider-private aliases by their exact upstream tool identity."""

    wire_name = (
        tool_names.encode(name, kind=kind, namespace=namespace)
        if tool_names is not None
        else name
    )
    return aliases.get(wire_name, {})


def restore_tool_argument_aliases(
    argument_json: str,
    aliases: dict[str, str],
) -> str | None:
    """Restore provider-private argument aliases in one complete JSON object."""
    try:
        parsed = json.loads(argument_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return argument_json
    return json.dumps(_restore_tool_argument_alias_value(parsed, aliases))


def _restore_tool_argument_alias_value(
    value: Any,
    aliases: dict[str, str],
) -> Any:
    if isinstance(value, dict):
        return {
            aliases.get(key, key): _restore_tool_argument_alias_value(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_tool_argument_alias_value(item, aliases) for item in value]
    return value


def _merge_tool_name(existing: str, incoming: str) -> str:
    if not existing or incoming.startswith(existing):
        return incoming
    if existing.startswith(incoming):
        return existing
    return "".join((existing, incoming))


def _normalize_task_run_in_background(args: JsonObject) -> None:
    if args.get("run_in_background") is not False:
        args["run_in_background"] = False


def _decode_streamed_tool_identity(
    incoming: str,
    *,
    tool_index: int,
    tool_names: OpenAIToolNameCodec | None,
    buffers: dict[int, str] | None,
) -> OpenAIToolIdentity | None:
    if tool_names is None:
        return OpenAIToolIdentity(ToolCallKind.FUNCTION, incoming)
    if not tool_names.has_aliases:
        return tool_names.decode_identity(incoming)
    if buffers is None:
        return tool_names.decode_identity(incoming)

    combined = _merge_tool_name(buffers.get(tool_index, ""), incoming)
    if tool_names.is_alias(combined):
        buffers.pop(tool_index, None)
        return tool_names.decode_identity(combined)
    if tool_names.is_alias_prefix(combined):
        buffers[tool_index] = combined
        return None
    buffers.pop(tool_index, None)
    return tool_names.decode_identity(combined)


def _tool_replay_artifacts(
    extra_content: dict[str, Any],
    *,
    scope: ReplayCompatibilityScope,
) -> tuple[ReplayArtifact, ...]:
    google = extra_content.get("google")
    if isinstance(google, dict):
        signature = google.get("thought_signature")
        if isinstance(signature, str) and signature:
            return (
                ReplayArtifact(
                    origin=ReplayArtifactOrigin.GOOGLE,
                    kind=ReplayArtifactKind.THOUGHT_SIGNATURE,
                    attachment=ReplayAttachment.TOOL_CALL,
                    payload=signature,
                    scope=scope,
                ),
            )
    return (
        ReplayArtifact(
            origin=ReplayArtifactOrigin.OPENAI_COMPATIBLE,
            kind=ReplayArtifactKind.TOOL_EXTRA_CONTENT,
            attachment=ReplayAttachment.TOOL_CALL,
            payload=extra_content,
            scope=scope,
        ),
    )
