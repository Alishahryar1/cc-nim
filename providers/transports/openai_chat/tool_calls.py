"""OpenAI-chat tool-call assembly helpers."""

import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from core.anthropic.streaming import (
    AnthropicStreamLedger,
    tool_schemas_by_name,
)

RecordToolExtraContent = Callable[[str, dict[str, Any]], None]


def iter_heuristic_tool_use_sse(
    ledger: AnthropicStreamLedger, tool_use: dict[str, Any]
) -> Iterator[str]:
    """Emit SSE for one heuristic tool_use block."""
    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
        task_input = tool_use["input"]
        if task_input.get("run_in_background") is not False:
            task_input["run_in_background"] = False
    yield from ledger.close_content_blocks()
    block_idx = ledger.blocks.allocate_index()
    yield ledger.content_block_start(
        block_idx,
        "tool_use",
        id=tool_use["id"],
        name=tool_use["name"],
    )
    yield ledger.content_block_delta(
        block_idx,
        "input_json_delta",
        json.dumps(tool_use["input"]),
    )
    yield ledger.content_block_stop(block_idx)


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


def has_committed_sse_output(ledger: AnthropicStreamLedger) -> bool:
    """Return whether any assistant content escaped the builder."""
    return (
        ledger.blocks.text_index != -1
        or ledger.blocks.thinking_index != -1
        or ledger.has_emitted_tool_block()
    )


def started_tool_states(ledger: AnthropicStreamLedger) -> list[tuple[int, Any]]:
    """Return started tool states in stream order."""
    return [
        (tool_index, state)
        for tool_index, state in ledger.blocks.tool_states.items()
        if state.started
    ]


def all_emitted_tools_complete(ledger: AnthropicStreamLedger, request: Any) -> bool:
    """Return whether every emitted tool block has schema-valid input."""
    return ledger.can_salvage_tool_use(tool_schemas_by_name(request))


class OpenAIToolCallAssembler:
    """Assemble OpenAI tool-call deltas into Anthropic SSE tool blocks."""

    def __init__(
        self, *, record_extra_content: RecordToolExtraContent | None = None
    ) -> None:
        self._record_extra_content = record_extra_content

    def process_tool_call(
        self,
        tc: dict[str, Any],
        ledger: AnthropicStreamLedger,
        *,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Process a single tool-call delta and yield Anthropic SSE events."""
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
            ledger.blocks.set_tool_extra_content(tc_index, extra_content)

        if incoming_name is not None:
            ledger.blocks.register_tool_name(tc_index, incoming_name)

        state = ledger.blocks.tool_states.get(tc_index)
        resolved_id = (state.tool_id if state and state.tool_id else None) or tc.get(
            "id"
        )
        resolved_name = (state.name if state else "") or ""

        if not state or not state.started:
            name_ok = bool((resolved_name or "").strip())
            if name_ok:
                tool_id = str(resolved_id) if resolved_id else f"tool_{uuid.uuid4()}"
                display_name = (resolved_name or "").strip() or "tool_call"
                start_extra_content = state.extra_content if state else extra_content
                if start_extra_content:
                    self._record_tool_call_extra_content(tool_id, start_extra_content)
                yield ledger.start_tool_block(
                    tc_index,
                    tool_id,
                    display_name,
                    extra_content=start_extra_content,
                )
                state = ledger.blocks.tool_states[tc_index]
                if state.pre_start_args:
                    pre = state.pre_start_args
                    state.pre_start_args = ""
                    yield from self._emit_tool_arg_delta(
                        ledger,
                        tc_index,
                        pre,
                        tool_argument_aliases=tool_argument_aliases,
                        tool_argument_alias_buffers=tool_argument_alias_buffers,
                    )

        state = ledger.blocks.tool_states.get(tc_index)
        if state is not None and state.tool_id and extra_content:
            self._record_tool_call_extra_content(state.tool_id, extra_content)
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
            tool_argument_aliases=tool_argument_aliases,
            tool_argument_alias_buffers=tool_argument_alias_buffers,
        )

    def flush_task_arg_buffers(self, ledger: AnthropicStreamLedger) -> Iterator[str]:
        """Emit buffered Task args as a single JSON delta."""
        for tool_index, out in ledger.blocks.flush_task_arg_buffers():
            yield ledger.emit_tool_delta(tool_index, out)

    def flush_tool_argument_alias_buffers(
        self,
        ledger: AnthropicStreamLedger,
        tool_argument_aliases: dict[str, dict[str, str]],
        tool_argument_alias_buffers: dict[int, str],
    ) -> Iterator[str]:
        """Emit remaining aliased args without losing malformed JSON."""
        for tool_index, buffered_args in list(tool_argument_alias_buffers.items()):
            if not buffered_args:
                tool_argument_alias_buffers.pop(tool_index, None)
                continue
            state = ledger.blocks.tool_states.get(tool_index)
            if state is None or state.name == "Task":
                continue
            aliases = tool_argument_aliases.get(state.name, {})
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
        ledger: AnthropicStreamLedger,
        tc_index: int,
        args: str,
        *,
        tool_argument_aliases: dict[str, dict[str, str]] | None = None,
        tool_argument_alias_buffers: dict[int, str] | None = None,
    ) -> Iterator[str]:
        """Emit one argument fragment for a started tool block."""
        if not args:
            return
        state = ledger.blocks.tool_states.get(tc_index)
        if state is None:
            return
        if state.name == "Task":
            parsed = ledger.blocks.buffer_task_args(tc_index, args)
            if parsed is not None:
                yield ledger.emit_tool_delta(tc_index, json.dumps(parsed))
            return
        aliases = (
            tool_argument_aliases.get(state.name, {}) if tool_argument_aliases else {}
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
        try:
            parsed = json.loads(argument_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return argument_json
        restored = self._restore_aliased_tool_argument_value(parsed, aliases)
        return json.dumps(restored)

    def _restore_aliased_tool_argument_value(
        self, value: Any, aliases: dict[str, str]
    ) -> Any:
        if isinstance(value, dict):
            return {
                aliases.get(key, key): self._restore_aliased_tool_argument_value(
                    item, aliases
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._restore_aliased_tool_argument_value(item, aliases)
                for item in value
            ]
        return value

    def _record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        if self._record_extra_content is not None:
            self._record_extra_content(tool_call_id, extra_content)
