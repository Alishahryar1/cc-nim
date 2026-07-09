"""Output ledger for OpenAI Responses streaming assembly."""

from collections.abc import Mapping
from typing import Any

from ..usage import estimate_text_tokens
from .blocks import BlockState


class ResponsesOutputLedger:
    """Track active blocks, reserved output slots, and accumulated usage."""

    def __init__(self) -> None:
        self._output_slots: list[dict[str, Any] | None] = []
        self._active_blocks: dict[int, BlockState] = {}
        self._fallback_text_index = -1
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._reasoning_tokens_estimate = 0

    def active_block(self, index: int) -> BlockState | None:
        return self._active_blocks.get(index)

    def set_active_block(self, state: BlockState) -> None:
        self._active_blocks[state.index] = state

    def pop_active_block(self, index: int) -> BlockState | None:
        return self._active_blocks.pop(index, None)

    def pop_active_blocks_by_output_order(self) -> list[BlockState]:
        states = sorted(
            self._active_blocks.values(), key=lambda state: state.output_index
        )
        self._active_blocks.clear()
        return states

    def reserve_output_slot(self) -> int:
        output_index = len(self._output_slots)
        self._output_slots.append(None)
        return output_index

    def commit_output(self, output_index: int, item: dict[str, Any]) -> None:
        while output_index >= len(self._output_slots):
            self._output_slots.append(None)
        self._output_slots[output_index] = item

    def output(self) -> list[dict[str, Any]]:
        return [item for item in self._output_slots if item is not None]

    def record_usage_delta(self, data: Mapping[str, Any]) -> None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        if isinstance(usage.get("input_tokens"), int):
            self._input_tokens = usage["input_tokens"]
        if isinstance(usage.get("output_tokens"), int):
            self._output_tokens = usage["output_tokens"]

    def add_reasoning_text(self, text: str) -> None:
        self._reasoning_tokens_estimate += estimate_text_tokens(text)

    def usage(self) -> dict[str, Any] | None:
        if self._input_tokens is None and self._output_tokens is None:
            return None
        input_tokens = self._input_tokens or 0
        output_tokens = self._output_tokens or 0
        usage: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        capped_reasoning_tokens = min(self._reasoning_tokens_estimate, output_tokens)
        if capped_reasoning_tokens:
            usage["output_tokens_details"] = {
                "reasoning_tokens": capped_reasoning_tokens
            }
        return usage

    def safe_text_index(self, index: int | None) -> int:
        if index is not None:
            return index
        value = self._fallback_text_index
        self._fallback_text_index -= 1
        return value
