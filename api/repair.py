"""Autonomous repair engine for model outputs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from api.models.anthropic import Message, MessagesRequest


class RepairEngine:
    def __init__(self, service: Any):
        self.service = service

    async def run_repair(
        self, request_data: MessagesRequest, target: str
    ) -> AsyncIterator[str]:
        """Run a request and automatically attempt to repair common failures."""
        candidates = self.service._model_router.resolve_candidates(target)

        # We'll track the full response to validate it
        last_tool_call = None

        # Buffer for the current stream
        buffer = []

        try:
            async for chunk in self.service._stream_with_failover(
                request_data, candidates
            ):
                buffer.append(chunk)
                # Attempt to track tool calls for validation
                if "tool_use" in chunk:
                    try:
                        for line in chunk.splitlines():
                            if line.startswith("data: "):
                                d = json.loads(line[6:].strip())
                                if (
                                    d.get("type") == "content_block_start"
                                    and d.get("content_block", {}).get("type")
                                    == "tool_use"
                                ):
                                    last_tool_call = d["content_block"]
                                    last_tool_call["input_json"] = ""
                                elif (
                                    d.get("type") == "content_block_delta"
                                    and d.get("delta", {}).get("type")
                                    == "input_json_delta"
                                ):
                                    if last_tool_call:
                                        last_tool_call["input_json"] += d["delta"][
                                            "partial_json"
                                        ]
                    except Exception:
                        pass

                # If we see an error event from upstream, trigger repair
                if 'type": "error"' in chunk or "error_type" in chunk:
                    logger.warning(
                        "RepairEngine: Detected error in stream, attempting repair"
                    )
                    repair_stream = await self._attempt_repair(
                        request_data,
                        candidates,
                        "The prior attempt resulted in an error. Please try again and ensure your response is complete and valid.",
                    )
                    async for rc in repair_stream:
                        yield rc
                    return

                yield chunk

            # After stream completes, validate the last tool call if any
            if last_tool_call:
                try:
                    json.loads(last_tool_call["input_json"])
                except json.JSONDecodeError:
                    logger.warning(
                        "RepairEngine: Detected malformed JSON in tool call, attempting repair"
                    )
                    # Note: We already yielded the malformed chunks!
                    # In SSE, we can only append. We'll append a 'repair' message.
                    # This might be tricky for the client, but it's better than nothing.
                    # Alternatively, we could have buffered the WHOLE thing, but that's slow.

                    repair_msg = f"Your tool call for '{last_tool_call['name']}' had malformed JSON: {last_tool_call['input_json']}. Please fix it."
                    repair_stream = await self._attempt_repair(
                        request_data, candidates, repair_msg
                    )
                    async for rc in repair_stream:
                        yield rc
                    return

        except Exception as e:
            logger.warning(
                "RepairEngine: Caught exception {}, attempting repair", type(e).__name__
            )
            repair_stream = await self._attempt_repair(
                request_data,
                candidates,
                f"An unexpected error occurred: {e!s}. Please retry.",
            )
            async for rc in repair_stream:
                yield rc

    async def _attempt_repair(
        self,
        original_request: MessagesRequest,
        candidates: list[Any],
        repair_prompt: str,
    ) -> AsyncIterator[str]:
        """Send a correction prompt to the model."""
        repair_req = original_request.model_copy(deep=True)
        # Append the repair turn
        repair_req.messages.append(
            Message(
                role="assistant",
                content="[Internal: Error detected, requesting correction]",
            )
        )
        repair_req.messages.append(Message(role="user", content=repair_prompt))

        return self.service._stream_with_failover(repair_req, candidates)
