"""NVIDIA NIM provider implementation."""

import logging
import json
import uuid
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.rate_limit import GlobalRateLimiter
from .request import build_request_body
from .response import convert_response
from .errors import map_error
from .utils import (
    SSEBuilder,
    map_stop_reason,
    ThinkTagParser,
    HeuristicToolParser,
    ContentType,
)

logger = logging.getLogger(__name__)


class NvidiaNimProvider(BaseProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key
        self._base_url = (
            config.base_url or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")
        self._nim_settings = config.nim_settings
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
        )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=2,
            timeout=300.0,
        )

    def _build_request_body(self, request: Any, stream: bool = False) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request, self._nim_settings, stream=stream)

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Wait if globally rate limited
        waited_reactively = await self._global_rate_limiter.wait_if_blocked()

        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        if waited_reactively:
            error_msg = "⏱️ Global rate limit active. Resuming now..."
            logger.info("NIM_STREAM: Reactive block detected, notified user")
            yield sse.message_start()
            for event in sse.emit_error(error_msg):
                yield event

        body = self._build_request_body(request, stream=True)
        logger.info(
            f"NIM_STREAM: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        # Only emit message_start if we haven't already emitted it during reactive wait
        if not waited_reactively:
            yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()

        finish_reason = None
        usage_info = None
        error_occurred = False
        error_message = ""

        try:
            stream = await self._client.chat.completions.create(**body, stream=True)
            async for chunk in stream:
                # OpenAI client returns objects, not JSON
                if getattr(chunk, "usage", None):
                    usage_info = chunk.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    logger.debug(f"NIM finish_reason: {finish_reason}")

                # Handle reasoning content from delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(reasoning)

                # Handle text content
                if delta.content:
                    for part in think_parser.feed(delta.content):
                        if part.type == ContentType.THINKING:
                            for event in sse.ensure_thinking_block():
                                yield event
                            yield sse.emit_thinking_delta(part.content)
                        else:
                            filtered_text, detected_tools = heuristic_parser.feed(
                                part.content
                            )

                            if filtered_text:
                                for event in sse.ensure_text_block():
                                    yield event
                                yield sse.emit_text_delta(filtered_text)

                            for tool_use in detected_tools:
                                for event in sse.close_content_blocks():
                                    yield event

                                block_idx = sse.blocks.allocate_index()
                                yield sse.content_block_start(
                                    block_idx,
                                    "tool_use",
                                    id=tool_use["id"],
                                    name=tool_use["name"],
                                )
                                yield sse.content_block_delta(
                                    block_idx,
                                    "input_json_delta",
                                    json.dumps(tool_use["input"]),
                                )
                                yield sse.content_block_stop(block_idx)

                # Handle native tool calls
                if delta.tool_calls:
                    for event in sse.close_content_blocks():
                        yield event
                    for tc in delta.tool_calls:
                        # Convert OpenAI tool call object to dict for existing logic
                        tc_info = {
                            "index": tc.index,
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for event in self._process_tool_call(tc_info, sse):
                            yield event

        except Exception as e:
            logger.error(f"NIM_ERROR: {type(e).__name__}: {e}")
            mapped_e = map_error(e)
            error_occurred = True
            error_message = str(mapped_e)
            logger.info(f"NIM_STREAM: Emitting SSE error event for {type(e).__name__}")
            # Ensure open blocks are closed before emitting error to follow Anthropic protocol
            for event in sse.close_content_blocks():
                yield event
            for event in sse.emit_error(error_message):
                yield event

        # Flush remaining content
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING:
                for event in sse.ensure_thinking_block():
                    yield event
                yield sse.emit_thinking_delta(remaining.content)
            else:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in sse.close_content_blocks():
                yield event

            block_idx = sse.blocks.allocate_index()
            yield sse.content_block_start(
                block_idx,
                "tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
            )
            yield sse.content_block_delta(
                block_idx,
                "input_json_delta",
                json.dumps(tool_use["input"]),
            )
            yield sse.content_block_stop(block_idx)

        if (
            not error_occurred
            and sse.blocks.text_index == -1
            and not sse.blocks.tool_indices
        ):
            for event in sse.ensure_text_block():
                yield event
            yield sse.emit_text_delta(" ")

        for event in sse.close_all_blocks():
            yield event

        output_tokens = (
            usage_info.completion_tokens
            if usage_info and hasattr(usage_info, "completion_tokens")
            else sse.estimate_output_tokens()
        )
        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()
        yield sse.done()

    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request."""
        await self._global_rate_limiter.wait_if_blocked()

        body = self._build_request_body(request, stream=False)
        logger.info(
            f"NIM_COMPLETE: model={body.get('model')} msgs={len(body.get('messages', []))} tools={len(body.get('tools', []))}"
        )

        try:
            response = await self._client.chat.completions.create(**body)
            # Responseconverter expects a dict
            return response.model_dump()
        except Exception as e:
            logger.error(f"NIM_ERROR: {type(e).__name__}: {e}")
            raise map_error(e)

    def convert_response(self, response_json: dict, original_request: Any) -> Any:
        """Convert provider response to Anthropic format."""
        return convert_response(response_json, original_request)

    def _process_tool_call(self, tc: dict, sse: Any):
        """Process a single tool call delta and yield SSE events.

        Args:
            tc: Tool call delta info dict
            sse: SSEBuilder instance
        """

        tc_index = tc.get("index", 0)
        if tc_index < 0:
            tc_index = len(sse.blocks.tool_indices)

        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )

        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"

                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            # INTERCEPTION: If this is a Task tool, force background=False
            current_name = sse.blocks.tool_names.get(tc_index, "")
            if current_name == "Task":
                try:
                    args_json = json.loads(args)
                    if args_json.get("run_in_background") is not False:
                        logger.info(
                            f"NIM_INTERCEPT: Forcing run_in_background=False for Task {tc.get('id', 'unknown')}"
                        )
                        args_json["run_in_background"] = False
                        args = json.dumps(args_json)
                except Exception as e:
                    logger.warning(
                        f"NIM_INTERCEPT: Failed to parse/modify Task args: {e}"
                    )

            yield sse.emit_tool_delta(tc_index, args)
