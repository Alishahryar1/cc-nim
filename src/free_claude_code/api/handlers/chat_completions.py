"""OpenAI Chat Completions API product flow."""

import json

from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.chat_completions.converter import (
    ChatCompletionsToAnthropicConverter,
)
from free_claude_code.core.chat_completions.models import ChatCompletionsRequest
from free_claude_code.core.chat_completions.stream import (
    chat_completions_sse_from_anthropic,
)
from free_claude_code.core.diagnostics import safe_exception_message


class ChatCompletionsHandler:
    """Handle OpenAI Chat Completions requests by converting to Anthropic format.

    The route acquires a RequestRuntimeLease and passes the already-resolved
    provider_executor here. This handler only owns conversion logic.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model_router: ModelRouter | None = None,
        provider_executor: ProviderExecutor,
    ) -> None:
        self._model_router = model_router or ModelRouter(settings)
        self._provider_executor = provider_executor

    async def create(self, request_data: ChatCompletionsRequest) -> object:
        """Create a Chat Completions response."""
        try:
            anthropic_payload = (
                ChatCompletionsToAnthropicConverter.to_anthropic_payload(request_data)
            )
            response_request = MessagesRequest(**anthropic_payload)
            routed = self._model_router.resolve_messages_request(response_request)

            streamed = self._provider_executor.stream(
                routed,
                wire_api="messages",
                raw_log_label="CHAT_COMPLETIONS_PAYLOAD",
                raw_log_payload=request_data.model_dump(mode="json", exclude_none=True),
                request_id="",
            )

            if request_data.stream is False:
                from free_claude_code.core.anthropic import (
                    aggregate_anthropic_sse_to_message,
                )

                message = await aggregate_anthropic_sse_to_message(streamed)
                return self._to_non_streaming_response(message, request_data.model)

            async def event_stream():
                async for chunk in chat_completions_sse_from_anthropic(
                    streamed, request_data.model
                ):
                    yield chunk

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        except ApplicationError:
            raise
        except Exception as exc:
            logger.error("Chat Completions error: {}", safe_exception_message(exc))
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": safe_exception_message(exc),
                        "type": "server_error",
                    }
                },
            )

    def _to_non_streaming_response(self, message: object, model: str) -> dict:
        """Convert an aggregated Anthropic message to Chat Completions format."""
        content = ""
        tool_calls: list[dict] = []
        finish_reason = "stop"

        if hasattr(message, "content"):
            blocks = (
                message.content
                if isinstance(message.content, list)
                else [message.content]
            )
            for block in blocks:
                if not hasattr(block, "type"):
                    continue
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input)
                                if isinstance(block.input, dict)
                                else str(block.input),
                            },
                        }
                    )

        if hasattr(message, "stop_reason"):
            stop_map = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }
            finish_reason = stop_map.get(message.stop_reason, "stop")

        choice: dict = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content or None,
            },
            "finish_reason": finish_reason,
        }
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls

        import time
        import uuid

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
