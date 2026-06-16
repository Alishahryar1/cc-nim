from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from api.models.anthropic import Message, MessagesRequest
from core.anthropic import SSEBuilder


class Orchestrator:
    def __init__(self, proxy_service: Any):
        self.proxy_service = proxy_service

    async def run_cot(
        self, request_data: MessagesRequest, reasoner_model: str, writer_model: str
    ) -> AsyncIterator[str]:
        """Run a 2-step Chain-of-Thought pipeline: Reason -> Write."""
        import uuid

        logger.info(
            "Starting CoT Orchestration: reasoner={}, writer={}",
            reasoner_model,
            writer_model,
        )

        # Initialize SSE
        message_id = f"msg_cot_{uuid.uuid4().hex[:12]}"
        sse = SSEBuilder(message_id, request_data.model)
        yield sse.message_start()

        # STAGE 1: Reasoning
        yield sse.start_thinking_block()
        yield sse.emit_thinking_delta(
            f"Orchestrating Chain-of-Thought with {reasoner_model}...\n"
        )
        reasoning_request = request_data.model_copy(deep=True)
        reasoning_request.model = reasoner_model

        # Adjust prompt for reasoning
        reasoning_prompt = (
            "You are a Strategic Reasoner. Before answering the user, provide a detailed "
            "step-by-step plan and internal monologue on how to best fulfill this request.\n"
            "Focus on logic, edge cases, and accuracy. Do not provide the final answer yet, just the reasoning."
        )
        reasoning_request.messages.insert(
            0, Message(role="user", content=reasoning_prompt)
        )

        logger.debug("CoT Stage 1: Reasoning...")
        reasoning_output = await self._get_full_response(
            reasoner_model, reasoning_request
        )

        yield sse.emit_thinking_delta(
            "Reasoning complete. Synthesizing final response...\n"
        )
        yield sse.stop_thinking_block()

        # STAGE 2: Writing (Streaming)
        # Integrate with cache manually here if needed, or rely on service layer
        writing_request = request_data.model_copy(deep=True)
        writing_request.model = writer_model

        context_msg = (
            f"[INTERNAL REASONING CONTEXT]:\n{reasoning_output}\n\n"
            "Now, based on the above reasoning, provide the final, high-quality response to the user."
        )
        writing_request.messages.append(Message(role="user", content=context_msg))

        logger.debug("CoT Stage 2: Writing...")
        candidates = self.proxy_service._model_router.resolve_candidates(writer_model)
        resolved = candidates[0]
        provider = self.proxy_service._provider_getter(resolved.provider_id)

        writer_routed = writing_request.model_copy(deep=True)
        writer_routed.model = resolved.provider_model

        stream = provider.stream_response(writer_routed)
        async for chunk in stream:
            if "message_start" in chunk:
                continue
            yield chunk

    async def _get_full_response(
        self, model_ref: str, request_data: MessagesRequest
    ) -> str:
        """Robustly extract full text from a provider stream."""
        from core.anthropic.emitted_sse_tracker import EmittedNativeSseTracker

        tracker = EmittedNativeSseTracker()

        panel_request = request_data.model_copy(deep=True)
        panel_request.model = model_ref
        panel_request.stream = False

        candidates = self.proxy_service._model_router.resolve_candidates(model_ref)
        resolved = candidates[0]
        provider = self.proxy_service._provider_getter(resolved.provider_id)

        routed = panel_request.model_copy(deep=True)
        routed.model = resolved.provider_model

        async for chunk in provider.stream_response(routed):
            tracker.feed(chunk)

        return "".join(tracker._text_parts)
