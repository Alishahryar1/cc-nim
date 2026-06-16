import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from api.models.anthropic import Message, MessagesRequest
from core.anthropic import SSEBuilder


class FusionEngine:
    def __init__(self, proxy_service: Any):
        self.proxy_service = proxy_service

    async def _get_model_response(
        self, model_ref: str, request_data: MessagesRequest
    ) -> str:
        """Helper to get a full non-streaming text response from a specific model."""
        try:
            return await self.proxy_service._orchestrator._get_full_response(
                model_ref, request_data
            )
        except Exception as e:
            logger.error("Fusion panel model {} failed: {}", model_ref, e)
            return f"Error from {model_ref}: {e!s}"

    async def run_fusion(
        self,
        request_data: MessagesRequest,
        panel_models: list[str],
        judge_model: str,
        rounds: int = 1,
    ) -> AsyncIterator[str]:
        """Run parallel queries and synthesize with a multi-stage reflection pipeline."""
        logger.info(
            "Starting Recursive Fusion: panel={}, judge={}, rounds={}",
            panel_models,
            judge_model,
            rounds,
        )

        # Initialize SSE for progress
        message_id = f"msg_fusion_{uuid.uuid4().hex[:12]}"
        sse = SSEBuilder(message_id, request_data.model)
        yield sse.message_start()

        # STAGE 1: Parallel Drafting
        yield sse.start_thinking_block()
        yield sse.emit_thinking_delta(
            f"Gathering expert drafts from {len(panel_models)} models...\n"
        )

        tasks = [self._get_model_response(m, request_data) for m in panel_models]
        responses = await asyncio.gather(*tasks)

        yield sse.emit_thinking_delta(
            "Expert drafts received. Analyzing contradictions and potential hallucinations...\n"
        )

        # STAGE 2: Self-Correction & Critique (First synthesis pass)
        critique_prompt = (
            "You are a Quality Assurance Specialist. Analyze these conflicting responses to the request:\n\n"
            f"REQUEST: {request_data.messages[-1].content}\n\n"
        )
        for i, resp in enumerate(responses):
            critique_prompt += f"EXPERT {i + 1}:\n{resp}\n\n"

        critique_prompt += (
            "Identify:\n"
            "1. Potential errors or hallucinations in any model.\n"
            "2. The 'gold standard' points that appear most reliable.\n"
            "3. Any missing information from the draft set.\n"
            "Deliver a concise CRITIQUE report for the final synthesis."
        )

        critique_report = await self._get_model_response(
            judge_model,
            request_data.model_copy(
                update={
                    "messages": [
                        *request_data.messages,
                        Message(role="user", content=critique_prompt),
                    ]
                }
            ),
        )

        yield sse.emit_thinking_delta(
            f"Critique report generated. (Round 1/{rounds})\n"
        )

        # Multi-round Debate Extension
        current_critique = critique_report
        for r in range(2, rounds + 1):
            yield sse.emit_thinking_delta(f"Starting Debate Round {r}...\n")
            debate_prompt = (
                f"Analyze the prior expert drafts and the previous critique:\n{current_critique}\n\n"
                "Refine the critique. Focus on resolving remaining disagreements between experts."
            )
            current_critique = await self._get_model_response(
                judge_model,
                request_data.model_copy(
                    update={
                        "messages": [
                            *request_data.messages,
                            Message(role="user", content=debate_prompt),
                        ]
                    }
                ),
            )
            yield sse.emit_thinking_delta(f"Round {r} complete.\n")

        yield sse.emit_thinking_delta(
            "Finalizing synthesis based on expert strengths and debate...\n"
        )
        yield sse.stop_thinking_block()

        # STAGE 3: Final Synthesis with Recursive Context
        final_prompt = (
            "You are the Lead Architect. Synthesize a final response based on expert drafts and the QA critique.\n\n"
            "QA CRITIQUE REPORT:\n"
            f"{critique_report}\n\n"
            "EXPERT DRAFTS:\n"
        )
        for i, resp in enumerate(responses):
            final_prompt += f"EXPERT {i + 1} DRAFT:\n{resp}\n\n"

        final_prompt += (
            "FINAL INSTRUCTION: Create a master response. Prioritize accuracy over verbosity. "
            "Eliminate all contradictions identified in the QA report. Provide the final result now."
        )

        judge_request = request_data.model_copy(deep=True)
        judge_request.model = judge_model
        judge_request.messages.append(Message(role="user", content=final_prompt))

        candidates = self.proxy_service._model_router.resolve_candidates(judge_model)
        resolved = candidates[0]
        provider = self.proxy_service._provider_getter(resolved.provider_id)

        judge_routed = judge_request.model_copy(deep=True)
        judge_routed.model = resolved.provider_model

        # Final pass: Stream result using Judge provider directly but skip their message_start
        # since we already sent one.
        stream = provider.stream_response(judge_routed)
        async for chunk in stream:
            if "message_start" in chunk:
                continue
            yield chunk
