import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from api.models.anthropic import Message, MessagesRequest
from core.anthropic import SSEBuilder


class AgentEngine:
    def __init__(self, proxy_service: Any):
        self.proxy_service = proxy_service

    async def run_agent(
        self, request_data: MessagesRequest, planner_model: str, executor_model: str
    ) -> AsyncIterator[str]:
        """Run an autonomous Plan-and-Execute loop."""
        logger.info(
            "Starting Plan-and-Execute Agent: planner={}, executor={}",
            planner_model,
            executor_model,
        )

        message_id = f"msg_agent_{uuid.uuid4().hex[:12]}"
        sse = SSEBuilder(message_id, request_data.model)
        yield sse.message_start()

        # STAGE 1: Planning
        yield sse.start_thinking_block()
        yield sse.emit_thinking_delta(
            f"Planning execution strategy using {planner_model}...\n"
        )

        plan_prompt = (
            "You are a Project Planner. Break down the user's request into a list of discrete, "
            "sequential tasks that an AI executor can perform. \n"
            "Output the plan as a valid JSON list of strings under a 'tasks' key.\n"
            f"REQUEST: {request_data.messages[-1].content}"
        )

        planner_request = request_data.model_copy(
            update={"messages": [Message(role="user", content=plan_prompt)]}
        )

        plan_raw = await self.proxy_service._orchestrator._get_full_response(
            planner_model, planner_request
        )

        try:
            match = re.search(r"\{.*\}", plan_raw, re.DOTALL)
            plan_json = json.loads(match.group(0)) if match else {}
            tasks = plan_json.get("tasks", [request_data.messages[-1].content])
        except Exception:
            logger.warning(
                "Agent Planning failed to produce JSON, using raw request as single task."
            )
            tasks = [request_data.messages[-1].content]

        yield sse.emit_thinking_delta(f"Plan generated with {len(tasks)} steps.\n")

        # STAGE 2: Iterative Execution
        results = []
        for i, task in enumerate(tasks):
            yield sse.emit_thinking_delta(f"Step {i + 1}/{len(tasks)}: {task}...\n")

            exec_prompt = (
                f"You are an AI Executor. Task {i + 1} of {len(tasks)}: '{task}'\n\n"
                f"CONTEXT FROM PREVIOUS STEPS:\n" + "\n".join(results) + "\n\n"
                "Execute the task and provide a concise output of findings or results."
            )

            exec_request = request_data.model_copy(
                update={"messages": [Message(role="user", content=exec_prompt)]}
            )

            res = await self.proxy_service._orchestrator._get_full_response(
                executor_model, exec_request
            )
            results.append(f"Result {i + 1}: {res}")

        # STAGE 3: Final Aggregation
        yield sse.emit_thinking_delta(
            "Execution finished. Synthesizing final response...\n"
        )
        yield sse.stop_thinking_block()

        agg_prompt = (
            "You are the Aggregator. Synthesize the following task results into a single, cohesive, "
            "comprehensive response to the original user request.\n\n"
            f"ORIGINAL REQUEST: {request_data.messages[-1].content}\n\n"
            "EXECUTION LOG:\n" + "\n".join(results)
        )

        agg_request = request_data.model_copy(
            update={
                "messages": [
                    *request_data.messages,
                    Message(role="user", content=agg_prompt),
                ]
            }
        )

        candidates = self.proxy_service._model_router.resolve_candidates(executor_model)
        stream = self.proxy_service._stream_with_failover(agg_request, candidates)
        async for chunk in stream:
            if "message_start" in chunk:
                continue
            yield chunk
