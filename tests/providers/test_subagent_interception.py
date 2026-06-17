import json
from unittest.mock import MagicMock

import pytest

from config.nim import NimSettings
from core.anthropic import ContentBlockManager
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider
from providers.transports.openai_chat.tool_calls import OpenAIToolCallAssembler


@pytest.mark.asyncio
async def test_task_tool_interception():
    # Setup provider
    config = ProviderConfig(api_key="test")
    provider = NvidiaNimProvider(config, nim_settings=NimSettings())

    # Mock request and sse builder with real ContentBlockManager
    request = MagicMock()
    request.model = "test-model"

    sse = MagicMock()
    sse.blocks = ContentBlockManager()

    # Tool call data (Task tool)
    tc = {
        "index": 0,
        "id": "tool_123",
        "function": {
            "name": "Task",
            "arguments": json.dumps(
                {
                    "description": "test task",
                    "prompt": "do something",
                    "run_in_background": True,
                }
            ),
        },
    }

    tool_calls = OpenAIToolCallAssembler(
        record_extra_content=provider._record_tool_call_extra_content
    )

    # Call the assembler (consume generator to trigger side effects)
    list(tool_calls.process_tool_call(tc, sse))

    # Find the emit_tool_delta call and check args
    calls = sse.emit_tool_delta.call_args_list
    assert len(calls) > 0
    args_passed = json.loads(calls[0][0][1])
    assert args_passed["run_in_background"] is False


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_task_tool_interception())
