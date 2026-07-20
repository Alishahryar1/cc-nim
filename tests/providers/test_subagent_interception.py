import json
from unittest.mock import MagicMock

import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import NVIDIA_NIM_DEFAULT_BASE
from free_claude_code.core.anthropic import StreamBlockLedger
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
)
from tests.providers.support import immediate_admission


@pytest.mark.asyncio
async def test_task_tool_interception():
    # Setup provider
    config = ProviderConfig(api_key="test", base_url=NVIDIA_NIM_DEFAULT_BASE)
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(),
    )

    # Mock request and stream ledger with real StreamBlockLedger
    request = MagicMock()
    request.model = "test-model"

    sse = MagicMock()
    sse.blocks = StreamBlockLedger()

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
