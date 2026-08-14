import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import (
    ConversationDriver,
    SmokeServerDriver,
    assert_product_stream,
    tool_use_blocks,
)
from smoke.lib.skips import skip_if_upstream_unavailable_events

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("providers")]

_STEPFUN_MODEL = "nvidia_nim/stepfun-ai/step-3.7-flash"


@pytest.mark.smoke_target("tools")
def test_nvidia_nim_stepfun_tool_use_e2e(
    smoke_config: SmokeConfig,
) -> None:
    """Normalize StepFun tool use when Claude relies on its auto default."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the Bash tool now to run the command printf FCC_STEP_TOOL. "
                    "Do not answer in prose."
                ),
            }
        ],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            }
        ],
        "thinking": {"type": "adaptive"},
    }
    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-stepfun-tool-use",
        env_overrides={"MODEL": _STEPFUN_MODEL, "MESSAGING_PLATFORM": "none"},
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)

    skip_if_upstream_unavailable_events(turn.events)
    assert_product_stream(turn.events)
    calls = tool_use_blocks(turn.events)
    assert len(calls) == 1, turn.text
    assert calls[0]["name"] == "Bash"
    assert calls[0]["input"] == {"command": "printf FCC_STEP_TOOL"}
    assert "<tool_call>" not in turn.text
