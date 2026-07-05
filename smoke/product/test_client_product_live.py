import shutil
from pathlib import Path

import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import (
    ClientProtocolDriver,
    ConversationDriver,
    ProviderMatrixDriver,
    SmokeServerDriver,
    assert_product_stream,
)

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("clients")
def test_vscode_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-vscode",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            ClientProtocolDriver.adaptive_thinking_payload(),
            headers=ClientProtocolDriver.vscode_headers(),
        )

    assert_product_stream(turn.events)


@pytest.mark.smoke_target("clients")
def test_jetbrains_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-jetbrains",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.stream(
            ClientProtocolDriver.tool_result_payload(),
            headers=ClientProtocolDriver.jetbrains_headers(smoke_config),
        )

    assert_product_stream(first.events)


@pytest.mark.smoke_target("cli")
def test_claude_cli_adaptive_thinking_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()

    with SmokeServerDriver(
        smoke_config,
        name="product-claude-cli-adaptive",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        result = ClientProtocolDriver.run_claude_prompt(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=tmp_path,
            prompt="think hard, then reply with exactly FCC_SMOKE_CLI",
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "POST /v1/messages" in server_log
    assert " 422 " not in server_log
    assert 'HTTP/1.1" 422' not in server_log
    assert "400 Bad Request" not in result.stdout
    assert "FCC_SMOKE_CLI" in result.stdout


@pytest.mark.smoke_target("cli")
def test_claude_cli_multiturn_tool_protocol_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    with SmokeServerDriver(
        smoke_config,
        name="product-claude-cli-protocol",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            ClientProtocolDriver.tool_result_payload()
        )

    assert_product_stream(turn.events)
