import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
            headers=ClientProtocolDriver.jetbrains_headers(),
        )

    assert_product_stream(first.events)


@pytest.mark.smoke_target("clients")
def test_pi_cli_prompt_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    if not shutil.which("pi"):
        pytest.skip("missing_env: Pi CLI not found")
    uv_bin = shutil.which("uv")
    if not uv_bin:
        pytest.skip("missing_env: uv not found")
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    auth_token = "fcc-pi-smoke-token"

    with SmokeServerDriver(
        smoke_config,
        name="product-pi-cli",
        env_overrides={
            "MODEL": provider_model.full_model,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        env = os.environ.copy()
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(server.port),
                "FCC_OPEN_BROWSER": "0",
                "ANTHROPIC_AUTH_TOKEN": auth_token,
                "PI_CODING_AGENT_DIR": str(tmp_path / "pi-agent"),
            }
        )
        result = subprocess.run(
            [
                uv_bin,
                "run",
                "--project",
                str(smoke_config.root),
                "--no-sync",
                "fcc-pi",
                "--no-session",
                "--no-approve",
                "--print",
                "Reply with exactly FCC_SMOKE_PI",
            ],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s + 15,
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    assert result.returncode == 0, result.stderr or result.stdout
    assert "FCC_SMOKE_PI" in result.stdout
    assert "POST /v1/messages" in server_log


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
def test_claude_cli_provider_error_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"missing_env: Claude CLI not found: {smoke_config.claude_bin}")
    broken_model = "lmstudio/fcc-smoke-failing-model"

    with (
        _deliberately_failing_openai_provider() as (
            provider_base_url,
            provider_requests,
        ),
        SmokeServerDriver(
            smoke_config,
            name="product-claude-cli-provider-error",
            env_overrides={
                "MODEL": broken_model,
                "MODEL_FABLE": broken_model,
                "MODEL_OPUS": broken_model,
                "MODEL_SONNET": broken_model,
                "MODEL_HAIKU": broken_model,
                "LM_STUDIO_BASE_URL": provider_base_url,
                "MESSAGING_PLATFORM": "none",
            },
        ).run() as server,
    ):
        result = ClientProtocolDriver.run_claude_prompt(
            claude_bin=claude_bin,
            server=server,
            config=smoke_config,
            cwd=tmp_path,
            prompt="Reply with exactly FCC_SMOKE_UNREACHABLE.",
            model="claude-sonnet-4-5-20250929",
        )
        server_log = server.log_path.read_text(encoding="utf-8", errors="replace")

    combined = f"{result.stdout}\n{result.stderr}"
    lower = combined.lower()
    downstream_requests = sum(
        "POST /v1/messages" in line and "HTTP/1.1" in line
        for line in server_log.splitlines()
    )

    assert result.returncode != 0
    assert "empty or malformed" not in lower
    assert "proxy or gateway intercepting" not in lower
    assert "api error" in lower or "selected model" in lower
    assert "fcc smoke provider rejected the request deliberately" in lower
    assert downstream_requests == 1, server_log
    assert provider_requests == ["/v1/chat/completions"]


@contextmanager
def _deliberately_failing_openai_provider() -> Iterator[tuple[str, list[str]]]:
    provider_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "fcc-smoke-failing-model",
                                "object": "model",
                            }
                        ],
                    },
                )
                return
            if self.path == "/api/v0/models":
                self._write_json(HTTPStatus.OK, {"data": []})
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length)
            provider_requests.append(self.path)
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "fcc_smoke_failure",
                        "message": "FCC smoke provider rejected the request deliberately.",
                    }
                },
            )

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write_json(self, status: HTTPStatus, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        yield f"http://127.0.0.1:{port}/v1", provider_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
