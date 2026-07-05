from pathlib import Path

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.server import start_server

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("auth")]


def test_auth_token_is_enforced_for_all_supported_header_shapes(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    token = "fcc-smoke-token"
    env_file = tmp_path / "auth.env"
    env_file.write_text(f'ANTHROPIC_AUTH_TOKEN="{token}"\n', encoding="utf-8")

    with start_server(
        smoke_config,
        env_overrides={
            "ANTHROPIC_AUTH_TOKEN": token,
            "FCC_ENV_FILE": str(env_file),
            "MESSAGING_PLATFORM": "none",
        },
        name="auth",
    ) as server:
        assert httpx.get(f"{server.base_url}/").status_code == 401
        assert (
            httpx.get(f"{server.base_url}/", headers={"x-api-key": "wrong"}).status_code
            == 401
        )
        assert (
            httpx.get(f"{server.base_url}/", headers={"x-api-key": token}).status_code
            == 200
        )
        assert (
            httpx.get(
                f"{server.base_url}/",
                headers={"authorization": f"Bearer {token}"},
            ).status_code
            == 200
        )
        assert (
            httpx.get(
                f"{server.base_url}/",
                headers={"anthropic-auth-token": token},
            ).status_code
            == 200
        )
