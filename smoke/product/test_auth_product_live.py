import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("auth")]


def test_api_auth_header_variants_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    token = "product-smoke-token"
    env_file = tmp_path / "auth-product.env"
    env_file.write_text(f'ANTHROPIC_AUTH_TOKEN="{token}"\n', encoding="utf-8")
    with SmokeServerDriver(
        smoke_config,
        name="product-auth",
        env_overrides={
            "ANTHROPIC_AUTH_TOKEN": token,
            "FCC_ENV_FILE": str(env_file),
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        unauth = httpx.get(
            f"{server.base_url}/v1/models", timeout=smoke_config.timeout_s
        )
        x_api_key = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"x-api-key": token},
            timeout=smoke_config.timeout_s,
        )
        bearer = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"authorization": f"Bearer {token}"},
            timeout=smoke_config.timeout_s,
        )
        anthropic = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"anthropic-auth-token": token},
            timeout=smoke_config.timeout_s,
        )
        invalid = httpx.get(
            f"{server.base_url}/v1/models",
            headers={"x-api-key": "wrong"},
            timeout=smoke_config.timeout_s,
        )

    assert unauth.status_code == 401
    assert x_api_key.status_code == 200
    assert bearer.status_code == 200
    assert anthropic.status_code == 200
    assert invalid.status_code == 401
