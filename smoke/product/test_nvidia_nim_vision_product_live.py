"""Live smoke test: NVIDIA NIM vision (image input) handling.

Gated behind FCC_LIVE_SMOKE=1 and smoke_target("nvidia_nim_vision").
Requires NVIDIA_NIM_API_KEY and a vision-capable model configured in smoke config.
"""

from __future__ import annotations

import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("nvidia_nim_vision")]

# 1x1 black PNG (base64-encoded)
_BLACK_1X1_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def test_nim_vision_base64_image_succeeds(
    smoke_config: SmokeConfig,
) -> None:
    """Send a base64 PNG image to a vision-capable NIM model and expect a 200 response."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    vision_model = os.getenv("FCC_SMOKE_NIM_VISION_MODEL")
    if not vision_model:
        pytest.skip("missing_env: FCC_SMOKE_NIM_VISION_MODEL is not configured")

    provider_model = provider_models[0]

    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-vision-base64",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "NVIDIA_NIM_VISION_ENABLED": "true",
        },
    ).run() as server:
        import httpx

        response = httpx.post(
            f"http://localhost:{server.port}/v1/messages",
            headers={
                "x-api-key": server.api_key,
                "content-type": "application/json",
            },
            json={
                "model": provider_model.full_model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _BLACK_1X1_PNG_B64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "이 이미지는 검은색이야?",
                            },
                        ],
                    }
                ],
            },
            timeout=60.0,
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:500]}"
        )
        body = response.text
        assert "content_block_delta" in body or "content_block_start" in body, (
            f"Expected SSE content blocks in response: {body[:500]}"
        )


def test_nim_vision_disabled_rejects_image(
    smoke_config: SmokeConfig,
) -> None:
    """When NVIDIA_NIM_VISION_ENABLED=false, image blocks should be rejected with 400."""
    if not smoke_config.has_provider_configuration("nvidia_nim"):
        pytest.skip("missing_env: NVIDIA_NIM_API_KEY is not configured")

    provider_models = smoke_config.nvidia_nim_cli_models()
    if not provider_models:
        pytest.skip("missing_env: no NVIDIA NIM vision smoke models configured")

    provider_model = provider_models[0]

    with SmokeServerDriver(
        smoke_config,
        name="product-nvidia-nim-vision-disabled",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
            "NVIDIA_NIM_VISION_ENABLED": "false",
        },
    ).run() as server:
        import httpx

        response = httpx.post(
            f"http://localhost:{server.port}/v1/messages",
            headers={
                "x-api-key": server.api_key,
                "content-type": "application/json",
            },
            json={
                "model": provider_model.full_model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _BLACK_1X1_PNG_B64,
                                },
                            },
                        ],
                    }
                ],
            },
            timeout=30.0,
        )
        assert response.status_code == 400, (
            f"Expected 400 (vision off), got {response.status_code}: {response.text[:500]}"
        )
        assert "image" in response.text.lower(), (
            f"Expected 'image' in error message: {response.text[:500]}"
        )
