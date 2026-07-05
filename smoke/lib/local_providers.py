"""Helpers for local-provider smoke availability checks."""

from urllib.parse import urljoin

import httpx
import pytest

LOCAL_PROVIDER_PROBE_TIMEOUT_S = 1.5


def first_local_provider_model_id(
    provider: str,
    base_url: str,
    *,
    timeout_s: float,
) -> str:
    """Return the first local model id, or skip when the local server is absent."""
    base_url = base_url.strip()
    if not base_url:
        pytest.skip(f"missing_env: {provider} base URL is not configured")

    timeout = min(timeout_s, LOCAL_PROVIDER_PROBE_TIMEOUT_S)
    if provider == "ollama":
        return _first_ollama_model_id(base_url, timeout_s=timeout)
    return _first_openai_compatible_model_id(
        provider,
        base_url,
        timeout_s=timeout,
    )


def _first_openai_compatible_model_id(
    provider: str,
    base_url: str,
    *,
    timeout_s: float,
) -> str:
    models_url = urljoin(base_url.rstrip("/") + "/", "models")
    response = _get_local_provider_response(provider, models_url, timeout_s=timeout_s)
    assert response.status_code == 200, response.text
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                return item["id"]
        pytest.skip(f"missing_env: {provider} local server has no loaded models")
    pytest.fail("product_failure: local /models did not expose a model id")


def _first_ollama_model_id(base_url: str, *, timeout_s: float) -> str:
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    response = _get_local_provider_response("ollama", tags_url, timeout_s=timeout_s)
    assert response.status_code == 200, response.text
    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                return item["name"]
        pytest.skip("missing_env: ollama local server has no pulled models")
    pytest.fail("product_failure: ollama /api/tags did not expose models")


def _get_local_provider_response(
    provider: str,
    url: str,
    *,
    timeout_s: float,
) -> httpx.Response:
    try:
        response = httpx.get(url, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        pytest.skip(f"missing_env: {provider} local server is not reachable: {exc}")
    except httpx.NetworkError as exc:
        pytest.skip(f"missing_env: {provider} local server is not running: {exc}")

    if response.status_code in {404, 405, 502, 503}:
        pytest.skip(
            f"missing_env: {provider} local server is not available at {url}: "
            f"HTTP {response.status_code}"
        )
    return response
