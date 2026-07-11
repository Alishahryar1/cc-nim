from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from free_claude_code.api.dependencies import (
    get_services,
    get_settings,
    require_api_key,
    resolve_provider,
)
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.ports import RequestRuntimeLease
from free_claude_code.config.settings import Settings
from free_claude_code.providers.exceptions import AuthenticationError
from tests.api.support import create_test_app


def _request(*, headers: dict[str, str], token: str) -> tuple[Request, Settings]:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [
                (key.lower().encode(), value.encode()) for key, value in headers.items()
            ],
        }
    )
    settings = Settings.model_construct(anthropic_auth_token=token)
    return request, settings


def _lease(*, provider=None, error: Exception | None = None):
    lease = MagicMock(spec=RequestRuntimeLease)
    lease.is_provider_cached.return_value = False
    if error is None:
        lease.resolve_provider.return_value = provider or MagicMock()
    else:
        lease.resolve_provider.side_effect = error
    return lease


def test_get_services_reads_the_single_app_state_boundary() -> None:
    app = create_test_app()
    request = Request({"type": "http", "app": app})

    services = get_services(request)

    assert services is app.state.services
    assert isinstance(services, ApiServices)


def test_get_settings_reads_current_request_runtime_settings() -> None:
    app = create_test_app(
        Settings.model_construct(
            model="deepseek/test-model",
            anthropic_auth_token="",
        )
    )

    assert get_settings(app.state.services).model == "deepseek/test-model"


def test_resolve_provider_uses_retained_lease_and_logs_first_initialization() -> None:
    provider = MagicMock()
    lease = _lease(provider=provider)

    with patch("free_claude_code.api.dependencies.logger.info") as log_info:
        result = resolve_provider("nvidia_nim", lease=lease)

    assert result is provider
    lease.resolve_provider.assert_called_once_with("nvidia_nim")
    log_info.assert_called_once_with("Provider initialized: {}", "nvidia_nim")


def test_resolve_provider_skips_initialization_log_for_cached_provider() -> None:
    lease = _lease()
    lease.is_provider_cached.return_value = True

    with patch("free_claude_code.api.dependencies.logger.info") as log_info:
        resolve_provider("nvidia_nim", lease=lease)

    log_info.assert_not_called()


def test_resolve_provider_missing_key_raises_503() -> None:
    lease = _lease(
        error=AuthenticationError(
            "OPENROUTER_API_KEY is required. Get one at https://openrouter.ai"
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_provider("open_router", lease=lease)

    assert exc_info.value.status_code == 503
    assert "OPENROUTER_API_KEY" in exc_info.value.detail
    assert "openrouter.ai" in exc_info.value.detail


def test_resolve_provider_unrelated_error_is_not_reclassified() -> None:
    lease = _lease(error=ValueError("unrelated config"))

    with pytest.raises(ValueError, match="unrelated config"):
        resolve_provider("nvidia_nim", lease=lease)


def test_require_api_key_allows_when_no_token_configured():
    request, settings = _request(headers={}, token="")

    require_api_key(request, settings)


def test_require_api_key_rejects_missing_token():
    request, settings = _request(headers={}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing API key"


def test_require_api_key_accepts_x_api_key():
    request, settings = _request(headers={"x-api-key": "secret"}, token="secret")

    require_api_key(request, settings)


def test_require_api_key_accepts_bearer_token_and_strips_model_suffix():
    request, settings = _request(
        headers={"authorization": "Bearer secret:claude-sonnet"},
        token="secret",
    )

    require_api_key(request, settings)


def test_require_api_key_rejects_invalid_token():
    request, settings = _request(headers={"x-api-key": "wrong"}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid API key"
