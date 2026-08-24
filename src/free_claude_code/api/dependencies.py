"""FastAPI dependencies for the explicit runtime service boundary."""

import secrets

from fastapi import Depends, HTTPException, Request
from loguru import logger

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.ports import ProviderPort, RequestRuntimeLease
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings

from .ports import ApiServices


def _extract_proxy_token(request: Request) -> str | None:
    """Token from an ``Authorization: Bearer`` header, else ``x-api-key``.

    Anthropic-native clients (including Claude Desktop's inference gateway)
    authenticate with ``x-api-key``, so it is accepted as an equal scheme.
    A malformed Authorization header wins and yields nothing; without one,
    the ``x-api-key`` value is used as-is.
    """

    authorization = request.headers.get("authorization")
    if authorization is not None:
        parts = authorization.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].casefold() == "bearer":
            token = parts[1].strip()
            return token or None
        return None
    api_key = request.headers.get("x-api-key")
    if api_key is None:
        return None
    return api_key.strip() or None


def get_services(request: Request) -> ApiServices:
    """Return the complete services supplied when the app was constructed."""
    return request.app.state.services


def get_settings(services: ApiServices = Depends(get_services)) -> Settings:
    """Return the current request-runtime settings snapshot."""
    return services.requests.current_settings()


def resolve_provider(
    provider_type: str,
    *,
    lease: RequestRuntimeLease,
) -> ProviderPort:
    """Resolve a provider through one retained generation."""
    should_log_init = not lease.is_provider_cached(provider_type)
    try:
        provider = lease.resolve_provider(provider_type)
    except UnknownProviderError:
        logger.error(
            "Unknown provider_type: '{}'. Supported: {}",
            provider_type,
            ", ".join(f"'{key}'" for key in PROVIDER_CATALOG),
        )
        raise
    if should_log_init:
        logger.info("Provider initialized: {}", provider_type)
    return provider


def require_proxy_auth(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require the configured proxy token as Bearer or Anthropic-style auth."""
    if not settings.proxy_auth_enabled:
        return

    token = _extract_proxy_token(request)
    if token is None:
        # A present-but-malformed Authorization header is a bad credential,
        # not a missing one; with no Authorization at all it's just missing.
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid proxy authentication token"
                if request.headers.get("authorization")
                else "Missing proxy authentication token"
            ),
        )

    if not secrets.compare_digest(
        token.encode("utf-8"),
        settings.proxy_auth_token.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid proxy authentication token",
        )
