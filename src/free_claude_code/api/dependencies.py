"""FastAPI dependencies for the explicit runtime service boundary."""

import secrets

from fastapi import Depends, HTTPException, Request
from loguru import logger

from free_claude_code.application.ports import ProviderPort, RequestRuntimeLease
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import get_user_facing_error_message
from free_claude_code.providers.exceptions import (
    AuthenticationError,
    UnknownProviderTypeError,
)

from .ports import ApiServices


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
    except AuthenticationError as exc:
        detail = str(exc).strip() or get_user_facing_error_message(exc)
        raise HTTPException(status_code=503, detail=detail) from exc
    except UnknownProviderTypeError:
        logger.error(
            "Unknown provider_type: '{}'. Supported: {}",
            provider_type,
            ", ".join(f"'{key}'" for key in PROVIDER_CATALOG),
        )
        raise
    if should_log_init:
        logger.info("Provider initialized: {}", provider_type)
    return provider


def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require the configured Anthropic-style server API key."""
    anthropic_auth_token = settings.anthropic_auth_token.strip()
    if not anthropic_auth_token:
        return

    header = (
        request.headers.get("x-api-key")
        or request.headers.get("authorization")
        or request.headers.get("anthropic-auth-token")
    )
    if not header:
        raise HTTPException(status_code=401, detail="Missing API key")

    token = header.strip()
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    if token and ":" in token:
        token = token.split(":", 1)[0].strip()

    if not secrets.compare_digest(
        token.encode("utf-8"),
        anthropic_auth_token.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
