"""Dependency injection for FastAPI."""

import secrets

from fastapi import Depends, HTTPException, Request
from loguru import logger
from starlette.applications import Starlette

from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from config.settings import get_settings as _get_settings
from core.anthropic import get_user_facing_error_message
from providers.base import BaseProvider
from providers.exceptions import (
    AuthenticationError,
    ServiceUnavailableError,
    UnknownProviderTypeError,
)
from providers.runtime import ProviderRuntime


def get_settings() -> Settings:
    """Return cached :class:`~config.settings.Settings` (FastAPI-friendly alias)."""
    return _get_settings()


def get_provider_runtime(app: Starlette) -> ProviderRuntime:
    """Return the app-scoped provider runtime installed by ``AppRuntime``."""
    runtime = getattr(app.state, "provider_runtime", None)
    if isinstance(runtime, ProviderRuntime):
        return runtime
    raise ServiceUnavailableError(
        "Provider runtime is not configured. Ensure AppRuntime startup ran "
        "or assign app.state.provider_runtime for test apps."
    )


def maybe_provider_runtime(app: Starlette) -> ProviderRuntime | None:
    """Return the app-scoped provider runtime when it is installed."""
    runtime = getattr(app.state, "provider_runtime", None)
    return runtime if isinstance(runtime, ProviderRuntime) else None


def resolve_provider(
    provider_type: str,
    *,
    app: Starlette,
) -> BaseProvider:
    """Resolve a provider through the app-scoped provider runtime."""
    runtime = get_provider_runtime(app)
    should_log_init = not runtime.is_cached(provider_type)
    try:
        provider = runtime.resolve_provider(provider_type)
    except AuthenticationError as e:
        # Provider :class:`~providers.exceptions.AuthenticationError` messages are
        # curated configuration hints (env var names, docs links), not upstream noise.
        detail = str(e).strip() or get_user_facing_error_message(e)
        raise HTTPException(status_code=503, detail=detail) from e
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
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Require a server API key (Anthropic-style).

    Checks `x-api-key` header or `Authorization: Bearer ...` against
    `Settings.anthropic_auth_token`. If `ANTHROPIC_AUTH_TOKEN` is empty, this is a no-op.
    """
    anthropic_auth_token = settings.anthropic_auth_token.strip()
    if not anthropic_auth_token:
        # No API key configured -> allow
        return

    header = (
        request.headers.get("x-api-key")
        or request.headers.get("authorization")
        or request.headers.get("anthropic-auth-token")
    )
    if not header:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Support both raw key in X-API-Key and Bearer token in Authorization
    token = header.strip()
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()

    # Strip anything after the first colon to handle tokens with appended model names
    if token and ":" in token:
        token = token.split(":", 1)[0].strip()

    # Constant-time comparison to avoid leaking the configured token via
    # response-time differences on a per-byte mismatch (CWE-208).
    if not secrets.compare_digest(
        token.encode("utf-8"), anthropic_auth_token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
