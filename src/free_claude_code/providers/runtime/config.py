"""Provider configuration construction from neutral catalog metadata."""

import os

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.env_files import env_file_override
from free_claude_code.config.provider_catalog import ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig

CREDENTIAL_ROTATION_POLICIES = frozenset(
    {"single", "round_robin", "least_used", "failover", "on_error"}
)
DEFAULT_CREDENTIAL_ROTATION = "single"


def string_setting(settings: Settings, attr_name: str | None, default: str = "") -> str:
    """Return a string-valued settings attribute, ignoring non-string mocks."""
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def provider_credential(descriptor: ProviderDescriptor, settings: Settings) -> str:
    """Return the configured credential for a provider descriptor."""
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    if descriptor.credential_attr:
        return string_setting(settings, descriptor.credential_attr)
    return ""


def require_provider_credential(
    descriptor: ProviderDescriptor, credential: str
) -> None:
    """Raise a user-facing configuration error when a required key is missing."""
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise ApplicationUnavailableError(message)


def parse_credential_keys(credential: str) -> tuple[str, ...]:
    """Split a comma-separated credential value into individual keys."""
    return tuple(
        key for key in (part.strip() for part in credential.split(",")) if key
    )


def credential_rotation_policy(
    descriptor: ProviderDescriptor, settings: Settings
) -> str:
    """Resolve the credential rotation policy for one provider.

    The policy is read from ``{CREDENTIAL_ENV}_ROTATION`` in the configured
    dotenv files first, then the process environment. Unknown values fall back
    to ``single`` so a typo never breaks provider construction.
    """
    if descriptor.credential_env is None:
        return DEFAULT_CREDENTIAL_ROTATION
    env_key = f"{descriptor.credential_env}_ROTATION"
    value = env_file_override(settings.model_config, env_key)
    if value is None:
        value = os.environ.get(env_key, "")
    value = value.strip().lower()
    if value in CREDENTIAL_ROTATION_POLICIES:
        return value
    return DEFAULT_CREDENTIAL_ROTATION


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    """Build shared provider configuration for one provider descriptor."""
    credential = provider_credential(descriptor, settings)
    require_provider_credential(descriptor, credential)
    api_keys = parse_credential_keys(credential)
    rotation = credential_rotation_policy(descriptor, settings)
    base_url = string_setting(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    resolved_base_url = base_url or descriptor.default_base_url
    if not resolved_base_url:
        raise AssertionError(
            f"Provider {descriptor.provider_id!r} has no configured base URL."
        )
    proxy = string_setting(settings, descriptor.proxy_attr)
    return ProviderConfig(
        api_key=api_keys[0] if api_keys else credential,
        base_url=resolved_base_url,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
        api_keys=api_keys,
        credential_rotation=rotation,
    )
