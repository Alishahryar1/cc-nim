"""Shared OpenAI-compatible client env for Atomic Agents, Prime Agent, and Muse."""

from collections.abc import Mapping

from free_claude_code.cli.local_http import with_local_proxy_bypass

_OPENAI_STRIP_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
    }
)

_DEFAULT_AUTH = "fcc-no-auth"


def openai_compat_base_url(proxy_root_url: str) -> str:
    """Return the OpenAI-style ``/v1`` base URL for the local FCC proxy."""

    return f"{proxy_root_url.rstrip('/')}/v1"


def proxy_bearer_token(auth_token: str) -> str:
    """Return a non-empty bearer token for OpenAI-compatible clients."""

    stripped = auth_token.strip()
    return stripped or _DEFAULT_AUTH


def build_openai_compat_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
    model: str | None = None,
) -> dict[str, str]:
    """Point OpenAI-compatible SDKs at FCC without leaking parent OpenAI keys."""

    env = with_local_proxy_bypass(
        {key: value for key, value in base_env.items() if key not in _OPENAI_STRIP_KEYS},
        proxy_root_url=proxy_root_url,
    )
    base_url = openai_compat_base_url(proxy_root_url)
    token = proxy_bearer_token(auth_token)
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_BASE"] = base_url
    env["OPENAI_API_KEY"] = token
    if model:
        env["FCC_MODEL"] = model
    return env
