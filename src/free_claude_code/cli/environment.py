"""Shared child-only environment setup for native clients."""

from collections.abc import Mapping, Sequence

from .local_http import with_local_proxy_bypass


def client_environment(
    base_env: Mapping[str, str],
    *,
    proxy_root_url: str,
    remove_keys: Sequence[str] = (),
    remove_prefixes: tuple[str, ...] = (),
    updates: Mapping[str, str] | None = None,
    case_sensitive: bool = True,
) -> dict[str, str]:
    """Replace owned connection settings while preserving native user state."""

    keys = set(remove_keys if case_sensitive else map(str.casefold, remove_keys))
    prefixes = (
        remove_prefixes
        if case_sensitive
        else tuple(prefix.casefold() for prefix in remove_prefixes)
    )
    env = {
        key: value
        for key, value in base_env.items()
        if (name := key if case_sensitive else key.casefold()) not in keys
        and not name.startswith(prefixes)
    }
    if updates:
        env.update(updates)
    return with_local_proxy_bypass(env, proxy_root_url=proxy_root_url)


def require_unset_environment(base_env: Mapping[str, str], keys: Sequence[str]) -> None:
    """Avoid replacing another owner's explicit process configuration."""

    for key in keys:
        if base_env.get(key, "").strip():
            raise ValueError(f"{key} is already set. Unset it before using FCC.")
