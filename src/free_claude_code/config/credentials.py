"""Credential strategy definitions and helpers."""

from enum import StrEnum
from typing import Any


class CredentialStrategy(StrEnum):
    """Supported strategies for selecting API keys from the pool."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"
    SEQUENTIAL = "sequential"


def get_credential_strategy(name: str | None = None) -> CredentialStrategy:
    if not name:
        return CredentialStrategy.ROUND_ROBIN
    try:
        return CredentialStrategy(name.lower())
    except ValueError:
        return CredentialStrategy.ROUND_ROBIN


def parse_api_keys(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(str(k) for k in raw)
    if isinstance(raw, str):
        return tuple(k.strip() for k in raw.split(",") if k.strip())
    return ()
