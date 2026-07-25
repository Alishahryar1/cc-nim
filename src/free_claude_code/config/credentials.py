"""Credential strategy definitions and helpers."""

from enum import Enum


class CredentialStrategy(str, Enum):
    """Supported strategies for selecting API keys from the pool."""

    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


def get_credential_strategy(strategy: str | CredentialStrategy | None) -> CredentialStrategy:
    """Resolve a strategy configuration input into a valid CredentialStrategy enum."""
    if isinstance(strategy, CredentialStrategy):
        return strategy
    if isinstance(strategy, str):
        cleaned = strategy.strip().lower()
        for member in CredentialStrategy:
            if member.value == cleaned or member.name.lower() == cleaned:
                return member
    return CredentialStrategy.ROUND_ROBIN
