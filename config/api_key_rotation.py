"""API-key rotation mode configuration."""

from enum import StrEnum


class ApiKeyRotationMode(StrEnum):
    """Supported API-key rotation modes."""

    ROUND_ROBIN = "round_robin"
    FAILOVER_ON_LIMIT = "failover_on_limit"
