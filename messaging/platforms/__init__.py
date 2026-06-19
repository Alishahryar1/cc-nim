"""Messaging platform adapters (Telegram, Discord, etc.)."""

from .base import (
    ManagedClaudeSessionManagerProtocol,
    ManagedClaudeSessionProtocol,
    MessagingPlatform,
)
from .factory import create_messaging_platform

__all__ = [
    "ManagedClaudeSessionManagerProtocol",
    "ManagedClaudeSessionProtocol",
    "MessagingPlatform",
    "create_messaging_platform",
]
