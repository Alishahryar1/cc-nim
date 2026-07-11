"""Platform-agnostic messaging layer."""

from .event_parser import parse_cli_event
from .managed_protocols import (
    ManagedClaudeSessionManagerProtocol,
    ManagedClaudeSessionProtocol,
)
from .models import IncomingMessage, MessageScope
from .platforms.ports import OutboundMessenger
from .session import SessionStore
from .workflow import MessagingWorkflow

__all__ = [
    "IncomingMessage",
    "ManagedClaudeSessionManagerProtocol",
    "ManagedClaudeSessionProtocol",
    "MessageScope",
    "MessagingWorkflow",
    "OutboundMessenger",
    "SessionStore",
    "parse_cli_event",
]
