"""Platform-agnostic messaging layer."""

from .event_parser import parse_cli_event
from .managed_protocols import (
    ManagedClaudeSessionManagerProtocol,
    ManagedClaudeSessionProtocol,
)
from .models import IncomingMessage
from .platforms.ports import OutboundMessenger
from .session import SessionStore
from .trees import MessageNode, MessageState, MessageTree, TreeQueueManager
from .workflow import MessagingWorkflow

__all__ = [
    "IncomingMessage",
    "ManagedClaudeSessionManagerProtocol",
    "ManagedClaudeSessionProtocol",
    "MessageNode",
    "MessageState",
    "MessageTree",
    "MessagingWorkflow",
    "OutboundMessenger",
    "SessionStore",
    "TreeQueueManager",
    "parse_cli_event",
]
