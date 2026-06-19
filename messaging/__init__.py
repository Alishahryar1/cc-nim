"""Platform-agnostic messaging layer."""

from .event_parser import parse_cli_event
from .models import IncomingMessage
from .platforms.base import (
    ManagedClaudeSessionManagerProtocol,
    ManagedClaudeSessionProtocol,
    MessagingPlatform,
)
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
    "MessagingPlatform",
    "MessagingWorkflow",
    "SessionStore",
    "TreeQueueManager",
    "parse_cli_event",
]
