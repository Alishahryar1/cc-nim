"""Platform-agnostic messaging layer."""

from .base import MessagingPlatform, SessionManagerInterface, CLISession
from .models import IncomingMessage, OutgoingMessage
from .handler import ClaudeMessageHandler
from .session import SessionStore
from .tree_data import MessageTree, MessageNode, MessageState
from .tree_queue import TreeQueueManager
from .event_parser import parse_cli_event

__all__ = [
    "MessagingPlatform",
    "SessionManagerInterface",
    "CLISession",
    "IncomingMessage",
    "OutgoingMessage",
    "ClaudeMessageHandler",
    "SessionStore",
    "TreeQueueManager",
    "MessageTree",
    "MessageNode",
    "MessageState",
    "parse_cli_event",
]
