"""Abstract base class for messaging platforms."""

from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Optional, Any, Protocol, Tuple, runtime_checkable
from .models import IncomingMessage


class CLISession(ABC):
    """Abstract base for CLI session - avoid circular import from cli package."""

    @abstractmethod
    async def start_task(self, prompt: str, session_id: Optional[str] = None):
        """Start a task in the CLI session."""
        pass

    @property
    @abstractmethod
    def is_busy(self) -> bool:
        """Check if session is busy."""
        pass


@runtime_checkable
class SessionManagerInterface(Protocol):
    """
    Protocol for session managers to avoid tight coupling with cli package.

    Implementations: CLISessionManager
    """

    async def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> Tuple[CLISession, str, bool]:
        """
        Get an existing session or create a new one.

        Returns: Tuple of (session, session_id, is_new_session)
        """
        ...

    async def register_real_session_id(self, temp_id: str, real_session_id: str) -> bool:
        """Register the real session ID from CLI output."""
        ...

    async def stop_all(self) -> None:
        """Stop all sessions."""
        ...

    def get_stats(self) -> dict:
        """Get session statistics."""
        ...


class MessagingPlatform(ABC):
    """
    Base class for all messaging platform adapters.

    Implement this to add support for Telegram, Discord, Slack, etc.
    """

    name: str = "base"

    @abstractmethod
    async def start(self) -> None:
        """Initialize and connect to the messaging platform."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and cleanup resources."""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> str:
        """
        Send a message to a chat.

        Args:
            chat_id: The chat/channel ID to send to
            text: Message content
            reply_to: Optional message ID to reply to
            parse_mode: Optional formatting mode ("markdown", "html")

        Returns:
            The message ID of the sent message
        """
        pass

    @abstractmethod
    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        """
        Edit an existing message.

        Args:
            chat_id: The chat/channel ID
            message_id: The message ID to edit
            text: New message content
            parse_mode: Optional formatting mode
        """
        pass

    @abstractmethod
    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> Optional[str]:
        """
        Enqueue a message to be sent.

        If fire_and_forget is True, returns None immediately.
        Otherwise, waits for the rate limiter and returns message ID.
        """
        pass

    @abstractmethod
    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> None:
        """
        Enqueue a message edit.

        If fire_and_forget is True, returns immediately.
        Otherwise, waits for the rate limiter.
        """
        pass

    @abstractmethod
    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """
        Register a message handler callback.

        The handler will be called for each incoming message.

        Args:
            handler: Async function that processes incoming messages
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if the platform is connected."""
        return False
