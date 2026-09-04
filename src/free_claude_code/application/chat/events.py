"""Bounded in-process fan-out for observable Chat state changes."""

import asyncio
from collections.abc import AsyncIterator

from free_claude_code.core.json_types import JsonObject

from .models import ChatEventOverflowError, ChatPublishedEvent

_DEFAULT_QUEUE_SIZE = 128


class _Closed:
    pass


_CLOSED = _Closed()


class _Overflow:
    def __init__(self, cursor: int) -> None:
        self.cursor = cursor


type _QueueItem = ChatPublishedEvent | _Overflow | _Closed


class ChatEventSubscription:
    """One observer's independent view of the process-local Chat feed."""

    def __init__(
        self,
        publisher: ChatEventPublisher,
        *,
        cursor: int,
        queue_size: int,
    ) -> None:
        self._publisher = publisher
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=queue_size)
        self.cursor = cursor
        self._closed = False

    def __aiter__(self) -> AsyncIterator[ChatPublishedEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[ChatPublishedEvent]:
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            if isinstance(item, _Overflow):
                raise ChatEventOverflowError(item.cursor)
            if isinstance(item, ChatPublishedEvent):
                yield item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.unsubscribe(self)
        self.signal(_CLOSED)

    def signal(self, item: _QueueItem) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(item)


class ChatEventPublisher:
    """Publish without allowing one observer to backpressure Chat work."""

    def __init__(self, *, queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        if queue_size <= 0:
            raise ValueError("Chat event queue size must be positive.")
        self._queue_size = queue_size
        self._sequence = 0
        self._subscriptions: set[ChatEventSubscription] = set()
        self._closed = False

    def subscribe(self) -> ChatEventSubscription:
        if self._closed:
            raise RuntimeError("Chat event publisher is closed.")
        subscription = ChatEventSubscription(
            self,
            cursor=self._sequence,
            queue_size=self._queue_size,
        )
        self._subscriptions.add(subscription)
        return subscription

    def publish(self, event: str, data: JsonObject) -> ChatPublishedEvent:
        if self._closed:
            raise RuntimeError("Chat event publisher is closed.")
        self._sequence += 1
        published = ChatPublishedEvent(event=event, id=self._sequence, data={**data})
        for subscription in tuple(self._subscriptions):
            try:
                subscription._queue.put_nowait(published)
            except asyncio.QueueFull:
                self._subscriptions.discard(subscription)
                subscription.signal(_Overflow(self._sequence))
        return published

    def unsubscribe(self, subscription: ChatEventSubscription) -> None:
        self._subscriptions.discard(subscription)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.disconnect_subscribers()

    def disconnect_subscribers(self) -> None:
        """Finish observer feeds while allowing owned work to settle and publish."""
        subscriptions = tuple(self._subscriptions)
        self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.signal(_CLOSED)
