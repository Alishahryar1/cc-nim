import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.voice import PendingVoiceRegistry

TELEGRAM_CHAT = MessageScope(platform="telegram", chat_id="chat")
DISCORD_CHAT = MessageScope(platform="discord", chat_id="chat")


@pytest.mark.asyncio
async def test_pending_voice_registry_tracks_voice_and_status_ids():
    registry = PendingVoiceRegistry()

    await registry.register(TELEGRAM_CHAT, "voice-1", "status-1")

    assert await registry.is_pending(TELEGRAM_CHAT, "voice-1") is True
    assert await registry.cancel(TELEGRAM_CHAT, "status-1") == (
        "voice-1",
        "status-1",
    )
    assert await registry.is_pending(TELEGRAM_CHAT, "voice-1") is False


@pytest.mark.asyncio
async def test_pending_voice_registry_isolates_platform_scopes():
    registry = PendingVoiceRegistry()
    await registry.register(DISCORD_CHAT, "voice-1", "status-1")
    await registry.register(TELEGRAM_CHAT, "voice-1", "status-1")

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") == (
        "voice-1",
        "status-1",
    )
    assert await registry.is_pending(DISCORD_CHAT, "voice-1") is True
    assert await registry.cancel(DISCORD_CHAT, "voice-1") == (
        "voice-1",
        "status-1",
    )


@pytest.mark.asyncio
async def test_pending_voice_registry_complete_removes_entries():
    registry = PendingVoiceRegistry()

    await registry.register(TELEGRAM_CHAT, "voice-1", "status-1")
    await registry.complete(TELEGRAM_CHAT, "voice-1", "status-1")

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None
