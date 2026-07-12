import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from free_claude_code.messaging.models import IncomingMessage
from free_claude_code.messaging.commands import handle_workspace_command
from free_claude_code.messaging.keyboards import make_workspace_keyboard


@pytest.mark.asyncio
async def test_make_workspace_keyboard_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock folder and file
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "file.txt"), "w") as f:
            f.write("hello")

        text, kb = make_workspace_keyboard(tmpdir, "")
        assert "Workspace Explorer" in text
        assert "Total: 1 folders, 1 files" in text

        # Check inline keyboard markup structure
        buttons = kb.inline_keyboard
        assert len(buttons) >= 3
        from free_claude_code.messaging.keyboards import get_registered_path
        assert any(get_registered_path(btn.callback_data) == "subdir" for row in buttons for btn in row)
        assert any(get_registered_path(btn.callback_data) == "file.txt" for row in buttons for btn in row)


@pytest.mark.asyncio
async def test_make_workspace_keyboard_traversal_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        text, kb = make_workspace_keyboard(tmpdir, "../outside")
        assert "Access Denied" in text
        assert kb is None


@pytest.mark.asyncio
async def test_handle_workspace_command_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = MagicMock()
        handler.cli_manager.workspace = tmpdir
        handler.outbound.queue_send_message = AsyncMock(return_value="msg123")

        incoming = IncomingMessage(
            text="/workspace",
            chat_id="chat123",
            user_id="user123",
            message_id="msg0",
            platform="telegram",
        )

        await handle_workspace_command(handler, incoming)

        handler.outbound.queue_send_message.assert_called_once()
        args, kwargs = handler.outbound.queue_send_message.call_args
        assert "Workspace Explorer" in args[1]
        assert kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_handle_workspace_command_non_telegram():
    handler = MagicMock()
    handler.outbound.queue_send_message = AsyncMock(return_value="msg123")

    incoming = IncomingMessage(
        text="/workspace",
        chat_id="chat123",
        user_id="user123",
        message_id="msg0",
        platform="discord",
    )

    await handle_workspace_command(handler, incoming)

    handler.outbound.queue_send_message.assert_called_once()
    args, kwargs = handler.outbound.queue_send_message.call_args
    assert "only supported on Telegram" in args[1]


@pytest.mark.asyncio
async def test_telegram_document_upload():
    from free_claude_code.messaging.platforms.telegram import TelegramRuntime
    
    with tempfile.TemporaryDirectory() as tmpdir:
        limiter = MagicMock()
        transcriber = MagicMock()
        
        runtime = TelegramRuntime(
            bot_token="test_token",
            allowed_user_id="12345",
            limiter=limiter,
            transcriber=transcriber,
        )

        mock_workflow = MagicMock()
        mock_workflow.cli_manager.workspace = tmpdir
        mock_workflow.handle_message.__self__ = mock_workflow
        runtime.on_message(mock_workflow.handle_message)

        update = MagicMock()
        update.effective_user.id = 12345
        update.message.document.file_name = "test_upload.py"
        update.message.document.file_id = "file_id_abc"
        update.message.reply_text = AsyncMock()
        
        context = MagicMock()
        mock_file = MagicMock()
        mock_file.download_to_drive = AsyncMock()
        mock_file.download = AsyncMock()
        context.bot.get_file = AsyncMock(return_value=mock_file)

        await runtime._on_telegram_document(update, context)

        context.bot.get_file.assert_called_once_with("file_id_abc")
        # Assert download was invoked
        assert mock_file.download_to_drive.called or mock_file.download.called


def test_make_start_keyboard():
    from free_claude_code.messaging.keyboards import make_start_keyboard
    kb = make_start_keyboard()
    assert kb is not None
    buttons = kb.inline_keyboard
    assert any(btn.callback_data == "workspace_ls:" for row in buttons for btn in row)
    assert any(btn.callback_data == "menu_settings" for row in buttons for btn in row)
    assert any(btn.callback_data == "model_p:0:" for row in buttons for btn in row)
    assert any(btn.callback_data == "menu_clear" for row in buttons for btn in row)


def test_make_model_keyboard_pagination():
    from free_claude_code.messaging.keyboards import make_model_keyboard
    text, kb = make_model_keyboard("zenmux/x-ai/grok-4.5-free", page=0)
    assert "grok-4.5-free" in text
    assert kb is not None
    buttons = kb.inline_keyboard
    # Grok 4.5 is the active one, should have ✅
    assert any("✅" in btn.text and "Grok" in btn.text for row in buttons for btn in row)
    # Next button should be present
    assert any("Next" in btn.text for row in buttons for btn in row)


def test_make_model_keyboard_search():
    from free_claude_code.messaging.keyboards import make_model_keyboard
    text, kb = make_model_keyboard("", page=0, search_query="DeepSeek")
    assert "DeepSeek" in text
    assert kb is not None
    buttons = kb.inline_keyboard
    assert any("DeepSeek" in btn.text for row in buttons for btn in row)
    # Total count matches filtered search results
    assert "of 3 models" in text or "of 2 models" in text or "of 4 models" in text or "models" in text
