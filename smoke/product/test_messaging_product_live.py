import json

import pytest

from messaging.trees import MessageState
from smoke.lib.e2e import FakePlatformDriver, default_cli_events

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("messaging")]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_fake_full_flow_e2e(platform_name: str, tmp_path) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)

    incoming = await driver.send("Please inspect README.", message_id="root_1")

    tree = driver.workflow.tree_queue.get_tree_for_node(incoming.message_id)
    assert tree is not None
    node = tree.get_node(incoming.message_id)
    assert node is not None
    assert node.state == MessageState.COMPLETED
    assert driver.platform.sent
    assert driver.platform.edits
    edit_text = "\n".join(edit["text"] for edit in driver.platform.edits)
    assert "Fake platform answer" in edit_text
    assert "Read" in edit_text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_subagent_control_e2e(platform_name: str, tmp_path) -> None:
    task_events = [
        {"type": "session_info", "session_id": "sess_task"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Need a focused worker."},
                    {
                        "type": "tool_use",
                        "id": "toolu_task",
                        "name": "Task",
                        "input": {"description": "inspect", "prompt": "inspect"},
                    },
                    {"type": "text", "text": "Subagent result rendered."},
                ]
            },
        },
        {"type": "exit", "code": 0, "stderr": None},
    ]
    driver = FakePlatformDriver(platform_name, tmp_path, event_batches=[task_events])

    await driver.send("Delegate this safely.", message_id="root_task")

    edit_text = "\n".join(edit["text"] for edit in driver.platform.edits)
    assert "Subagent" in edit_text
    assert "Tool calls" in edit_text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_commands_stop_clear_stats_e2e(
    platform_name: str, tmp_path
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    root = await driver.send("start work", message_id="root_1")

    await driver.send("/stats", message_id="stats_1")
    await driver.send("/stop", message_id="stop_1", reply_to=root.message_id)
    await driver.send("/clear", message_id="clear_1", reply_to=root.message_id)
    await driver.send("/clear", message_id="clear_all")

    sent_text = "\n".join(sent["text"] for sent in driver.platform.sent)
    assert "Stats" in sent_text
    assert "Stopped" in sent_text
    assert driver.platform.deletes
    assert driver.session_store.load_conversation_snapshot().trees == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_tree_threading_e2e(platform_name: str, tmp_path) -> None:
    batches = [default_cli_events("sess_root"), default_cli_events("sess_branch")]
    driver = FakePlatformDriver(platform_name, tmp_path, event_batches=batches)

    root = await driver.send("root prompt", message_id="root_1")
    branch = await driver.send(
        "branch prompt", message_id="branch_1", reply_to=root.message_id
    )

    tree = driver.workflow.tree_queue.get_tree_for_node(root.message_id)
    assert tree is not None
    branch_node = tree.get_node(branch.message_id)
    assert branch_node is not None
    assert branch_node.parent_id == root.message_id
    assert driver.cli_manager.sessions[1].calls[0]["session_id"] == "sess_root"
    assert driver.cli_manager.sessions[1].calls[0]["fork_session"] is True


@pytest.mark.asyncio
async def test_restart_restore_and_session_persistence_e2e(tmp_path) -> None:
    first = FakePlatformDriver("telegram", tmp_path)
    root = await first.send("persist me", message_id="root_1")
    first.session_store.flush_pending_save()

    session_file = tmp_path / "telegram-sessions.json"
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["conversation"]["trees"]
    assert payload["message_log"]

    restored = FakePlatformDriver("telegram", tmp_path)
    saved = restored.session_store.load_conversation_snapshot()
    assert saved.trees
    assert root.message_id in saved.derive_node_to_tree()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_voice_platform_fake_e2e(platform_name: str, tmp_path) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    driver.platform.register_pending_voice("chat_1", "voice_msg_1", "voice_status_1")

    await driver.send("/clear", message_id="clear_voice", reply_to="voice_msg_1")

    deleted = {entry["message_id"] for entry in driver.platform.deletes}
    assert {"voice_msg_1", "voice_status_1", "clear_voice"} <= deleted
    sent_text = "\n".join(sent["text"] for sent in driver.platform.sent)
    assert "Voice note cancelled" in sent_text
