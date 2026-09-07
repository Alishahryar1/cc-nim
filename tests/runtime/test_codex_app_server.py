import asyncio
import os
import sys
from pathlib import Path

import pytest

from free_claude_code.application.code_sessions.models import CodeUnavailableError
from free_claude_code.runtime.codex_app_server import CodexAppServer
from tests.code_sessions_support import FakeHarness


async def connect(tmp_path, mode):
    events = []
    completed = asyncio.Event()
    prompted = asyncio.Event()

    async def receive(event):
        events.append(event)
        if event.kind == "turn_completed":
            completed.set()
        if event.kind == "prompt":
            prompted.set()

    native = CodexAppServer(
        [sys.executable, str(Path(__file__).with_name("codex_fake_process.py")), mode],
        dict(os.environ),
        str(tmp_path),
        receive,
        model_slugs={"provider/model": "provider/model"},
        fingerprints={"provider/model": "capabilities-1"},
    )
    await native.start()
    return native, events, completed, prompted


@pytest.mark.asyncio
async def test_jsonl_large_unicode_events_can_precede_rpc_ack(tmp_path):
    native, events, completed, _ = await connect(tmp_path, "large")
    try:
        assert (await native.create_thread()).id == "native-1"
        assert (
            await native.start_turn(
                "hello", FakeHarness().prepare("provider/model", None), "input-1"
            )
            == "turn-1"
        )
        await asyncio.wait_for(completed.wait(), 3)
        texts = [
            event.item.text for event in events if event.item and event.item.complete
        ]
        assert texts == ["snow ☃ " * 16000]
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_server_rpc_during_start_preserves_numeric_zero_id(tmp_path):
    native, events, completed, prompted = await connect(tmp_path, "prompt")
    try:
        await native.create_thread()
        await native.start_turn(
            "hello", FakeHarness().prepare("provider/model", None), "input-1"
        )
        await asyncio.wait_for(prompted.wait(), 3)
        response = native.prepare_answer(0, {"choice": "0"})
        await native.respond(0, response)
        await asyncio.wait_for(completed.wait(), 3)
        assert any(event.item and event.item.text == "accept" for event in events)
        assert any(
            event.kind == "resolved" and event.request_id == 0 for event in events
        )
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_cancelled_creation_waiter_retains_late_native_identity(tmp_path):
    native, _, _, _ = await connect(tmp_path, "delayed-create")
    try:
        creation = asyncio.create_task(native.create_thread())
        await native.rpc("test/barrier", {})
        creation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creation
        await native.rpc("test/release-create", {})
        assert native.thread_id == "native-1"
    finally:
        await native.close()


@pytest.mark.asyncio
async def test_eof_completes_pending_calls_and_cleanup(tmp_path):
    native, _, _, _ = await connect(tmp_path, "large")
    try:
        with pytest.raises(CodeUnavailableError):
            await native.rpc("test/eof", {})
    finally:
        await native.close()
    assert native.process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["test/malformed", "test/malformed-flood"])
async def test_malformed_frame_reports_closed_only_after_process_termination(
    tmp_path, method
):
    closed = asyncio.Event()
    returncodes = []

    async def receive(event):
        if event.kind == "closed":
            returncodes.append(native.process.returncode)
            closed.set()

    native = CodexAppServer(
        [
            sys.executable,
            str(Path(__file__).with_name("codex_fake_process.py")),
            "large",
        ],
        dict(os.environ),
        str(tmp_path),
        receive,
    )
    try:
        await native.start()
        with pytest.raises(CodeUnavailableError):
            await native.rpc(method, {})
        await asyncio.wait_for(closed.wait(), 8)
        assert returncodes and returncodes[0] is not None
        assert native.process.stdout is not None and native.process.stdout.at_eof()
    finally:
        if native.process.stdout is not None and not native.process.stdout.at_eof():
            await native.process.stdout.read()
        await native.close()


@pytest.mark.asyncio
async def test_spawned_agent_prompt_is_visible_in_its_registered_root_session(tmp_path):
    native, events, completed, prompted = await connect(tmp_path, "child-prompt")
    try:
        await native.create_thread()
        await native.start_turn(
            "delegate", FakeHarness().prepare("provider/model", None), "input-1"
        )
        await asyncio.wait_for(prompted.wait(), 3)
        event = next(event for event in events if event.kind == "prompt")
        assert event.thread_id == "native-1"
        assert event.prompt.turn_id is None
        assert event.prompt.raw["threadId"] == "child-1"
        await native.respond(0, native.prepare_answer(0, {"choice": "0"}))
        await asyncio.wait_for(completed.wait(), 3)
    finally:
        await native.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [True, False])
async def test_turn_start_resets_sticky_effort_and_preserves_client_identity(
    tmp_path, monkeypatch, reasoning
):
    async def receive(event):
        pass

    native = CodexAppServer(
        [],
        {},
        str(tmp_path),
        receive,
        model_slugs={"provider/model": "native-slug"},
        reasoning={"provider/model": reasoning},
    )
    native.thread_id = "thread"
    requests = []

    async def rpc(method, params):
        assert method == "turn/start"
        requests.append(params)
        return {"turn": {"id": "turn"}}

    monkeypatch.setattr(native, "rpc", rpc)
    harness = FakeHarness()
    for effort in (None, "high", None):
        await native.start_turn(
            "hello", harness.prepare("provider/model", effort), "operation"
        )
    assert [request.get("effort") for request in requests] == (
        ["medium", "high", "medium"] if reasoning else [None, None, None]
    )
    assert all(
        request["clientUserMessageId"] == "operation"
        and request["model"] == "native-slug"
        for request in requests
    )
