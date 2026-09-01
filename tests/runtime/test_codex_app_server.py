"""Contracts for the concrete Codex app-server stdio owner."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from free_claude_code.application.work import (
    CodexCompatibilityError,
    CodexConnectionError,
    CodexConnectionLost,
    CodexNotification,
    CodexProtocolError,
    CodexServerRequest,
    CodexThreadSettings,
    CodexTurnSettings,
)
from free_claude_code.cli.launchers.codex import CodexModelCatalogPlan
from free_claude_code.config.settings import Settings
from free_claude_code.runtime import codex_app_server
from free_claude_code.runtime.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerProcessPlan,
    prepare_codex_app_server_process_plan,
)

_FAKE_SERVER = (
    Path(__file__).resolve().parents[1] / "fixtures" / "fake_codex_app_server.py"
)


def _client(
    tmp_path: Path,
    *,
    scenario: str = "normal",
    request_log: Path | None = None,
    launch_counter: Path | None = None,
) -> CodexAppServerClient:
    env = os.environ.copy()
    env["FAKE_CODEX_SCENARIO"] = scenario
    if request_log is not None:
        env["FAKE_CODEX_REQUEST_LOG"] = str(request_log)
    if launch_counter is not None:
        env["FAKE_CODEX_LAUNCH_COUNTER"] = str(launch_counter)

    async def plan() -> CodexAppServerProcessPlan:
        return CodexAppServerProcessPlan(
            command=(sys.executable, str(_FAKE_SERVER)),
            env=env,
            binary_path=sys.executable,
            version="codex-cli 1.2.3",
        )

    return CodexAppServerClient(plan, client_version="9.8.7")


async def _next(
    events: AsyncIterator[CodexNotification | CodexServerRequest | CodexConnectionLost],
) -> CodexNotification | CodexServerRequest | CodexConnectionLost:
    async with asyncio.timeout(2):
        return await anext(events)


@pytest.mark.asyncio
async def test_production_plan_reuses_launcher_config_and_scrubbed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def find_binary(command: str, *, path: str | None) -> str | None:
        assert command == "codex"
        assert path == "test-path"
        return "resolved-codex.exe"

    monkeypatch.setattr(codex_app_server.shutil, "which", find_binary)
    monkeypatch.setattr(
        codex_app_server,
        "codex_model_catalog_plan",
        lambda _proxy, _settings: CodexModelCatalogPlan(),
    )
    monkeypatch.setattr(
        codex_app_server,
        "_read_codex_version",
        lambda _binary, _env: "codex-cli 1.2.3",
    )
    settings = Settings(
        model="nvidia_nim/test-model",
        proxy_auth_enabled=False,
        proxy_auth_token="proxy-token",
    )

    plan = await prepare_codex_app_server_process_plan(
        settings=settings,
        proxy_root_url="http://127.0.0.1:8082",
        base_env={
            "Path": "test-path",
            "OPENAI_API_KEY": "must-not-leak",
            "CODEX_HOME": "keep-home",
        },
    )

    assert plan.binary_path == "resolved-codex.exe"
    assert plan.version == "codex-cli 1.2.3"
    assert plan.command[0] == "resolved-codex.exe"
    assert 'model_provider="fcc"' in plan.command
    assert 'model_providers.fcc.wire_api="responses"' in plan.command
    assert plan.command[-2:] == ("app-server", "--stdio")
    assert "OPENAI_API_KEY" not in plan.env
    assert plan.env["CODEX_HOME"] == "keep-home"
    assert plan.env["NO_PROXY"] == "127.0.0.1,localhost,::1"


@pytest.mark.asyncio
async def test_initializes_once_and_reads_concurrent_native_catalogs(
    tmp_path: Path,
) -> None:
    request_log = tmp_path / "requests.log"
    client = _client(tmp_path, request_log=request_log)
    try:
        initialization, controls = await asyncio.gather(
            client.initialize(),
            client.controls(cwd=str(tmp_path)),
        )

        availability = await client.availability()
        assert availability.available is True
        assert availability.version == "codex-cli 1.2.3"
        assert initialization.user_agent == "fake-codex/1.2.3"
        assert initialization.connection_id
        assert [model["id"] for model in controls.models] == ["model-1", "model-2"]
        assert controls.permission_profiles[0]["id"] == ":workspace"
        assert controls.collaboration_modes[0]["mode"] == "plan"
        assert controls.config["approval_policy"] == "on-request"
        methods = request_log.read_text(encoding="utf-8").splitlines()
        assert methods.count("initialize") == 1
        assert methods.count("initialized") == 1
        assert methods.count("model/list") == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_thread_turn_events_and_bidirectional_server_requests(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    events = client.events()
    try:
        thread = await client.start_thread(
            CodexThreadSettings(
                cwd=str(tmp_path),
                model="provider/model",
                approval_policy="on-request",
                permission_profile=":workspace",
            )
        )
        assert thread.thread_id == "thread-1"
        assert thread.response["futureField"] == {"kept": True}

        resumed = await client.resume_thread(
            "thread-existing",
            CodexThreadSettings(cwd=str(tmp_path)),
        )
        assert resumed.thread_id == "thread-existing"

        turn = await client.start_turn(
            thread_id=thread.thread_id,
            text="hello",
            settings=CodexTurnSettings(
                effort="high",
                collaboration_mode={"mode": "plan"},
            ),
        )
        assert turn.turn_id == "turn-1"

        notification = await _next(events)
        assert isinstance(notification, CodexNotification)
        assert notification.method == "future/notification"

        observed: dict[str, object] = {}
        approval: CodexServerRequest | None = None
        while len(observed) < 2 or approval is None:
            event = await _next(events)
            if isinstance(event, CodexServerRequest):
                approval = event
                continue
            assert isinstance(event, CodexNotification)
            observed[event.method] = event.params
        assert set(observed) == {"fixture/currentTime", "fixture/methodNotFound"}
        current_time = observed["fixture/currentTime"]
        assert isinstance(current_time, dict)
        assert isinstance(current_time["result"], dict)
        assert isinstance(current_time["result"]["currentTimeAt"], int)
        method_not_found = observed["fixture/methodNotFound"]
        assert isinstance(method_not_found, dict)
        assert isinstance(method_not_found["error"], dict)
        assert method_not_found["error"]["code"] == -32601
        assert approval.method == "item/commandExecution/requestApproval"
        await client.respond(
            connection_id=approval.connection_id,
            request_id=approval.request_id,
            result={"decision": "decline"},
        )
        answered = await _next(events)
        assert isinstance(answered, CodexNotification)
        assert answered.method == "fixture/approvalAnswered"

        await client.interrupt_turn(
            thread_id=thread.thread_id,
            turn_id=turn.turn_id,
        )
        await client.delete_thread(thread.thread_id)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["malformed", "oversized"])
async def test_invalid_protocol_disconnects_without_hanging_pending_calls(
    tmp_path: Path,
    scenario: str,
) -> None:
    client = _client(tmp_path, scenario=scenario)
    events = client.events()
    try:
        with pytest.raises(CodexProtocolError):
            await client.start_thread(CodexThreadSettings(cwd=str(tmp_path)))
        lost = await _next(events)
        assert isinstance(lost, CodexConnectionLost)
        assert lost.connection_id
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_process_failure_is_not_replayed_and_next_call_starts_cleanly(
    tmp_path: Path,
) -> None:
    request_log = tmp_path / "requests.log"
    launch_counter = tmp_path / "launch-count"
    client = _client(
        tmp_path,
        scenario="fail_once",
        request_log=request_log,
        launch_counter=launch_counter,
    )
    try:
        thread = await client.start_thread(CodexThreadSettings(cwd=str(tmp_path)))
        with pytest.raises(CodexConnectionError):
            await client.start_turn(
                thread_id=thread.thread_id,
                text="first",
                settings=CodexTurnSettings(),
            )
        await asyncio.sleep(0.05)
        assert launch_counter.read_text(encoding="utf-8") == "1"
        assert (
            request_log.read_text(encoding="utf-8").splitlines().count("turn/start")
            == 1
        )

        restarted = await client.start_turn(
            thread_id=thread.thread_id,
            text="second",
            settings=CodexTurnSettings(),
        )
        assert restarted.turn_id == "turn-1"
        assert launch_counter.read_text(encoding="utf-8") == "2"
        assert (
            request_log.read_text(encoding="utf-8").splitlines().count("turn/start")
            == 2
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_missing_required_catalog_method_is_actionable(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, scenario="missing_method")
    try:
        with pytest.raises(
            CodexCompatibilityError,
            match=r"codex-cli 1\.2\.3.*permissionProfile/list.*update Codex",
        ):
            await client.controls(cwd=str(tmp_path))
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_event_overflow_fails_the_connection_instead_of_dropping_silently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(codex_app_server, "_EVENT_QUEUE_LIMIT", 2)
    client = _client(tmp_path, scenario="flood")
    events = client.events()
    try:
        thread = await client.start_thread(CodexThreadSettings(cwd=str(tmp_path)))
        await client.start_turn(
            thread_id=thread.thread_id,
            text="flood",
            settings=CodexTurnSettings(),
        )
        lost = await _next(events)
        assert isinstance(lost, CodexConnectionLost)
        assert "overflowed" in lost.message
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_escalates_for_child_ignoring_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(codex_app_server, "_GRACEFUL_CLOSE_SECONDS", 0.05)
    monkeypatch.setattr(codex_app_server, "_TERMINATE_SECONDS", 0.05)
    client = _client(tmp_path, scenario="hang_on_close")
    await client.initialize()

    await client.close()
    await client.close()

    availability = await client.availability()
    assert availability.available is False
