"""Deterministic JSONL child used by Codex app-server contract tests."""

import json
import os
import sys
import time
from pathlib import Path

_LINE_LIMIT = 16 * 1024 * 1024


def _emit(message: object) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _emit_together(*messages: object) -> None:
    sys.stdout.write(
        "".join(
            json.dumps(message, separators=(",", ":")) + "\n" for message in messages
        )
    )
    sys.stdout.flush()


def _append(path: str | None, value: str) -> None:
    if path:
        with Path(path).open("a", encoding="utf-8") as output:
            output.write(value + "\n")


def _launch_number(path: str | None) -> int:
    if path is None:
        return 1
    counter = Path(path)
    try:
        value = int(counter.read_text(encoding="utf-8")) + 1
    except FileNotFoundError, ValueError:
        value = 1
    counter.write_text(str(value), encoding="utf-8")
    return value


def _control_path(name: str) -> Path | None:
    root = os.environ.get("FAKE_CODEX_CONTROL_DIR")
    return None if root is None else Path(root) / name


def _signal(name: str) -> None:
    path = _control_path(name)
    if path is not None:
        path.touch()


def _wait_for(name: str) -> None:
    path = _control_path(name)
    if path is None:
        return
    while not path.exists():
        time.sleep(0.01)


def _normal_response(method: str, params: object) -> object:
    if method == "initialize":
        return {
            "userAgent": "fake-codex/1.2.3",
            "codexHome": "/fake/codex",
            "platformFamily": "test",
            "platformOs": "test",
            "futureField": True,
        }
    if method == "model/list":
        cursor = params.get("cursor") if isinstance(params, dict) else None
        if cursor is None:
            return {"data": [{"id": "model-1"}], "nextCursor": "models-2"}
        return {"data": [{"id": "model-2", "futureField": 1}]}
    if method == "permissionProfile/list":
        return {"data": [{"id": ":workspace", "name": "Workspace"}]}
    if method == "collaborationMode/list":
        return {"data": [{"name": "Plan", "mode": "plan"}]}
    if method == "config/read":
        return {"config": {"approval_policy": "on-request"}, "origins": {}}
    if method in {"thread/start", "thread/resume"}:
        thread_id = "thread-1"
        if method == "thread/resume" and isinstance(params, dict):
            supplied = params.get("threadId")
            if isinstance(supplied, str):
                thread_id = supplied
        return {"thread": {"id": thread_id}, "futureField": {"kept": True}}
    if method in {"thread/delete", "turn/interrupt"}:
        return {}
    if method == "turn/start":
        return {"turn": {"id": "turn-1"}}
    return {}


def _approval_request() -> object:
    return {
        "id": "approval-1",
        "method": "item/commandExecution/requestApproval",
        "params": {"availableDecisions": ["accept", "decline"]},
    }


def _approval_resolved() -> object:
    return {
        "method": "serverRequest/resolved",
        "params": {
            "threadId": "thread-1",
            "requestId": "approval-1",
            "futureField": {"kept": True},
        },
    }


def main() -> None:
    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "normal")
    missing_method = os.environ.get("FAKE_CODEX_MISSING_METHOD")
    failing_method = os.environ.get("FAKE_CODEX_FAILING_METHOD")
    log_path = os.environ.get("FAKE_CODEX_REQUEST_LOG")
    launch_number = _launch_number(os.environ.get("FAKE_CODEX_LAUNCH_COUNTER"))
    delayed_model: tuple[object, object] | None = None

    def release_delayed_model() -> None:
        nonlocal delayed_model
        if delayed_model is None:
            return
        delayed_id, delayed_params = delayed_model
        _emit(
            {
                "id": delayed_id,
                "result": _normal_response("model/list", delayed_params),
            }
        )
        delayed_model = None

    for raw_line in sys.stdin:
        message = json.loads(raw_line)
        if not isinstance(message, dict):
            continue
        method = message.get("method")
        request_id = message.get("id")
        if isinstance(method, str):
            _append(log_path, method)
        if method == "initialized":
            if scenario == "hang_on_close":
                for _remaining in sys.stdin:
                    pass
                while True:
                    time.sleep(60)
            continue
        if not isinstance(method, str):
            if request_id == "clock-1":
                _emit({"method": "fixture/currentTime", "params": message})
            elif request_id == "future-1":
                _emit({"method": "fixture/methodNotFound", "params": message})
            elif request_id == "approval-1":
                _append(log_path, "response:approval-1")
                if scenario == "replay_after_response":
                    _emit(_approval_request())
                elif scenario == "hold_resolution_after_replay":
                    _emit_together(
                        _approval_request(),
                        {
                            "method": "fixture/postWriteReplay",
                            "params": {"observed": True},
                        },
                    )
                    _wait_for("release-approval-resolution")
                _emit_together(
                    _approval_resolved(),
                    {"method": "fixture/approvalAnswered", "params": message},
                )
            continue
        params = message.get("params", {})

        if method == "initialize" and scenario == "delay_initialize":
            _signal("initialize-seen")
            _wait_for("release-initialize")
        if method == "initialize" and scenario == "invalid_initialize":
            _emit({"id": request_id, "result": []})
            continue
        if method == "initialize" and scenario == "notification_with_initialize":
            _emit_together(
                {
                    "id": request_id,
                    "result": _normal_response(method, params),
                },
                {
                    "method": "fixture/ready",
                    "params": {"initialized": True},
                },
            )
            continue
        if method == "thread/start" and scenario == "delay_thread_start":
            _signal("thread-start-seen")
            _wait_for("release-thread-start")
        if method == "thread/start" and scenario == "malformed":
            sys.stdout.write("{not-json}\n")
            sys.stdout.flush()
            continue
        if method == "thread/start" and scenario == "malformed_then_notification":
            sys.stdout.write("{not-json}\n")
            _emit(
                {
                    "method": "fixture/afterTerminal",
                    "params": {"mustNotAppear": True},
                }
            )
            continue
        if method == "thread/start" and scenario == "oversized":
            sys.stdout.buffer.write(b'{"oversized":"' + b"x" * _LINE_LIMIT + b'"}\n')
            sys.stdout.buffer.flush()
            continue
        if method == "thread/start" and scenario == "invalid_thread_result":
            _emit({"id": request_id, "result": []})
            continue
        if method == "turn/start" and scenario == "fail_once" and launch_number == 1:
            sys.exit(7)
        if method == missing_method or (
            scenario == "missing_method" and method == "permissionProfile/list"
        ):
            _emit(
                {
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            if method == "config/read":
                release_delayed_model()
            continue
        if method == failing_method:
            _emit(
                {
                    "id": request_id,
                    "error": {"code": -32000, "message": "Injected request failure"},
                }
            )
            if method == "config/read":
                release_delayed_model()
            continue

        if method == "model/list" and not (
            isinstance(params, dict) and params.get("cursor") is not None
        ):
            delayed_model = (request_id, params)
            continue

        _emit({"id": request_id, "result": _normal_response(method, params)})

        if method == "thread/resume":
            if scenario == "replay_on_resume":
                _emit(_approval_request())
            elif scenario == "conflicting_replay_on_resume":
                _emit(
                    {
                        "id": "approval-1",
                        "method": "item/commandExecution/requestApproval",
                        "params": {"availableDecisions": ["decline"]},
                    }
                )

        if method == "config/read":
            release_delayed_model()

        if method == "turn/start":
            if scenario == "flood":
                for index in range(3):
                    _emit({"method": "fixture/flood", "params": {"index": index}})
                continue
            _emit(
                {
                    "method": "future/notification",
                    "params": {"newField": [1, 2, 3]},
                    "futureEnvelopeField": True,
                }
            )
            _emit(
                {
                    "id": "clock-1",
                    "method": "currentTime/read",
                    "params": {},
                }
            )
            _emit(
                {
                    "id": "future-1",
                    "method": "future/serverRequest",
                    "params": {},
                }
            )
            _emit(_approval_request())
            if scenario == "resolve_before_response":
                _emit(_approval_resolved())


if __name__ == "__main__":
    main()
