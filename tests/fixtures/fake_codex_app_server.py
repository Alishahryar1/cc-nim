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


def main() -> None:
    scenario = os.environ.get("FAKE_CODEX_SCENARIO", "normal")
    log_path = os.environ.get("FAKE_CODEX_REQUEST_LOG")
    launch_number = _launch_number(os.environ.get("FAKE_CODEX_LAUNCH_COUNTER"))
    delayed_model: tuple[object, object] | None = None

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
                _emit({"method": "fixture/approvalAnswered", "params": message})
            continue
        params = message.get("params", {})

        if method == "thread/start" and scenario == "malformed":
            sys.stdout.write("{not-json}\n")
            sys.stdout.flush()
            continue
        if method == "thread/start" and scenario == "oversized":
            sys.stdout.buffer.write(b'{"oversized":"' + b"x" * _LINE_LIMIT + b'"}\n')
            sys.stdout.buffer.flush()
            continue
        if method == "turn/start" and scenario == "fail_once" and launch_number == 1:
            sys.exit(7)
        if scenario == "missing_method" and method == "permissionProfile/list":
            _emit(
                {
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            continue

        if method == "model/list" and not (
            isinstance(params, dict) and params.get("cursor") is not None
        ):
            delayed_model = (request_id, params)
            continue

        _emit({"id": request_id, "result": _normal_response(method, params)})

        if method == "config/read" and delayed_model is not None:
            delayed_id, delayed_params = delayed_model
            _emit(
                {
                    "id": delayed_id,
                    "result": _normal_response("model/list", delayed_params),
                }
            )
            delayed_model = None

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
            _emit(
                {
                    "id": "approval-1",
                    "method": "item/commandExecution/requestApproval",
                    "params": {"availableDecisions": ["accept", "decline"]},
                }
            )


if __name__ == "__main__":
    main()
