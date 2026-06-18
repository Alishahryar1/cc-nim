"""Codex CLI adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, cast

from loguru import logger

from .base import CliInvocation, CliParseState, CliTaskRequest

_CODEX_AUTH_ENV_KEY = "FCC_CODEX_API_KEY"
_STRIPPED_CODEX_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "CODEX_API_KEY",
        _CODEX_AUTH_ENV_KEY,
    }
)


class CodexCliAdapter:
    """Client CLI adapter for OpenAI Codex CLI."""

    id = "codex"
    display_name = "Codex CLI"
    default_binary = "codex"
    install_hint = "Install Codex with: npm install -g @openai/codex"
    trace_stage = "codex_cli"
    process_launch_event = "codex_cli.process.launch"
    trace_source = "codex_cli"

    def build_task_invocation(
        self,
        *,
        config: Any,
        request: CliTaskRequest,
        base_env: Mapping[str, str],
    ) -> CliInvocation:
        """Build a Codex JSONL subprocess invocation for managed messaging."""

        env = self._task_env(
            api_url=config.api_url,
            auth_token=config.auth_token,
            base_env=base_env,
        )
        codex_bin = getattr(config, "codex_bin", None) or self.default_binary
        cmd = self._task_command(
            codex_bin=codex_bin,
            prompt=request.prompt,
            session_id=request.session_id,
            fork_session=request.fork_session,
            api_url=config.api_url,
            allowed_dirs=config.allowed_dirs,
            workspace_path=config.workspace_path,
        )
        resume_session_id = (
            request.session_id
            if request.session_id
            and not request.session_id.startswith("pending_")
            and not request.fork_session
            else None
        )
        return CliInvocation(
            argv=tuple(cmd),
            env=env,
            cwd=config.workspace_path,
            trace_metadata={
                "client_cli_id": self.id,
                "resume_session_id": resume_session_id,
                "fork_session": request.fork_session,
                "prompt": request.prompt,
                "cwd": config.workspace_path,
                "codex_binary": codex_bin,
                "cli_argv": cmd,
            },
        )

    def parse_stdout_line(self, line: str, state: CliParseState) -> Iterable[Any]:
        """Parse one Codex JSONL line into existing parser-ready events."""

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if state.log_raw_cli_diagnostics:
                logger.debug("Non-JSON Codex output: {}", line)
            else:
                logger.debug("Non-JSON Codex CLI line: char_len={}", len(line))
            yield {"type": "raw", "content": line}
            return

        if not state.session_id_extracted:
            extracted_id = self.extract_session_id(event)
            if extracted_id:
                state.session_id_extracted = True
                logger.info("Extracted Codex session ID: {}", extracted_id)
                yield {"type": "session_info", "session_id": extracted_id}

        mapped = list(_codex_event_to_parser_events(event, state))
        if mapped:
            yield from mapped
            return
        if event.get("type") in {"response.output_item.done", "response.completed"}:
            return

        yield {"type": "raw", "content": line}

    def extract_session_id(self, event: Any) -> str | None:
        """Extract a Codex session or thread id from supported event shapes."""

        if not isinstance(event, dict):
            return None
        for key in ("session_id", "sessionId", "thread_id", "threadId"):
            if session_id := _string_value(event.get(key)):
                return session_id
        for key in ("thread", "session", "conversation", "metadata"):
            nested = event.get(key)
            if not isinstance(nested, dict):
                continue
            for nested_key in ("id", "session_id", "thread_id", "conversation_id"):
                if session_id := _string_value(nested.get(nested_key)):
                    return session_id
        return None

    def get_launcher_binary_name(self, settings: Any) -> str:
        """Return the configured Codex binary name."""

        configured = getattr(settings, "codex_cli_bin", "")
        return configured or self.default_binary

    def build_launcher_command(
        self,
        *,
        binary_path: str,
        argv: Iterable[str],
        settings: Any,
        proxy_root_url: str,
    ) -> list[str]:
        """Return a Codex command with ephemeral FCC provider config."""

        return [
            binary_path,
            *self._codex_config_args(
                api_url=_ensure_v1_url(proxy_root_url),
                model=getattr(settings, "model", None),
            ),
            *argv,
        ]

    def build_launcher_env(
        self,
        *,
        proxy_root_url: str,
        auth_token: str,
        base_env: Mapping[str, str],
    ) -> dict[str, str]:
        """Return a Codex environment that targets the local proxy provider."""

        env = _base_codex_env(base_env)
        env[_CODEX_AUTH_ENV_KEY] = auth_token.strip() or "fcc-no-auth"
        return env

    def build_model_catalog_config_args(self, catalog_path: str) -> list[str]:
        """Return Codex config args for a generated model catalog."""

        return ["-c", _toml_assignment("model_catalog_json", catalog_path)]

    def _task_env(
        self,
        *,
        api_url: str,
        auth_token: str,
        base_env: Mapping[str, str],
    ) -> dict[str, str]:
        env = _base_codex_env(base_env)
        env[_CODEX_AUTH_ENV_KEY] = auth_token.strip() or "fcc-no-auth"
        env["TERM"] = "dumb"
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _task_command(
        self,
        *,
        codex_bin: str,
        prompt: str,
        session_id: str | None,
        fork_session: bool,
        api_url: str,
        allowed_dirs: list[str],
        workspace_path: str,
    ) -> list[str]:
        common_args = [
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            *self._codex_config_args(api_url=api_url),
        ]
        if session_id and not session_id.startswith("pending_") and not fork_session:
            return [
                codex_bin,
                "exec",
                "resume",
                *common_args,
                session_id,
                prompt,
            ]

        cmd = [
            codex_bin,
            "exec",
            *common_args,
            "-C",
            workspace_path,
        ]
        for directory in allowed_dirs:
            cmd.extend(["--add-dir", directory])
        cmd.append(prompt)
        return cmd

    def _codex_config_args(
        self, *, api_url: str, model: str | None = None
    ) -> list[str]:
        args = [
            "-c",
            _toml_assignment("model_provider", "fcc"),
            "-c",
            _toml_assignment("model_providers.fcc.name", "Free Claude Code"),
            "-c",
            _toml_assignment("model_providers.fcc.base_url", _ensure_v1_url(api_url)),
            "-c",
            _toml_assignment("model_providers.fcc.env_key", _CODEX_AUTH_ENV_KEY),
            "-c",
            _toml_assignment("model_providers.fcc.wire_api", "responses"),
        ]
        if model:
            args.extend(["-c", _toml_assignment("model", model)])
        return args


def _codex_event_to_parser_events(
    event: dict[str, Any], state: CliParseState
) -> Iterable[dict[str, Any]]:
    event_type = event.get("type")
    if event_type in {"error", "turn.failed"}:
        yield {"type": "error", "error": {"message": _event_message(event)}}
        return
    if event_type == "response.failed":
        _finish_response_scope(state)
        yield {"type": "error", "error": {"message": _event_message(event)}}
        return
    if event_type == "response.output_text.delta":
        _mark_streamed_message_item_seen(event, state)
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": str(event.get("delta", ""))},
        }
        return
    if event_type == "response.reasoning_text.delta":
        yield {
            "type": "content_block_delta",
            "index": _event_output_index(event),
            "delta": {
                "type": "thinking_delta",
                "thinking": str(event.get("delta", "")),
            },
        }
        return
    if event_type in {"agent_message", "assistant_message"}:
        text = _event_message(event)
        if text:
            yield {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        return
    if event_type == "response.output_item.done":
        item = event.get("item")
        if isinstance(item, dict) and _mark_response_item_unseen(
            item,
            state,
            response_scope=_response_scope(event, state),
            output_index=_optional_event_output_index(event),
        ):
            yield from _responses_item_to_parser_events(item)
        return
    if event_type == "response.completed":
        response = event.get("response")
        output = response.get("output") if isinstance(response, dict) else None
        response_scope = _response_scope(event, state)
        if isinstance(output, list):
            for output_index, item in enumerate(output):
                if not isinstance(item, dict):
                    continue
                item_mapping = cast(Mapping[str, Any], item)
                if _mark_response_item_unseen(
                    item_mapping,
                    state,
                    response_scope=response_scope,
                    output_index=output_index,
                ):
                    yield from _responses_item_to_parser_events(item_mapping)
        _finish_response_scope(state)
        return


def _responses_item_to_parser_events(
    item: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    item_type = item.get("type")
    if item_type == "message":
        text_parts: list[str] = []
        content = item.get("content")
        if isinstance(content, list):
            text_parts.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
        text = "".join(text_parts)
        if text:
            yield {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": text}]},
            }
        return
    if item_type == "function_call":
        yield {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": str(item.get("call_id", item.get("id", ""))),
                        "name": str(item.get("name", "")),
                        "input": _safe_json_object(item.get("arguments")),
                    }
                ]
            },
        }
        return
    if item_type == "custom_tool_call":
        yield {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": str(item.get("call_id", item.get("id", ""))),
                        "name": str(item.get("name", "")),
                        "input": {"input": _custom_tool_input_text(item.get("input"))},
                    }
                ]
            },
        }


def _event_message(event: Mapping[str, Any]) -> str:
    for key in ("message", "text", "content", "error"):
        value = event.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return str(value["message"])
    response = event.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
    return ""


def _event_output_index(event: Mapping[str, Any]) -> int:
    value = event.get("output_index")
    return value if isinstance(value, int) else 0


def _optional_event_output_index(event: Mapping[str, Any]) -> int | None:
    value = event.get("output_index")
    return value if isinstance(value, int) else None


def _mark_response_item_unseen(
    item: Mapping[str, Any],
    state: CliParseState,
    *,
    response_scope: str | None = None,
    output_index: int | None = None,
) -> bool:
    item_key = _response_item_key(
        item,
        response_scope=response_scope,
        output_index=output_index,
    )
    if item_key is None:
        return True
    if item_key in state.responses_seen_output_keys:
        return False
    state.responses_seen_output_keys.add(item_key)
    return True


def _mark_streamed_message_item_seen(
    event: Mapping[str, Any], state: CliParseState
) -> None:
    item_id = event.get("item_id")
    if isinstance(item_id, str) and item_id:
        state.responses_seen_output_keys.add(f"message:{item_id}")
    output_index = _optional_event_output_index(event)
    if output_index is not None:
        state.responses_seen_output_keys.add(
            _response_output_index_key(
                "message",
                _response_scope(event, state),
                output_index,
            )
        )


def _response_item_key(
    item: Mapping[str, Any],
    *,
    response_scope: str | None = None,
    output_index: int | None = None,
) -> str | None:
    item_type = item.get("type")
    if (
        item_type == "message"
        and response_scope is not None
        and output_index is not None
    ):
        return _response_output_index_key(item_type, response_scope, output_index)
    for key in ("id", "call_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return f"{item_type}:{value}"
    return None


def _response_scope(event: Mapping[str, Any], state: CliParseState) -> str:
    if state.responses_current_output_scope is not None:
        return state.responses_current_output_scope
    response_id = _event_response_id(event)
    if response_id is not None:
        scope = f"response:{response_id}"
    else:
        scope = f"implicit:{state.responses_next_implicit_output_scope}"
        state.responses_next_implicit_output_scope += 1
    state.responses_current_output_scope = scope
    return scope


def _finish_response_scope(state: CliParseState) -> None:
    state.responses_current_output_scope = None


def _event_response_id(event: Mapping[str, Any]) -> str | None:
    for key in ("response_id", "responseId"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    response = event.get("response")
    if isinstance(response, dict):
        value = response.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def _response_output_index_key(
    item_type: object, response_scope: str, output_index: int
) -> str:
    return f"{item_type}_output_index:{response_scope}:{output_index}"


def _safe_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _custom_tool_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _base_codex_env(base_env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in base_env.items()
        if key not in _STRIPPED_CODEX_ENV_KEYS and not key.startswith("OPENAI_")
    }


def _ensure_v1_url(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def _toml_assignment(key: str, value: str) -> str:
    return f"{key}={json.dumps(value)}"


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


CODEX_CLI_ADAPTER = CodexCliAdapter()
