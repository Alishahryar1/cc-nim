"""Heuristic parser for text-emitted tool calls."""

import json
import re
import uuid
from enum import Enum
from typing import Any

from loguru import logger

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"

_XML_TOOL_CALL_RE = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_]+)"
    r"(?:\s[^>]*)?>"
    r"(?P<params>.*?)"
    r"</(?P=name)>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)>(?P<value>.*?)</(?P=key)>",
    re.DOTALL,
)
_KNOWN_TOOL_NAMES = frozenset(
    {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Task",
        "WebFetch",
        "Question",
        "Skill",
        "BashTool",
    }
)


def _clean_xml_parameter(key: str, value: str) -> str:
    """Clean XML-style tool call parameter values.

    If the parameter is a single-line or metadata field, completely strip all leading
    and trailing whitespace.

    If the parameter is a multi-line code/text block (like ``oldText``, ``newText``,
    ``fileContent``, ``content``), strip only the formatting artifacts from XML blocks:
    exactly one leading and trailing newline (and any spaces/tabs preceding/succeeding
    them on those lines), without touching the inner content.
    """
    if not value.strip():
        return ""

    single_line_keys = {
        "filepath",
        "path",
        "pattern",
        "include",
        "url",
        "command",
        "id",
        "name",
        "task",
        "query",
    }
    if key.lower() in single_line_keys:
        return value.strip()

    cleaned = value
    leading_match = re.match(r"^[ \t]*\r?\n", cleaned)
    if leading_match:
        cleaned = cleaned[leading_match.end() :]

    trailing_match = re.search(r"\r?\n[ \t]*$", cleaned)
    if trailing_match:
        cleaned = cleaned[: trailing_match.start()]

    return cleaned


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


class HeuristicToolParser:
    """
    Stateful parser for raw text tool calls.

    Some OpenAI-compatible models emit tool calls as text rather than structured
    chunks. This parser converts the common ``● <function=...>`` form into
    Anthropic-style ``tool_use`` blocks.
    """

    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )

    def __init__(self):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}

    @staticmethod
    def extract_xml_tool_calls(
        text: str,
        valid_tool_names: set[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract XML-style tool calls like ``<Read><filePath>/tmp/x</filePath></Read>``.

        NIM models with ``thinking=True`` emit tool calls in this format inside
        ``reasoning_content`` instead of structured ``tool_calls``.  This method
        is stateless (works on complete text) and returns the remaining text
        after removing matched tool calls plus a list of detected tool_use dicts.
        """
        allowed = (
            valid_tool_names if valid_tool_names is not None else _KNOWN_TOOL_NAMES
        )
        detected: list[dict[str, Any]] = []
        remaining = text

        for match in _XML_TOOL_CALL_RE.finditer(text):
            tool_name = match.group("name")
            params_text = match.group("params")

            if tool_name not in allowed:
                continue

            params: dict[str, str] = {}
            for param_match in _XML_PARAM_RE.finditer(params_text):
                key = param_match.group("key")
                value = param_match.group("value")
                params[key] = _clean_xml_parameter(key, value)

            if not params and (tool_name in _KNOWN_TOOL_NAMES or params_text.strip()):
                continue

            detected.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_xml_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": params,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected XML-style tool call '{}' with {} params",
                tool_name,
                len(params),
            )

        if detected:
            remaining = _XML_TOOL_CALL_RE.sub("", text)
            if not remaining:
                remaining = ""

        return remaining, detected

    def _extract_web_tool_json_calls(self) -> tuple[str, list[dict[str, Any]]]:
        detected_tools: list[dict[str, Any]] = []

        for match in self._WEB_TOOL_JSON_PATTERN.finditer(self._buffer):
            try:
                tool_input = json.loads(match.group("json"))
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_input, dict):
                continue

            tool_name = match.group("tool")
            if tool_name == "WebFetch" and "url" not in tool_input:
                continue
            if tool_name == "WebSearch" and "query" not in tool_input:
                continue

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected JSON-style tool call '{}'",
                tool_name,
            )

        if not detected_tools:
            return self._buffer, []

        return "", detected_tools

    def _strip_control_tokens(self, text: str) -> str:
        return _CONTROL_TOKEN_RE.sub("", text)

    def _split_incomplete_control_token_tail(self) -> str:
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""

        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed text and return safe text plus detected tool calls."""
        self._buffer += text
        self._buffer = self._strip_control_tokens(self._buffer)
        self._buffer, detected_tools = self._extract_web_tool_json_calls()
        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break

                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                elif len(self._buffer) > 100:
                    filtered_output_parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                else:
                    break

            if self._state == ParserState.PARSING_PARAMETERS:
                finished_tool_call = False

                while True:
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        pre_match_text = self._buffer[: param_match.start()]
                        if pre_match_text:
                            filtered_output_parts.append(pre_match_text)

                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self._current_parameters[key] = val
                        self._buffer = self._buffer[param_match.end() :]
                    else:
                        break

                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    if "<parameter=" not in self._buffer:
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": self._current_parameters,
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        return "".join(filtered_output_parts), detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any remaining tool call in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
            partial_matches = re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            )
            for match in partial_matches:
                key = match.group(1).strip()
                val = match.group(2).strip()
                self._current_parameters[key] = val

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": self._current_parameters,
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
