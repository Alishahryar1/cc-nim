"""Tests for NVIDIA NIM request policy helpers."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from config.nim import NimSettings
from core.anthropic import set_if_not_none
from providers.nvidia_nim.request_options import (
    _set_extra,
)
from providers.nvidia_nim.request_options import (
    build_nim_request_body as build_request_body,
)
from providers.nvidia_nim.retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_content,
)
from providers.nvidia_nim.tool_schema import (
    NIM_TOOL_ARGUMENT_ALIASES_KEY,
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)

GREP_SCHEMA_FROM_SERVER_LOG: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "The regular expression"},
        "path": {"type": "string", "description": "File or directory to search"},
        "glob": {"type": "string", "description": "Glob to filter files"},
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
        },
        "-A": {"type": "number", "description": "Lines after match"},
        "-B": {"type": "number", "description": "Lines before match"},
        "-C": {"type": "number", "description": "Lines around match"},
        "-i": {"type": "boolean", "description": "Case insensitive"},
        "-n": {"type": "boolean", "description": "Show line numbers"},
        "type": {"type": "string", "description": "File type to search"},
    },
    "additionalProperties": False,
    "required": ["pattern"],
}


@pytest.fixture
def req():
    r = MagicMock()
    r.model = "test"
    r.messages = [MagicMock(role="user", content="hi")]
    r.max_tokens = 100
    r.system = None
    r.temperature = None
    r.top_p = None
    r.stop_sequences = None
    r.tools = None
    r.tool_choice = None
    r.extra_body = None
    r.top_k = None
    return r


class TestSetIfNotNone:
    def test_value_not_none_sets(self):
        body = {}
        set_if_not_none(body, "key", "value")
        assert body["key"] == "value"

    def test_value_none_skips(self):
        body = {}
        set_if_not_none(body, "key", None)
        assert "key" not in body


class TestSetExtra:
    def test_key_in_extra_body_skips(self):
        extra = {"top_k": 42}
        _set_extra(extra, "top_k", 10)
        assert extra["top_k"] == 42

    def test_value_none_skips(self):
        extra = {}
        _set_extra(extra, "top_k", None)
        assert "top_k" not in extra

    def test_value_equals_ignore_value_skips(self):
        extra = {}
        _set_extra(extra, "top_k", -1, ignore_value=-1)
        assert "top_k" not in extra

    def test_value_set_when_valid(self):
        extra = {}
        _set_extra(extra, "top_k", 10, ignore_value=-1)
        assert extra["top_k"] == 10


class TestBuildRequestBody:
    def test_max_tokens_capped_by_nim(self, req):
        req.max_tokens = 100000
        nim = NimSettings(max_tokens=4096)
        body = build_request_body(req, nim, thinking_enabled=True)
        assert body["max_tokens"] == 4096

    def test_presence_penalty_included_when_nonzero(self, req):
        nim = NimSettings(presence_penalty=0.5)
        body = build_request_body(req, nim, thinking_enabled=True)
        assert body["presence_penalty"] == 0.5

    def test_include_stop_str_in_output_not_sent(self, req):
        body = build_request_body(req, NimSettings(), thinking_enabled=True)
        assert "include_stop_str_in_output" not in body.get("extra_body", {})

    def test_parallel_tool_calls_included(self, req):
        nim = NimSettings(parallel_tool_calls=False)
        body = build_request_body(req, nim, thinking_enabled=True)
        assert body["parallel_tool_calls"] is False

    def test_tool_schema_boolean_subschemas_are_removed_without_mutating_request(
        self, req
    ):
        tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": False},
                "blocked": False,
                "nested": {"type": "object", "additionalProperties": False},
                "choice": {"anyOf": [False, {"type": "string"}]},
            },
            "additionalProperties": False,
            "required": ["query"],
        }
        req.tools = [
            SimpleNamespace(
                name="search",
                description="search",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, NimSettings(), thinking_enabled=False)

        parameters = body["tools"][0]["function"]["parameters"]
        properties = parameters["properties"]
        assert "additionalProperties" not in parameters
        assert "blocked" not in properties
        assert "additionalProperties" not in properties["nested"]
        assert properties["choice"]["anyOf"] == [{"type": "string"}]
        assert properties["query"]["default"] is False
        assert tool_schema["additionalProperties"] is False
        assert tool_schema["properties"]["nested"]["additionalProperties"] is False

    def test_grep_schema_type_parameter_is_aliased_without_mutating_request(self, req):
        tool_schema = deepcopy(GREP_SCHEMA_FROM_SERVER_LOG)
        tool_schema["properties"]["_fcc_arg_type"] = {
            "type": "string",
            "description": "Existing safe property that collides with the alias",
        }
        tool_schema["required"] = ["pattern", "-A", "_fcc_arg_type"]
        original_schema = deepcopy(tool_schema)
        req.tools = [
            SimpleNamespace(
                name="Grep",
                description="Search file contents",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, NimSettings(), thinking_enabled=False)

        parameters = body["tools"][0]["function"]["parameters"]
        properties = parameters["properties"]
        aliases = body[NIM_TOOL_ARGUMENT_ALIASES_KEY]["Grep"]
        assert "additionalProperties" not in parameters
        assert properties["-A"] == original_schema["properties"]["-A"]
        assert properties["-B"] == original_schema["properties"]["-B"]
        assert properties["-C"] == original_schema["properties"]["-C"]
        assert properties["-i"] == original_schema["properties"]["-i"]
        assert properties["-n"] == original_schema["properties"]["-n"]
        assert "type" not in properties
        assert properties["pattern"] == original_schema["properties"]["pattern"]
        assert properties["output_mode"]["enum"] == [
            "content",
            "files_with_matches",
            "count",
        ]
        assert (
            properties["_fcc_arg_type"]
            == original_schema["properties"]["_fcc_arg_type"]
        )
        assert aliases == {"_fcc_arg_type_2": "type"}
        assert properties["_fcc_arg_type_2"] == original_schema["properties"]["type"]
        assert "-A" in parameters["required"]
        assert "_fcc_arg_type" in parameters["required"]
        assert tool_schema == original_schema

    def test_safe_tool_schema_does_not_add_alias_metadata(self, req):
        tool_schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "output_mode": {"type": "string", "enum": ["content", "count"]},
            },
            "required": ["pattern"],
        }
        req.tools = [
            SimpleNamespace(
                name="Glob",
                description="Find files",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, NimSettings(), thinking_enabled=False)

        assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in body
        parameters = body["tools"][0]["function"]["parameters"]
        assert parameters["properties"] == tool_schema["properties"]
        assert parameters["required"] == ["pattern"]

    def test_nested_schema_keyword_properties_are_aliased_without_mutating_request(
        self, req
    ):
        tool_schema = {
            "type": "object",
            "properties": {
                "parent": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["page_id"]},
                        "id": {"type": "string"},
                    },
                    "required": ["type", "id"],
                }
            },
            "required": ["parent"],
        }
        original_schema = deepcopy(tool_schema)
        req.tools = [
            SimpleNamespace(
                name="NotionLike",
                description="Nested type schema",
                input_schema=tool_schema,
            )
        ]

        body = build_request_body(req, NimSettings(), thinking_enabled=False)

        aliases = body[NIM_TOOL_ARGUMENT_ALIASES_KEY]["NotionLike"]
        parent = body["tools"][0]["function"]["parameters"]["properties"]["parent"]
        parent_properties = parent["properties"]
        assert "type" not in parent_properties
        assert parent_properties["_fcc_arg_type"] == {
            "type": "string",
            "enum": ["page_id"],
        }
        assert parent["required"] == ["_fcc_arg_type", "id"]
        assert aliases == {"_fcc_arg_type": "type"}
        assert tool_schema == original_schema

    def test_private_alias_metadata_is_stripped_without_mutating_body(self):
        body = {
            "model": "test",
            NIM_TOOL_ARGUMENT_ALIASES_KEY: {"Grep": {"_fcc_arg_A": "-A"}},
        }

        upstream_body = body_without_nim_tool_argument_aliases(body)

        assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in upstream_body
        assert body[NIM_TOOL_ARGUMENT_ALIASES_KEY] == {"Grep": {"_fcc_arg_A": "-A"}}
        assert nim_tool_argument_aliases_from_body(body) == {
            "Grep": {"_fcc_arg_A": "-A"}
        }

    def test_reasoning_params_in_extra_body(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [MagicMock(role="user", content="hi")]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        nim = NimSettings()
        body = build_request_body(req, nim, thinking_enabled=True)
        extra = body["extra_body"]
        assert extra["chat_template_kwargs"] == {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_budget": body["max_tokens"],
        }
        assert "reasoning_budget" not in extra

    def test_clone_body_without_chat_template(self):
        body = {
            "model": "test",
            "extra_body": {
                "chat_template": "custom_template",
                "chat_template_kwargs": {
                    "thinking": True,
                    "enable_thinking": True,
                    "reasoning_budget": 100,
                },
                "ignore_eos": False,
            },
        }

        cloned = clone_body_without_chat_template(body)

        assert cloned is not None
        assert "chat_template" not in cloned["extra_body"]
        assert cloned["extra_body"]["chat_template_kwargs"] == {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_budget": 100,
        }
        assert cloned["extra_body"]["ignore_eos"] is False
        assert body["extra_body"]["chat_template"] == "custom_template"

    def test_no_chat_template_kwargs_when_thinking_disabled(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [MagicMock(role="user", content="hi")]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        nim = NimSettings()
        body = build_request_body(req, nim, thinking_enabled=False)
        extra = body.get("extra_body", {})
        assert "chat_template_kwargs" not in extra
        assert "reasoning_budget" not in extra

    def test_reasoning_budget_respects_existing_chat_template_kwargs(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [MagicMock(role="user", content="hi")]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.top_k = None
        req.extra_body = {
            "chat_template_kwargs": {"enable_thinking": False, "custom": "value"}
        }

        body = build_request_body(req, NimSettings(), thinking_enabled=True)
        assert body["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": False,
            "custom": "value",
            "reasoning_budget": body["max_tokens"],
        }

    def test_chat_template_fields_present_for_mistral_model(self):
        req = MagicMock()
        req.model = "mistralai/mixtral-8x7b-instruct-v0.1"
        req.messages = [MagicMock(role="user", content="hi")]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        nim = NimSettings(chat_template="custom_template")
        body = build_request_body(req, nim, thinking_enabled=True)
        extra = body.get("extra_body", {})
        assert extra["chat_template_kwargs"] == {
            "thinking": True,
            "enable_thinking": True,
            "reasoning_budget": body["max_tokens"],
        }
        assert extra["chat_template"] == "custom_template"

    def test_no_reasoning_params_in_extra_body(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [MagicMock(role="user", content="hi")]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        nim = NimSettings()
        body = build_request_body(req, nim, thinking_enabled=False)
        extra = body.get("extra_body", {})
        for param in (
            "thinking",
            "reasoning_split",
            "return_tokens_as_token_ids",
            "include_reasoning",
            "reasoning_effort",
        ):
            assert param not in extra

    def test_assistant_thinking_blocks_removed_when_disabled(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [
            MagicMock(
                role="assistant",
                content=[
                    MagicMock(type="thinking", thinking="secret"),
                    MagicMock(type="text", text="answer"),
                ],
            )
        ]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        body = build_request_body(req, NimSettings(), thinking_enabled=False)
        assert "<think>" not in body["messages"][0]["content"]
        assert "answer" in body["messages"][0]["content"]

    def test_assistant_thinking_replayed_as_reasoning_content_when_enabled(self):
        req = MagicMock()
        req.model = "test"
        req.messages = [
            MagicMock(
                role="assistant",
                content=[
                    MagicMock(type="thinking", thinking="secret"),
                    MagicMock(type="text", text="answer"),
                ],
                reasoning_content=None,
            )
        ]
        req.max_tokens = 100
        req.system = None
        req.temperature = None
        req.top_p = None
        req.stop_sequences = None
        req.tools = None
        req.tool_choice = None
        req.extra_body = None
        req.top_k = None

        body = build_request_body(req, NimSettings(), thinking_enabled=True)
        assistant = body["messages"][0]
        assert assistant["reasoning_content"] == "secret"
        assert assistant["content"] == "answer"
        assert "<think>" not in assistant["content"]

    def test_clone_body_without_reasoning_content(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "secret",
                },
            ],
        }

        cloned = clone_body_without_reasoning_content(body)

        assert cloned is not None
        assert "reasoning_content" not in cloned["messages"][1]
        assert body["messages"][1]["reasoning_content"] == "secret"

    def test_clone_body_without_reasoning_content_returns_none_when_unchanged(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}

        assert clone_body_without_reasoning_content(body) is None
