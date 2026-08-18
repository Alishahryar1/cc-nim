"""Contracts for pure Settings and canonical source composition."""

from enum import Enum

import pytest
from pydantic import ValidationError

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    HTTP_CONNECT_TIMEOUT_DEFAULT,
)
from free_claude_code.config.loader import (
    ConfigSource,
    clear_settings_cache,
    compose_settings_snapshot,
    get_settings,
)
from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    parse_model_name,
    parse_provider_type,
)
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.paths import messaging_state_dir_path, server_log_path
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings


def test_settings_defaults_are_valid_and_nonempty() -> None:
    settings = Settings()

    assert settings.provider_rate_limit == 1
    assert settings.provider_rate_window == 2
    assert settings.provider_max_concurrency == 2
    assert settings.provider_progress_timeout == 600.0
    assert settings.http_read_timeout == 120.0
    assert settings.http_write_timeout == 10.0
    assert settings.http_connect_timeout == HTTP_CONNECT_TIMEOUT_DEFAULT
    assert settings.voice_note_enabled is True
    assert settings.whisper_device == "cpu"
    assert settings.whisper_model == "base"
    assert settings.enable_web_server_tools is False
    assert settings.proxy_auth_enabled is False
    assert settings.proxy_auth_token == "freecc"
    assert [
        name for name, value in settings if isinstance(value, str) and not value
    ] == []
    assert [
        name
        for name, field in Settings.model_fields.items()
        if field.get_default(call_default_factory=True) == ""
    ] == []


def test_every_external_setting_has_one_explicit_alias() -> None:
    aliases = [
        field.validation_alias
        for name, field in Settings.model_fields.items()
        if name != "nim"
    ]
    assert all(isinstance(alias, str) for alias in aliases)
    assert len(aliases) == len(set(aliases))
    assert Settings.model_fields["nim"].validation_alias is None


def test_direct_settings_construction_performs_no_environment_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL", "deepseek/process-model")
    monkeypatch.setenv("PROVIDER_RATE_LIMIT", "99")

    settings = Settings()

    assert settings.model.startswith("nvidia_nim/")
    assert settings.provider_rate_limit == 1


@pytest.mark.parametrize(
    ("key", "attribute", "value", "expected"),
    [
        ("MODEL", "model", "deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("PROVIDER_RATE_LIMIT", "provider_rate_limit", "20", 20),
        (
            "PROVIDER_PROGRESS_TIMEOUT",
            "provider_progress_timeout",
            "900",
            900.0,
        ),
        ("HTTP_READ_TIMEOUT", "http_read_timeout", "600", 600.0),
        ("FCC_OPEN_BROWSER", "open_admin_browser", "false", False),
        ("REASONING_POLICY", "reasoning_policy", "off", ReasoningPreference.OFF),
        ("GROQ_API_KEY", "groq_api_key", " secret ", "secret"),
        ("OPENROUTER_PROXY", "open_router_proxy", " http://proxy ", "http://proxy"),
    ],
)
def test_process_values_are_parsed_at_the_loader_boundary(
    key: str,
    attribute: str,
    value: str,
    expected: object,
) -> None:
    snapshot = compose_settings_snapshot({}, {key: value})

    assert getattr(snapshot.settings, attribute) == expected
    assert snapshot.sources[attribute] is ConfigSource.PROCESS


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_provider_progress_timeout_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(provider_progress_timeout=value)


@pytest.mark.parametrize("value", ["0", "-1", "inf", "-inf", "nan"])
def test_loader_rejects_invalid_provider_progress_timeout(value: str) -> None:
    with pytest.raises(ValidationError):
        compose_settings_snapshot({}, {"PROVIDER_PROGRESS_TIMEOUT": value})


@pytest.mark.parametrize(
    "key",
    [
        "GROQ_API_KEY",
        "OPENROUTER_PROXY",
        "MODEL_OPUS",
        "TELEGRAM_BOT_TOKEN",
        "ALLOWED_DIR",
    ],
)
def test_optional_blank_process_values_normalize_to_none(key: str) -> None:
    snapshot = compose_settings_snapshot({}, {key: "  "})
    attribute = next(
        name
        for name, field in Settings.model_fields.items()
        if field.validation_alias == key
    )

    assert getattr(snapshot.settings, attribute) is None


def test_blank_required_process_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="MODEL"):
        compose_settings_snapshot({}, {"MODEL": " "})


def test_blank_process_auth_token_uses_retained_default() -> None:
    snapshot = compose_settings_snapshot({}, {"ANTHROPIC_AUTH_TOKEN": ""})

    assert snapshot.settings.proxy_auth_token == "freecc"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.DEFAULT


def test_process_precedence_and_managed_token_exception() -> None:
    snapshot = compose_settings_snapshot(
        {
            "MODEL": "deepseek/managed",
            "ANTHROPIC_AUTH_TOKEN": "managed-token",
        },
        {
            "MODEL": "groq/process",
            "ANTHROPIC_AUTH_TOKEN": "stale-process-token",
        },
    )

    assert snapshot.settings.model == "groq/process"
    assert snapshot.sources["model"] is ConfigSource.PROCESS
    assert snapshot.settings.proxy_auth_token == "managed-token"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.MANAGED


def test_get_settings_is_cached_and_creates_managed_schema() -> None:
    clear_settings_cache()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_optional_strings_share_one_normalization_rule() -> None:
    settings = Settings.model_validate(
        {
            "GROQ_API_KEY": "  key  ",
            "OPENROUTER_PROXY": " ",
            "MODEL_OPUS": "",
            "ALLOWED_DIR": None,
        }
    )

    assert settings.groq_api_key == "key"
    assert settings.open_router_proxy is None
    assert settings.model_opus is None
    assert settings.allowed_dir is None


@pytest.mark.parametrize(
    "field",
    ["MODEL", "HOST", "WHISPER_MODEL", "LOG_LEVEL", "ANTHROPIC_AUTH_TOKEN"],
)
def test_required_strings_reject_blank(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "   "})


def test_model_validation_and_routing() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_opus="open_router/anthropic/claude-opus",
    )

    router = ModelRouter(settings)
    assert router.resolve("claude-opus-4").provider_model_ref == (
        "open_router/anthropic/claude-opus"
    )
    assert router.resolve("unknown").provider_model_ref == "deepseek/fallback"
    with pytest.raises(ValidationError, match="Invalid provider"):
        Settings(model="unknown/model")


def test_configured_chat_model_refs_are_unique() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_fable="open_router/anthropic/claude-fable",
        model_sonnet="deepseek/fallback",
    )

    refs = configured_chat_model_refs(settings)

    assert [ref.model_ref for ref in refs] == [
        "deepseek/fallback",
        "open_router/anthropic/claude-fable",
    ]


@pytest.mark.parametrize(
    ("model_ref", "provider", "model"),
    [
        ("nvidia_nim/meta/llama", "nvidia_nim", "meta/llama"),
        ("open_router/deepseek/r1", "open_router", "deepseek/r1"),
        ("ollama_cloud/qwen3-coder:480b", "ollama_cloud", "qwen3-coder:480b"),
    ],
)
def test_model_ref_parsing(model_ref: str, provider: str, model: str) -> None:
    assert parse_provider_type(model_ref) == provider
    assert parse_model_name(model_ref) == model


def test_paths_are_owned_by_fcc_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert messaging_state_dir_path() == tmp_path / ".fcc" / "agent_workspace"
    assert server_log_path() == tmp_path / ".fcc" / "logs" / "server.log"


def test_nim_settings_keep_request_local_validation() -> None:
    settings = NimSettings.model_validate(
        {
            "max_tokens": "1024",
            "temperature": "0.5",
            "seed": "7",
            "stop": "",
        }
    )

    assert settings.max_tokens == 1024
    assert settings.temperature == 0.5
    assert settings.seed == 7
    assert settings.stop is None
    assert NimSettings().max_tokens == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    with pytest.raises(ValidationError):
        NimSettings(top_p=1.1)


def test_settings_defaults_do_not_contain_empty_enum_strings() -> None:
    for _name, value in Settings():
        if isinstance(value, Enum):
            assert value.value
