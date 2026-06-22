from __future__ import annotations

import os
import subprocess

import pytest

from config.settings import Settings
from messaging.platforms.factory import create_messaging_components
from providers.registry import PROVIDER_DESCRIPTORS, build_provider_config
from smoke.lib.child_process import cmd_free_claude_code_serve, cmd_python_c
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("config")
def test_env_precedence_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    env_file = tmp_path / "product.env"
    env_file.write_text(
        'MODEL="open_router/test/model"\nANTHROPIC_AUTH_TOKEN="dotenv-token"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FCC_ENV_FILE"] = str(env_file)
    env["MODEL"] = "nvidia_nim/process-model"
    env["ANTHROPIC_AUTH_TOKEN"] = "process-token"
    script = (
        "from config.settings import get_settings; "
        "s=get_settings(); "
        "print(s.model); print(s.anthropic_auth_token)"
    )
    result = subprocess.run(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == ["nvidia_nim/process-model", "dotenv-token"]


@pytest.mark.smoke_target("config")
def test_removed_env_migration_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    env_file = tmp_path / "removed.env"
    env_file.write_text('NIM_ENABLE_THINKING="true"\n', encoding="utf-8")
    env = os.environ.copy()
    env["FCC_ENV_FILE"] = str(env_file)
    result = subprocess.run(
        cmd_python_c("from config.settings import Settings; Settings()"),
        cwd=smoke_config.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.smoke_target("config")
def test_per_model_thinking_config_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    env_file = tmp_path / "thinking.env"
    env_file.write_text(
        'ENABLE_MODEL_THINKING="false"\n'
        'ENABLE_OPUS_THINKING="true"\n'
        "ENABLE_SONNET_THINKING=\n"
        'ENABLE_HAIKU_THINKING="false"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FCC_ENV_FILE"] = str(env_file)
    script = (
        "from config.settings import Settings; "
        "s=Settings(); "
        "print(s.resolve_thinking('claude-opus-4-20250514')); "
        "print(s.resolve_thinking('claude-sonnet-4-20250514')); "
        "print(s.resolve_thinking('claude-haiku-4-20250514')); "
        "print(s.resolve_thinking('unknown-model'))"
    )
    result = subprocess.run(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "False", "False", "False"]


@pytest.mark.smoke_target("config")
def test_proxy_timeout_config_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    env_file = tmp_path / "timeouts.env"
    env_file.write_text(
        'MODEL="open_router/test/model"\n'
        'OPENROUTER_API_KEY="key"\n'
        'OPENROUTER_PROXY="socks5://127.0.0.1:9999"\n'
        'HTTP_READ_TIMEOUT="321"\n'
        'HTTP_CONNECT_TIMEOUT="7"\n'
        'HTTP_WRITE_TIMEOUT="8"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FCC_ENV_FILE"] = str(env_file)
    script = (
        "from config.settings import Settings; "
        "from providers.registry import PROVIDER_DESCRIPTORS, build_provider_config; "
        "s=Settings(); c=build_provider_config(PROVIDER_DESCRIPTORS['open_router'], s); "
        "print(c.proxy); print(c.http_read_timeout); "
        "print(c.http_connect_timeout); print(c.http_write_timeout)"
    )
    result = subprocess.run(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "socks5://127.0.0.1:9999",
        "321.0",
        "7.0",
        "8.0",
    ]


@pytest.mark.smoke_target("extensibility")
def test_provider_registry_e2e() -> None:
    settings_kwargs: dict[str, str] = {}
    for descriptor in PROVIDER_DESCRIPTORS.values():
        if descriptor.credential_attr is not None:
            settings_kwargs[_settings_init_key(descriptor.credential_attr)] = (
                f"{descriptor.provider_id}-key"
            )
        if descriptor.base_url_attr is not None and descriptor.default_base_url:
            settings_kwargs[_settings_init_key(descriptor.base_url_attr)] = (
                descriptor.default_base_url
            )
    settings = Settings.model_validate(settings_kwargs)
    for descriptor in PROVIDER_DESCRIPTORS.values():
        config = build_provider_config(descriptor, settings)
        assert config.base_url
        assert config.api_key


def _settings_init_key(field_name: str) -> str:
    alias = Settings.model_fields[field_name].validation_alias
    return alias if isinstance(alias, str) else field_name


@pytest.mark.smoke_target("extensibility")
def test_platform_factory_e2e() -> None:
    assert create_messaging_components("not-a-platform") is None
    assert create_messaging_components("telegram") is None
    assert create_messaging_components("discord") is None


@pytest.mark.smoke_target("cli")
def test_entrypoint_server_e2e(smoke_config: SmokeConfig) -> None:
    with SmokeServerDriver(
        smoke_config,
        name="product-entrypoint",
        command=cmd_free_claude_code_serve(),
        env_overrides={"MESSAGING_PLATFORM": "none"},
    ).run() as server:
        assert server.process.poll() is None
