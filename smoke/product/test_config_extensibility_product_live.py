import os

import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.messaging.platforms.factory import create_messaging_components
from free_claude_code.providers.runtime import build_provider_config
from smoke.lib.child_process import (
    cmd_free_claude_code_serve,
    cmd_python_c,
    run_captured_text,
)
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
        "from free_claude_code.config.settings import get_settings; "
        "s=get_settings(); "
        "print(s.model); print(s.anthropic_auth_token)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
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
    result = run_captured_text(
        cmd_python_c(
            "from free_claude_code.config.settings import Settings; Settings()"
        ),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.smoke_target("config")
def test_per_model_thinking_config_e2e(smoke_config: SmokeConfig, tmp_path) -> None:
    env_file = tmp_path / "thinking.env"
    env_file.write_text(
        'ENABLE_MODEL_THINKING="false"\n'
        'ENABLE_FABLE_THINKING="true"\n'
        'ENABLE_OPUS_THINKING="true"\n'
        "ENABLE_SONNET_THINKING=\n"
        'ENABLE_HAIKU_THINKING="false"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FCC_ENV_FILE"] = str(env_file)
    script = (
        "from free_claude_code.application.routing import ModelRouter; "
        "from free_claude_code.config.settings import Settings; "
        "s=Settings(); "
        "r=ModelRouter(s); "
        "print(r.resolve('claude-fable-5').thinking_enabled); "
        "print(r.resolve('claude-opus-4-20250514').thinking_enabled); "
        "print(r.resolve('claude-sonnet-4-20250514').thinking_enabled); "
        "print(r.resolve('claude-haiku-4-20250514').thinking_enabled); "
        "print(r.resolve('unknown-model').thinking_enabled)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "True",
        "True",
        "False",
        "False",
        "False",
    ]


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
        "from free_claude_code.config.settings import Settings; "
        "from free_claude_code.config.provider_catalog import PROVIDER_CATALOG; "
        "from free_claude_code.providers.runtime import build_provider_config; "
        "s=Settings(); c=build_provider_config(PROVIDER_CATALOG['open_router'], s); "
        "print(c.proxy); print(c.http_read_timeout); "
        "print(c.http_connect_timeout); print(c.http_write_timeout)"
    )
    result = run_captured_text(
        cmd_python_c(script),
        cwd=smoke_config.root,
        env=env,
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
def test_provider_runtime_config_e2e() -> None:
    settings_kwargs: dict[str, str] = {}
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_attr is not None:
            settings_kwargs[_settings_init_key(descriptor.credential_attr)] = (
                f"{descriptor.provider_id}-key"
            )
        if descriptor.base_url_attr is not None and descriptor.default_base_url:
            settings_kwargs[_settings_init_key(descriptor.base_url_attr)] = (
                descriptor.default_base_url
            )
    settings = Settings.model_validate(settings_kwargs)
    for descriptor in PROVIDER_CATALOG.values():
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
