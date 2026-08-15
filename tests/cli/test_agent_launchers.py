import os
import tempfile
from pathlib import Path

from free_claude_code.cli.launchers.ensure import (
    ClientSpec,
    auto_install_enabled,
    windows_path_text_to_wsl,
    wsl_exec,
)
from free_claude_code.cli.launchers.muse import (
    build_muse_launcher_command,
    build_muse_launcher_env,
    build_muse_mcp_servers,
    build_muse_settings_document,
)
from free_claude_code.cli.launchers.openai_compat import (
    build_openai_compat_env,
    openai_compat_base_url,
    proxy_bearer_token,
)
from free_claude_code.cli.launchers.prime import (
    build_prime_launcher_command,
    build_prime_launcher_env,
    build_prime_models_document,
    write_prime_agent_dir,
)


def test_openai_compat_env_points_at_local_proxy_without_parent_keys() -> None:
    env = build_openai_compat_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        base_env={"OPENAI_API_KEY": "sk-parent", "PATH": "/bin"},
        model="open_router/openai/gpt-4o-mini",
    )
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8082/v1"
    assert env["OPENAI_API_KEY"] == "proxy-token"
    assert env["FCC_MODEL"] == "open_router/openai/gpt-4o-mini"
    assert "sk-parent" not in env.values()


def test_proxy_bearer_token_uses_sentinel_when_empty() -> None:
    assert proxy_bearer_token("  ") == "fcc-no-auth"
    assert openai_compat_base_url("http://127.0.0.1:8082/") == "http://127.0.0.1:8082/v1"


def test_prime_models_json_uses_env_name_not_secret() -> None:
    document = build_prime_models_document(
        proxy_root_url="http://127.0.0.1:8082",
        model="open_router/openai/gpt-4o-mini",
    )
    provider = document["providers"]["fcc"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == "http://127.0.0.1:8082/v1"
    assert provider["apiKey"] == "FCC_PRIME_API_KEY"
    dumped = str(document)
    assert "sk-" not in dumped
    assert "password" not in dumped.lower()


def test_prime_command_defaults_provider_and_preserves_user_args() -> None:
    command = build_prime_launcher_command(
        binary_path="prime-agent",
        argv=["-p", "hello"],
        model="open_router/openai/gpt-4o-mini",
    )
    assert command[:5] == [
        "prime-agent",
        "--provider",
        "fcc",
        "--model",
        "fcc/open_router/openai/gpt-4o-mini",
    ]
    assert command[-2:] == ["-p", "hello"]


def test_prime_temp_dir_has_no_secrets() -> None:
    with tempfile.TemporaryDirectory(prefix="fcc-prime-test-") as raw:
        tmp_path = Path(raw)
        models_path = write_prime_agent_dir(
            tmp_path,
            proxy_root_url="http://127.0.0.1:8082",
            model="open_router/openai/gpt-4o-mini",
        )
        text = models_path.read_text(encoding="utf-8")
        assert "FCC_PRIME_API_KEY" in text
        assert "sk-" not in text
        env = build_prime_launcher_env(
            proxy_root_url="http://127.0.0.1:8082",
            auth_token="secret-token",
            model="open_router/openai/gpt-4o-mini",
            agent_dir=tmp_path,
            base_env={},
        )
        assert env["FCC_PRIME_API_KEY"] == "secret-token"
        assert env["PRIME_AGENT_CODING_AGENT_DIR"] == str(tmp_path)


def test_muse_command_and_optional_mcp() -> None:
    command = build_muse_launcher_command(
        binary_path="muse",
        argv=["exec", "ping"],
        proxy_root_url="http://127.0.0.1:8082",
        model="open_router/openai/gpt-4o-mini",
        settings_path=Path("/tmp/fcc-muse/settings.json"),
    )
    assert "--provider" in command and "meta" in command
    assert "--base-url" in command and "http://127.0.0.1:8082" in command
    assert command[-2:] == ["exec", "ping"]
    servers = build_muse_mcp_servers(
        {"FCC_MUSE_MCP_COMMAND": "npx", "FCC_MUSE_MCP_SERVERS": ""}
    )
    assert servers["fcc-tools"]["mode"] == "optional"
    document = build_muse_settings_document(mcp_servers=servers)
    assert "mcp_servers" in document
    env = build_muse_launcher_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="proxy-token",
        model="open_router/openai/gpt-4o-mini",
        base_env={},
    )
    assert env["META_API_KEY"] == "proxy-token"
    assert env["META_BASE_URL"] == "http://127.0.0.1:8082"


def test_auto_install_env_defaults_on() -> None:
    assert auto_install_enabled({}) is True
    assert auto_install_enabled({"FCC_NO_AUTO_INSTALL": "1"}) is False
    assert auto_install_enabled({"FCC_NO_AUTO_INSTALL": "true"}) is False


def test_wsl_muse_command_quotes_settings_path() -> None:
    client = ClientSpec(kind="wsl", binary="muse")
    settings_path = Path("/tmp/fcc-muse/settings.json")
    command = build_muse_launcher_command(
        client=client,
        argv=["exec", "ping"],
        proxy_root_url="http://127.0.0.1:8082",
        model="open_router/openai/gpt-4o-mini",
        settings_path=settings_path,
    )
    assert command[:3] == ["wsl", "bash", "-lc"]
    inner = command[3]
    assert "exec muse" in inner
    assert "--provider" in inner
    assert "meta" in inner
    assert "--settings" in inner
    assert windows_path_text_to_wsl(r"C:\Users\me\settings.json") == (
        "/mnt/c/Users/me/settings.json"
    )


def test_wsl_exec_quotes_spaces() -> None:
    command = wsl_exec("muse", ["--settings", "/tmp/my settings.json"])
    assert command[:3] == ["wsl", "bash", "-lc"]
    assert "'/tmp/my settings.json'" in command[3]


def test_prime_skips_install_when_disabled(monkeypatch, capsys) -> None:
    from free_claude_code.cli.launchers import ensure as ensure_mod

    monkeypatch.setattr(ensure_mod, "find_prime_binary", lambda: None)
    monkeypatch.setenv("FCC_NO_AUTO_INSTALL", "1")
    try:
        ensure_mod.ensure_prime_binary()
    except SystemExit as exc:
        assert exc.code == 127
    else:
        raise AssertionError("expected SystemExit 127")
    err = capsys.readouterr().err
    assert "Could not find Prime Agent" in err
    assert "fcc-prime can install it" in err or "github.com/PrimeIntellect-ai" in err
    assert os.environ["FCC_NO_AUTO_INSTALL"] == "1"
