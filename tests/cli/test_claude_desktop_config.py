"""Tests for the Claude Desktop config merge and binary launch helpers."""

import json
from pathlib import Path

import pytest

from free_claude_code.cli import claude_desktop, tls_proxy
from free_claude_code.config.settings import Settings


@pytest.fixture
def fake_config(tmp_path: Path) -> Path:
    return tmp_path / "claude_desktop_config.json"


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(host="127.0.0.1", port=8082, proxy_auth_token="tok-123")


@pytest.fixture(autouse=True)
def deterministic_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin gateway URL resolution so tests never probe the network.

    The TLS-front probe is disabled at its source (a developer machine may
    have a live front on 8443); ``desktop_gateway_base_url`` still runs for
    real, so these tests also pin the desktop path prefix wiring.
    """

    monkeypatch.setattr(tls_proxy, "probe_fcc_front", lambda *a, **kw: False)
    # ``resolve_gateway_base_url`` loads the identity before probing; pin it
    # so the real ``~/.fcc/caddy/front-identity`` is never created.
    monkeypatch.setattr(
        tls_proxy, "load_or_create_front_identity", lambda *a, **kw: "pinned"
    )


def test_managed_block_derives_url_and_key_from_settings(
    fake_settings: Settings,
) -> None:
    block = claude_desktop.fcc_managed_block(
        fake_settings, gateway_base_url="https://localhost:8443"
    )

    assert block["inferenceGatewayBaseUrl"] == "https://localhost:8443"
    assert block["inferenceAnthropicApiKey"] == "tok-123"
    assert block["provider"] == "gateway"


def test_managed_block_resolves_url_when_not_overridden(
    fake_settings: Settings,
) -> None:
    block = claude_desktop.fcc_managed_block(fake_settings)

    assert block["inferenceGatewayBaseUrl"] == ("http://127.0.0.1:8082/claude-desktop")


def test_configure_applies_block_when_file_is_missing(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert not fake_config.exists()

    changed = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    assert (
        data["inference"]["inferenceGatewayBaseUrl"]
        == "http://127.0.0.1:8082/claude-desktop"
    )
    assert data["inference"]["inferenceAnthropicApiKey"] == "tok-123"


def test_configure_is_idempotent(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    second = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )
    assert second is False


def test_configure_preserves_unrelated_keys(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "mcpServers": {"fs": {"command": "node"}},
                "coworkUserFilesPath": "/home/user/cowork",
                "env": {"ANTHROPIC_BASE_URL": "https://localhost:8443"},
            }
        ),
        encoding="utf-8",
    )

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["preferences"] == {"theme": "dark"}
    assert data["mcpServers"] == {"fs": {"command": "node"}}
    assert data["coworkUserFilesPath"] == "/home/user/cowork"
    assert data["env"] == {"ANTHROPIC_BASE_URL": "https://localhost:8443"}
    assert data["modelDiscoveryEnabled"] is True


def test_configure_overwrites_partial_inference_block(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.write_text(
        json.dumps({"inference": {"provider": "anthropic", "keep_me": True}}),
        encoding="utf-8",
    )

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    inference = data["inference"]
    assert inference["provider"] == "gateway"
    assert inference["keep_me"] is True
    assert (
        inference["inferenceGatewayBaseUrl"] == "http://127.0.0.1:8082/claude-desktop"
    )


def test_configure_aborts_on_malformed_json(
    fake_config: Path,
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text("{not json", encoding="utf-8")

    changed = claude_desktop.configure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is False
    assert fake_config.read_text(encoding="utf-8") == "{not json"
    captured = capsys.readouterr()
    assert captured.out == ""


def test_configure_aborts_on_non_object_root(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text("[1, 2]", encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )


def test_load_existing_config_raises_malformed_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(claude_desktop.MalformedConfigError):
        claude_desktop.load_existing_config(path)


def test_unconfigure_drops_only_fcc_keys(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    managed = claude_desktop.fcc_managed_block(fake_settings)
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "modelDiscoveryEnabled": True,
                "inference": managed | {"extra_user_key": "stay"},
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert data["inference"] == {"extra_user_key": "stay"}
    assert data["preferences"] == {"theme": "dark"}


def test_unconfigure_removes_block_when_gateway_url_no_longer_matches(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "unconfigure leaves the gateway
    # behind" finding: a previous configure persisted the HTTPS gateway
    # URL, but the front is now gone and URL resolution falls back to
    # plain HTTP. Removal must not depend on the recorded URL matching
    # current resolution, or the gateway URL and auth token stay behind.
    # The settings-derived token is stable across front state, so it is
    # the ownership marker that still identifies the block as FCC's.
    fake_config.write_text(
        json.dumps(
            {
                "preferences": {"theme": "dark"},
                "modelDiscoveryEnabled": True,
                "inference": {
                    "provider": "gateway",
                    "credentialKind": "static",
                    "inferenceProvider": "gateway",
                    "inferenceCredentialKind": "static",
                    "inferenceGatewayBaseUrl": "https://localhost:8443/claude-desktop",
                    "inferenceGatewayAuthScheme": "x-api-key",
                    "inferenceAnthropicApiKey": "tok-123",
                    "extra_user_key": "stay",
                },
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert data["inference"] == {"extra_user_key": "stay"}
    assert data["preferences"] == {"theme": "dark"}


def test_unconfigure_preserves_user_values_sharing_fcc_key_names(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "unconfigure erases user settings"
    # finding: keys that share FCC-managed names but hold values FCC never
    # wrote (neither the current resolution nor the FCC auth token) belong
    # to the user and must survive removal. Without a snapshot the
    # discovery flag is user-owned too — ownership is ambiguous, so it
    # must be preserved rather than deleted.
    fake_config.write_text(
        json.dumps(
            {
                "modelDiscoveryEnabled": True,
                "inference": {
                    "provider": "anthropic",
                    "inferenceGatewayBaseUrl": "https://api.user-gateway.example",
                    "inferenceAnthropicApiKey": "user-secret-key",
                },
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    # Nothing in the config is FCC-owned, so removal is a no-op and the
    # file is left exactly as the user wrote it.
    assert changed is False
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    assert data["inference"] == {
        "provider": "anthropic",
        "inferenceGatewayBaseUrl": "https://api.user-gateway.example",
        "inferenceAnthropicApiKey": "user-secret-key",
    }


def test_unconfigure_removes_discovery_when_legacy_token_marks_ownership(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "unconfigure deletes legacy
    # discovery preference" finding: without a snapshot the discovery
    # flag may only be removed once FCC ownership is established by the
    # recorded auth token — never before.
    fake_config.write_text(
        json.dumps(
            {
                "modelDiscoveryEnabled": True,
                "inference": {
                    "provider": "gateway",
                    "inferenceGatewayBaseUrl": "https://localhost:8443/claude-desktop",
                    "inferenceAnthropicApiKey": "tok-123",
                },
            }
        ),
        encoding="utf-8",
    )

    changed = claude_desktop.unconfigure_claude_desktop_config(
        fake_config, settings=fake_settings
    )

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert "inference" not in data


def test_configure_unconfigure_roundtrip_restores_prior_settings(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "unconfigure drops prior settings"
    # finding: configure overwrites the user's own provider, gateway, and
    # credential values, so unconfigure must restore the originals instead
    # of deleting them.
    original = {
        "modelDiscoveryEnabled": True,
        "preferences": {"theme": "dark"},
        "inference": {
            "provider": "anthropic",
            "credentialKind": "oauth",
            "inferenceProvider": "anthropic",
            "inferenceCredentialKind": "oauth",
            "inferenceGatewayBaseUrl": "https://api.user-gateway.example",
            "inferenceGatewayAuthScheme": "bearer",
            "inferenceAnthropicApiKey": "user-secret-key",
            "extra_user_key": "stay",
        },
    }
    fake_config.write_text(json.dumps(original), encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    configured = json.loads(fake_config.read_text(encoding="utf-8"))
    assert configured["inference"]["provider"] == "gateway"
    assert configured["inference"]["inferenceAnthropicApiKey"] == "tok-123"

    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    restored = json.loads(fake_config.read_text(encoding="utf-8"))
    assert restored == original


def test_configure_keeps_first_snapshot_across_remerges(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # The snapshot must record the user's ORIGINAL values: a re-merge that
    # refreshed it with FCC's own block would make unconfigure "restore"
    # the FCC values instead of the user's.
    fake_config.write_text(
        json.dumps(
            {
                "inference": {
                    "provider": "anthropic",
                    "inferenceAnthropicApiKey": "user-secret-key",
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    # No discovery entry: the key was absent before the first merge, and
    # the restore must delete it again rather than invent a value.
    assert data["fccPriorConfig"] == {
        "inference": {
            "provider": "anthropic",
            "inferenceAnthropicApiKey": "user-secret-key",
        },
    }


def test_configure_unconfigure_roundtrip_restores_disabled_discovery(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "discovery value restores
    # incorrectly" finding: a user who explicitly disabled model discovery
    # must get ``false`` back after unconfigure — presence alone would
    # restore ``true`` and permanently flip the preference.
    original = {
        "modelDiscoveryEnabled": False,
        "inference": {"provider": "anthropic"},
    }
    fake_config.write_text(json.dumps(original), encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    configured = json.loads(fake_config.read_text(encoding="utf-8"))
    assert configured["modelDiscoveryEnabled"] is True
    assert configured["fccPriorConfig"]["discovery"] is False

    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    restored = json.loads(fake_config.read_text(encoding="utf-8"))
    assert restored == original


@pytest.mark.parametrize(
    "raw_inference",
    ["legacy-string", ["a", "b"], None],
    ids=["string", "list", "null"],
)
def test_configure_unconfigure_roundtrip_restores_non_object_inference(
    fake_config: Path,
    fake_settings: Settings,
    raw_inference: object,
) -> None:
    # Regression guard for the Greptile "non-object inference values are
    # not restored" finding: a valid config whose inference value is a
    # string, list, or null must get that exact value back after a
    # configure/unconfigure round trip, not a deleted key.
    original = {"inference": raw_inference, "preferences": {"theme": "dark"}}
    fake_config.write_text(json.dumps(original), encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    configured = json.loads(fake_config.read_text(encoding="utf-8"))
    assert configured["inference"]["provider"] == "gateway"
    assert configured["fccPriorConfig"]["inferenceRaw"] == raw_inference

    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    restored = json.loads(fake_config.read_text(encoding="utf-8"))
    assert restored == original


def test_unconfigure_roundtrip_is_clean_noop(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data == {}

    # Second passes are no-ops.
    assert (
        claude_desktop.unconfigure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )


def test_unconfigure_silent_on_missing_file(tmp_path: Path) -> None:
    assert (
        claude_desktop.unconfigure_claude_desktop_config(tmp_path / "absent.json")
        is False
    )


def test_config_path_resolver_returns_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    for platform in ("linux", "darwin", "win32"):
        monkeypatch.setattr(claude_desktop.sys, "platform", platform)
        path = claude_desktop._config_path()
        assert path.name == claude_desktop.CONFIG_FILENAME
        assert path.is_absolute()


def test_main_configure_writes_block_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    fake = tmp_path / "claude_desktop_config.json"
    verified_url = "https://localhost:8443/claude-desktop"

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configure must not spawn a subprocess")

    monkeypatch.setattr(claude_desktop.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )
    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: verified_url
    )

    claude_desktop.main(["--configure", "--config-path", str(fake)])

    data = json.loads(fake.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    # The persisted block must point at the verified HTTPS front.
    assert data["inference"]["inferenceGatewayBaseUrl"] == verified_url


def test_main_configure_adopts_existing_front_without_spawning_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    # The command exits right after writing, so a front it spawned itself
    # would die with it and leave the config pointing at a dead gateway.
    # Adoption must be probe-only: verification runs for real here (probe
    # pinned), and any attempt to bring up a managed proxy trips the
    # construction guard.
    fake = tmp_path / "claude_desktop_config.json"

    def fail_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configure must never bring up its own front")

    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )
    monkeypatch.setattr(tls_proxy, "CaddyTlsProxy", fail_spawn)
    monkeypatch.setattr(tls_proxy, "probe_fcc_front", lambda *a, **kw: True)

    claude_desktop.main(["--configure", "--config-path", str(fake)])

    data = json.loads(fake.read_text(encoding="utf-8"))
    assert data["inference"]["inferenceGatewayBaseUrl"].startswith("https://")


def test_main_configure_refuses_malformed_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Regression guard for the Greptile "malformed configure reports
    # success" finding: configure returns False for BOTH an idempotent
    # no-op and a malformed-config skip, so the skip must be refused
    # with a nonzero exit instead of reported as "Already merged".
    fake = tmp_path / "claude_desktop_config.json"
    fake.write_text("{not json", encoding="utf-8")

    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )
    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: "https://x"
    )

    with pytest.raises(SystemExit) as exc:
        claude_desktop.main(["--configure", "--config-path", str(fake)])

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "malformed config" in captured.err
    assert "Already merged" not in captured.out
    # The malformed file is left untouched.
    assert fake.read_text(encoding="utf-8") == "{not json"


def test_main_configure_refuses_without_https_front(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The routing block embeds the reusable proxy token, so persisting it
    # against the plain-HTTP fallback would hand the credential to whatever
    # occupies that endpoint. Refuse without writing instead.
    fake = tmp_path / "claude_desktop_config.json"

    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )
    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: None
    )

    with pytest.raises(SystemExit) as exc:
        claude_desktop.main(["--configure", "--config-path", str(fake)])

    assert exc.value.code == 1
    assert not fake.exists()
    captured = capsys.readouterr()
    assert "no verified FCC HTTPS front" in captured.err


def test_launch_binary_spawns_without_certificate_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            calls.append(cmd)

    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.launch_binary(["--extra"])

    assert calls == [["/usr/bin/claude", "--extra"]]


def test_launch_binary_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: None)

    with pytest.raises(FileNotFoundError):
        claude_desktop.launch_binary()


def test_ensure_configured_and_launch_merges_then_spawns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    config_path = tmp_path / "claude_desktop_config.json"
    order: list[str] = []
    gateway_urls: list[str | None] = []
    original_configure = claude_desktop.configure_claude_desktop_config
    verified_url = "https://localhost:8443/claude-desktop"

    def fake_configure(
        path: Path | None = None,
        settings: Settings | None = None,
        gateway_base_url: str | None = None,
    ) -> bool:
        order.append("configure")
        gateway_urls.append(gateway_base_url)
        return original_configure(
            path or config_path, settings=settings, gateway_base_url=gateway_base_url
        )

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            order.append(f"spawn:{cmd[0]}")

    monkeypatch.setattr(
        claude_desktop, "configure_claude_desktop_config", fake_configure
    )
    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: verified_url
    )
    # The post-merge read-back must observe the same file the merge wrote.
    monkeypatch.setattr(claude_desktop, "_config_path", lambda: config_path)
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.ensure_configured_and_launch(settings=fake_settings)

    assert order[0] == "configure"
    assert order[1].startswith("spawn:")
    # The routing block must point at the verified HTTPS front, never the
    # plain-HTTP fallback.
    assert gateway_urls == [verified_url]


def test_ensure_configured_and_launch_refuses_when_merge_skips_malformed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile "launch after malformed config"
    # finding: ``configure`` returns ``False`` when it skips a malformed
    # config, so the launch must be refused rather than spawning Claude
    # Desktop without FCC routing. The read-back distinguishes the skip
    # from an idempotent no-op merge.
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text("{not json", encoding="utf-8")

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a skipped merge must never reach the binary")

    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: "https://x"
    )
    monkeypatch.setattr(claude_desktop, "_config_path", lambda: config_path)
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", fail_popen)

    with pytest.raises(RuntimeError, match="routing block could not be written"):
        claude_desktop.ensure_configured_and_launch(settings=fake_settings)

    # The malformed file is left untouched.
    assert config_path.read_text(encoding="utf-8") == "{not json"


def test_ensure_configured_and_launch_refuses_without_https_front(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_settings: Settings,
) -> None:
    # Regression guard for the Greptile tray-bypass finding: without a
    # verified HTTPS front the gateway URL would fall back to plain HTTP,
    # so the credential-bearing config write and the spawn must both be
    # refused instead of exposing the token over cleartext local transport.
    config_path = tmp_path / "claude_desktop_config.json"

    def fail_configure(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("no HTTPS front must never write the config")

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no HTTPS front must never reach the binary")

    monkeypatch.setattr(
        claude_desktop, "configure_claude_desktop_config", fail_configure
    )
    monkeypatch.setattr(
        claude_desktop, "verified_https_gateway_url", lambda settings: None
    )
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", fail_popen)

    with pytest.raises(RuntimeError, match="no verified FCC HTTPS front"):
        claude_desktop.ensure_configured_and_launch(settings=fake_settings)

    assert not config_path.exists()


def test_unconfigure_restores_wholesale_inference_replacement(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A post-merge wholesale ``inference`` replacement survives unconfigure.

    Regression guard for the Greptile "reconfiguration overwrites full
    inference replacements" finding: after the first merge the user swaps
    the whole ``inference`` entry for a scalar or ``null``. A later
    re-merge still writes the managed mapping (configure's contract is
    to ensure the block exists), but the replacement is now the origin
    the snapshot must restore — keeping the first-merge value would
    silently discard the user's later change.
    """

    for replacement in ("user-scalar", None):
        fake_config.write_text(json.dumps({}), encoding="utf-8")
        assert (
            claude_desktop.configure_claude_desktop_config(
                fake_config, settings=fake_settings
            )
            is True
        )
        fake_config.write_text(json.dumps({"inference": replacement}), encoding="utf-8")
        assert (
            claude_desktop.configure_claude_desktop_config(
                fake_config, settings=fake_settings
            )
            is True
        )
        assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True
        assert json.loads(fake_config.read_text(encoding="utf-8")) == {
            "inference": replacement
        }
