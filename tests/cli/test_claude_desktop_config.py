"""Tests for the Claude Desktop config merge and binary launch helpers."""

import json
import os
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


def test_unconfigure_restores_pre_merge_values(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    original = {
        "preferences": {"theme": "dark"},
        "modelDiscoveryEnabled": False,
        "inference": {"provider": "anthropic", "extra_user_key": "stay"},
    }
    fake_config.write_text(json.dumps(original), encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    assert json.loads(fake_config.read_text(encoding="utf-8")) == original


def test_unconfigure_without_record_leaves_config_untouched(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Settings FCC never wrote must survive ``--unconfigure``."""

    managed = claude_desktop.fcc_managed_block(fake_settings)
    preexisting = {
        "modelDiscoveryEnabled": True,
        "inference": managed | {"extra_user_key": "stay"},
    }
    fake_config.write_text(json.dumps(preexisting), encoding="utf-8")

    # Configure finds nothing to change, so FCC owns nothing.
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is False
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is False

    assert json.loads(fake_config.read_text(encoding="utf-8")) == preexisting


def test_unconfigure_drops_only_fcc_keys(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.write_text(
        json.dumps({"preferences": {"theme": "dark"}, "inference": {}}),
        encoding="utf-8",
    )

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    changed = claude_desktop.unconfigure_claude_desktop_config(fake_config)

    assert changed is True
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in data
    assert data["inference"] == {}
    assert data["preferences"] == {"theme": "dark"}
    assert not claude_desktop._record_path(fake_config).exists()


def test_unconfigure_ignores_malformed_ownership_record(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    original = '{"modelDiscoveryEnabled": true}'
    fake_config.write_text(original, encoding="utf-8")
    claude_desktop._record_path(fake_config).write_text("{corrupt", encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is False

    assert fake_config.read_text(encoding="utf-8") == original


def test_configure_keeps_earliest_snapshot_across_repeated_merges(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    fake_config.write_text(
        json.dumps({"inference": {"user": "value"}}), encoding="utf-8"
    )

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    # A second merge with different settings must not clobber the snapshot.
    rotated = fake_settings.model_copy(update={"proxy_auth_token": "tok-456"})
    assert (
        claude_desktop.configure_claude_desktop_config(fake_config, settings=rotated)
        is True
    )

    record = json.loads(
        claude_desktop._record_path(fake_config).read_text(encoding="utf-8")
    )
    # The earliest snapshot had no managed keys; a clobbering second
    # snapshot would have recorded the values from the first merge.
    assert record["inferenceKeys"]["provider"] == {"present": False, "value": None}

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True
    assert json.loads(fake_config.read_text(encoding="utf-8")) == {
        "inference": {"user": "value"}
    }


def test_unconfigure_keeps_inference_keys_added_after_configure(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Unconfiguration must not remove inference entries FCC never touched."""

    fake_config.write_text(json.dumps({"inference": {}}), encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"]["userAddedAfterConfigure"] = "keep"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert result["inference"] == {"userAddedAfterConfigure": "keep"}


def test_unconfigure_preserves_user_replaced_inference_value(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A non-object ``inference`` after configure is user-owned, not ours."""

    fake_config.write_text(
        json.dumps({"inference": {"provider": "anthropic"}}), encoding="utf-8"
    )

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"] = "user-replaced-scalar"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    # Unconfigure still removes the FCC-owned discovery key.
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in result
    assert result["inference"] == "user-replaced-scalar"


def test_unconfigure_preserves_user_deleted_inference_entry(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Removing ``inference`` wholesale after configure must be respected."""

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    del data["inference"]
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    # Unconfigure still removes the FCC-owned discovery key.
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in result
    assert "inference" not in result


def test_unconfigure_keeps_additions_when_fcc_created_inference(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """An inference entry FCC created must not swallow later additions."""

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"]["userAddedAfterConfigure"] = "must-survive"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "modelDiscoveryEnabled" not in result
    assert result["inference"] == {"userAddedAfterConfigure": "must-survive"}


def test_unconfigure_restores_non_object_inference_entry(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Configure clobbers a non-object ``inference``; unconfigure restores it."""

    original = json.dumps({"inference": "legacy-scalar"})
    fake_config.write_text(original, encoding="utf-8")

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    assert json.loads(fake_config.read_text(encoding="utf-8"))["inference"] == (
        "legacy-scalar"
    )


def test_save_config_writes_owner_only_permissions(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """The token-bearing block must stay 0o600 even under a permissive umask."""

    old_umask = os.umask(0o000)
    try:
        assert (
            claude_desktop.configure_claude_desktop_config(
                fake_config, settings=fake_settings
            )
            is True
        )
    finally:
        os.umask(old_umask)

    assert (fake_config.stat().st_mode & 0o777) == 0o600
    record = claude_desktop._record_path(fake_config)
    assert (record.stat().st_mode & 0o777) == 0o600


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
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data == {}

    # Second passes are no-ops.
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is False


def test_configure_write_failure_leaves_existing_config_intact(
    fake_config: Path,
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed save (e.g. read-only directory) must not corrupt the config."""

    original = json.dumps({"preferences": {"theme": "dark"}})
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.write_text(original, encoding="utf-8")

    def refuse_write(path: Path, data: dict[str, object]) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(claude_desktop, "_save_config", refuse_write)

    with pytest.raises(PermissionError):
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )

    assert fake_config.read_text(encoding="utf-8") == original


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

    def fail_popen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configure must not spawn a subprocess")

    monkeypatch.setattr(claude_desktop.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(
        claude_desktop, "get_settings", lambda: fake_settings, raising=False
    )

    claude_desktop.main(["--configure", "--config-path", str(fake)])

    data = json.loads(fake.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True


def test_launch_binary_spawns_binary_without_certificate_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TLS trust comes from the NSS-installed CA, not a Chromium bypass flag."""

    calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            calls.append(cmd)

    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.launch_binary(["--extra"])

    assert calls == [["/usr/bin/claude", "--extra"]]
    assert all("--ignore-certificate-errors" not in cmd for cmd in calls)


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
    original_configure = claude_desktop.configure_claude_desktop_config

    def fake_configure(
        path: Path | None = None, settings: Settings | None = None
    ) -> bool:
        order.append("configure")
        return original_configure(path or config_path, settings=settings)

    class FakeProcess:
        def __init__(self, cmd: list[str]) -> None:
            order.append(f"spawn:{cmd[0]}")

    monkeypatch.setattr(
        claude_desktop, "configure_claude_desktop_config", fake_configure
    )
    monkeypatch.setattr(claude_desktop, "find_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(claude_desktop.subprocess, "Popen", FakeProcess)

    claude_desktop.ensure_configured_and_launch(settings=fake_settings)

    assert order[0] == "configure"
    assert order[1].startswith("spawn:")


def test_unconfigure_preserves_user_replaced_object_over_scalar_root(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A user object replacing FCC's generated mapping survives unconfigure."""

    fake_config.write_text(json.dumps({"inference": "legacy-scalar"}), encoding="utf-8")
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    replacement = {"provider": "user-choice", "model": "user-model"}
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"] = dict(replacement)
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert result["inference"] == replacement
    assert "modelDiscoveryEnabled" not in result


def test_unconfigure_preserves_user_deleted_entry_over_scalar_root(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Deleting ``inference`` wholesale after a scalar clobber is respected."""

    fake_config.write_text(json.dumps({"inference": "legacy-scalar"}), encoding="utf-8")
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    del data["inference"]
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert "inference" not in result


def test_unconfigure_respects_user_edited_discovery_key(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A post-merge user edit to ``modelDiscoveryEnabled`` wins on unconfigure."""

    fake_config.write_text(
        json.dumps({"modelDiscoveryEnabled": False}), encoding="utf-8"
    )
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    assert data["modelDiscoveryEnabled"] is True
    data["modelDiscoveryEnabled"] = False
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    # The pre-merge value was False and the user re-applied False: restored.
    assert result["modelDiscoveryEnabled"] is False


def test_unconfigure_leaves_user_disabled_discovery_from_absent_root(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """FCC-inserted discovery flipped off by the user is not deleted twice."""

    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )

    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["modelDiscoveryEnabled"] = False
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    result = json.loads(fake_config.read_text(encoding="utf-8"))
    assert result["modelDiscoveryEnabled"] is False


def test_unconfigure_preserves_user_edited_managed_key(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A post-merge user edit to a managed key wins over the snapshot."""

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"]["inferenceAnthropicApiKey"] = "user-rotated-key"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    out = json.loads(fake_config.read_text(encoding="utf-8"))
    assert out["inference"]["inferenceAnthropicApiKey"] == "user-rotated-key"
    # Untouched FCC-inserted keys are still removed.
    assert "provider" not in out["inference"]


def test_unconfigure_preserves_user_edited_fcc_created_key(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """Same rule when FCC created ``inference`` itself."""

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"]["inferenceGatewayBaseUrl"] = "http://user-chosen:9"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    out = json.loads(fake_config.read_text(encoding="utf-8"))
    assert out["inference"]["inferenceGatewayBaseUrl"] == "http://user-chosen:9"
    assert "provider" not in out["inference"]


def test_repeated_merges_absorb_post_merge_user_edits(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """A merge after a user edit must not resurrect the first snapshot."""

    original = {"modelDiscoveryEnabled": False, "custom": "keep"}
    fake_config.write_text(json.dumps({"inference": dict(original)}), encoding="utf-8")

    claude_desktop.configure_claude_desktop_config(fake_config, settings=fake_settings)
    data = json.loads(fake_config.read_text(encoding="utf-8"))
    data["inference"]["inferenceAnthropicApiKey"] = "user-key-v2"
    fake_config.write_text(json.dumps(data), encoding="utf-8")

    # Re-merge with rotated credentials; then reverse everything.
    rotated = fake_settings.model_copy(update={"proxy_auth_token": "tok-456"})
    assert (
        claude_desktop.configure_claude_desktop_config(fake_config, settings=rotated)
        is True
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True

    out = json.loads(fake_config.read_text(encoding="utf-8"))
    assert out["inference"]["inferenceAnthropicApiKey"] == "user-key-v2"
    assert out["inference"]["custom"] == "keep"


def test_repeated_merges_keep_container_provenance(
    fake_config: Path,
    fake_settings: Settings,
) -> None:
    """The origin container shape survives repeated merges.

    Regression guard for the Greptile "repeated merge loses container
    provenance" finding: a second merge sees the ``inference`` object the
    FIRST merge wrote, so re-snapshotting it would record
    ``existed=True, was_object=True`` and leave unconfigure restoring into
    an FCC-shaped container the user never had — an empty ``inference``
    where the entry was absent, or ``{}`` where the user held a scalar. The
    original provenance must be retained from the ownership record.
    """

    # Absent origin: after configure + rotated re-configure + unconfigure,
    # the inference entry must be gone entirely.
    fake_config.write_text(json.dumps({}), encoding="utf-8")
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    rotated = fake_settings.model_copy(update={"proxy_auth_token": "tok-456"})
    assert (
        claude_desktop.configure_claude_desktop_config(fake_config, settings=rotated)
        is True
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True
    assert json.loads(fake_config.read_text(encoding="utf-8")) == {}

    # Scalar origin: the user's non-object value must come back verbatim.
    fake_config.write_text(json.dumps({"inference": "legacy-scalar"}), encoding="utf-8")
    assert (
        claude_desktop.configure_claude_desktop_config(
            fake_config, settings=fake_settings
        )
        is True
    )
    assert (
        claude_desktop.configure_claude_desktop_config(fake_config, settings=rotated)
        is True
    )
    assert claude_desktop.unconfigure_claude_desktop_config(fake_config) is True
    assert json.loads(fake_config.read_text(encoding="utf-8")) == {
        "inference": "legacy-scalar"
    }


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
    unconfigure must restore — re-snapshotting the pre-FCC state would
    silently discard the user's change.
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
