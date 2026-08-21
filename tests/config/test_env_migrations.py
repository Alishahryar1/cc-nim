"""Contracts for one-time managed-config consolidation."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from free_claude_code.config import env_migrations
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.env_migrations import (
    CONFIG_SCHEMA_VERSION,
    HUGGINGFACE_TOKEN_MIGRATION,
    OPENCODE_ZEN_MODEL_REF_MIGRATIONS,
    consolidate_managed_config,
    migrate_env_setting_in_text,
)
from free_claude_code.config.loader import resolve_settings_snapshot


def _paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    managed = tmp_path / ".fcc" / ".env"
    legacy = tmp_path / "legacy" / ".env"
    monkeypatch.setattr(env_migrations, "managed_env_path", lambda: managed)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: (legacy,))
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    return managed, legacy


def test_pure_key_and_value_migrations_remain_available() -> None:
    migrated, changed = migrate_env_setting_in_text(
        "HF_TOKEN=legacy\n",
        HUGGINGFACE_TOKEN_MIGRATION,
    )
    assert changed is True
    assert migrated == "HUGGINGFACE_API_KEY=legacy\n"

    migration = next(
        item for item in OPENCODE_ZEN_MODEL_REF_MIGRATIONS if item.old_key == "MODEL"
    )
    migrated, changed = migrate_env_setting_in_text(
        "MODEL=opencode/model\n",
        migration,
    )
    assert changed is True
    assert migrated == "MODEL=opencode_zen/model\n"


def test_fresh_install_writes_only_schema_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)

    result = consolidate_managed_config({})

    assert result.changed is True
    assert dotenv_values_from_file(managed) == {
        "FCC_CONFIG_SCHEMA": CONFIG_SCHEMA_VERSION
    }


def test_legacy_file_imports_only_recognized_values_and_is_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    original = "MODEL=deepseek/legacy\nUNRELATED_SECRET=leave-me\nHF_TOKEN=hf\n"
    legacy.write_text(original, encoding="utf-8")

    result = consolidate_managed_config({})
    values = dotenv_values_from_file(managed)

    assert result.imported_from == (legacy.resolve(),)
    assert values["MODEL"] == "deepseek/legacy"
    assert values["HUGGINGFACE_API_KEY"] == "hf"
    assert values["PROXY_AUTH_ENABLED"] == "false"
    assert "UNRELATED_SECRET" not in values
    assert legacy.read_text(encoding="utf-8") == original


def test_existing_managed_state_does_not_merge_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    managed.write_text("MODEL=groq/managed\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True)
    legacy.write_text("MODEL=deepseek/legacy\nGROQ_API_KEY=legacy-key\n")

    consolidate_managed_config({})
    values = dotenv_values_from_file(managed)

    assert values["MODEL"] == "groq/managed"
    assert "GROQ_API_KEY" not in values


def test_verified_checkout_imports_only_when_managed_and_legacy_are_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    checkout = tmp_path / "checkout.env"
    original = "MODEL=deepseek/checkout\n"
    checkout.write_text(original, encoding="utf-8")
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(
        env_migrations,
        "verified_checkout_env_path",
        lambda: checkout,
    )

    result = consolidate_managed_config({})

    assert result.imported_from == (checkout.resolve(),)
    assert dotenv_values_from_file(managed)["MODEL"] == "deepseek/checkout"
    assert checkout.read_text(encoding="utf-8") == original


def test_canonical_setting_wins_and_retired_key_is_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "HF_TOKEN=old-token\nHUGGINGFACE_API_KEY=current-token\n",
        encoding="utf-8",
    )

    consolidate_managed_config({})
    values = dotenv_values_from_file(managed)

    assert values["HUGGINGFACE_API_KEY"] == "current-token"
    assert "HF_TOKEN" not in values


def test_explicit_file_overlays_once_and_then_becomes_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("MODEL=deepseek/base\nGROQ_API_KEY=base-key\n")
    explicit = tmp_path / "explicit.env"
    original = "MODEL=open_router/override\nGROQ_API_KEY=\n"
    explicit.write_text(original, encoding="utf-8")

    consolidate_managed_config({"FCC_ENV_FILE": str(explicit)})
    first = dotenv_values_from_file(managed)
    assert explicit.read_text(encoding="utf-8") == original
    explicit.write_text("MODEL=groq/later\n", encoding="utf-8")
    result = consolidate_managed_config({"FCC_ENV_FILE": str(explicit)})

    assert first["MODEL"] == "open_router/override"
    assert "GROQ_API_KEY" not in first
    assert result.changed is False
    assert dotenv_values_from_file(managed)["MODEL"] == "open_router/override"
    assert explicit.read_text(encoding="utf-8") == "MODEL=groq/later\n"


def test_post_schema_invalid_explicit_path_is_never_dereferenced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    managed.write_text("FCC_CONFIG_SCHEMA=1\n", encoding="utf-8")

    result = consolidate_managed_config({"FCC_ENV_FILE": str(tmp_path / "missing.env")})

    assert result.changed is False


def test_pre_schema_invalid_explicit_path_fails_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match=r"missing\.env"):
        consolidate_managed_config({"FCC_ENV_FILE": str(tmp_path / "missing.env")})

    assert not managed.exists()


def test_pre_schema_malformed_explicit_file_fails_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    explicit = tmp_path / "malformed.env"
    original = 'MODEL="unterminated\n'
    explicit.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=r"malformed\.env.*line 1"):
        consolidate_managed_config({"FCC_ENV_FILE": str(explicit)})

    assert not managed.exists()
    assert explicit.read_text(encoding="utf-8") == original


def test_explicit_managed_path_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)

    result = consolidate_managed_config({"FCC_ENV_FILE": str(managed)})

    assert result.imported_from == ()
    assert dotenv_values_from_file(managed) == {
        "FCC_CONFIG_SCHEMA": CONFIG_SCHEMA_VERSION
    }


@pytest.mark.parametrize(
    ("dotenv", "process", "enabled", "token"),
    [
        ("ANTHROPIC_AUTH_TOKEN=secret\n", {}, "true", "secret"),
        ("ANTHROPIC_AUTH_TOKEN=\n", {}, "false", None),
        ("", {"ANTHROPIC_AUTH_TOKEN": "process-secret"}, "true", None),
        ("", {}, "false", None),
        (
            "PROXY_AUTH_ENABLED=false\nANTHROPIC_AUTH_TOKEN=secret\n",
            {},
            "false",
            "secret",
        ),
    ],
)
def test_auth_state_is_materialized_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dotenv: str,
    process: dict[str, str],
    enabled: str,
    token: str | None,
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(dotenv, encoding="utf-8")

    consolidate_managed_config(process)
    values = dotenv_values_from_file(managed)

    assert values["PROXY_AUTH_ENABLED"] == enabled
    assert values.get("ANTHROPIC_AUTH_TOKEN") == token


def test_process_auth_flag_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)

    consolidate_managed_config({"PROXY_AUTH_ENABLED": "true"})

    assert "PROXY_AUTH_ENABLED" not in dotenv_values_from_file(managed)


def test_only_managed_schema_marker_is_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("FCC_CONFIG_SCHEMA=99\nMODEL=deepseek/imported\n")

    consolidate_managed_config({"FCC_CONFIG_SCHEMA": "99"})

    values = dotenv_values_from_file(managed)
    assert values["FCC_CONFIG_SCHEMA"] == "1"
    assert values["MODEL"] == "deepseek/imported"


def test_unknown_managed_assignments_survive_canonicalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    managed.write_text("MODEL=deepseek/model\nCUSTOM=\n", encoding="utf-8")

    consolidate_managed_config({})

    values = dotenv_values_from_file(managed)
    assert values["CUSTOM"] == ""
    assert values["MODEL"] == "deepseek/model"


def test_repeated_migration_is_byte_for_byte_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("MODEL=opencode/model\n", encoding="utf-8")

    consolidate_managed_config({})
    first = managed.read_bytes()
    result = consolidate_managed_config({})

    assert result.changed is False
    assert managed.read_bytes() == first


def test_duplicate_recognized_assignments_collapse_to_last_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    managed.write_text(
        "MODEL=deepseek/first\nMODEL=groq/last\n",
        encoding="utf-8",
    )

    consolidate_managed_config({})

    assert dotenv_values_from_file(managed)["MODEL"] == "groq/last"
    assert managed.read_text(encoding="utf-8").count("MODEL=") == 1


def test_concurrent_loaders_produce_one_valid_managed_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, legacy = _paths(monkeypatch, tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("MODEL=deepseek/concurrent\n", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = list(
            executor.map(lambda _index: resolve_settings_snapshot({}), range(8))
        )

    assert {snapshot.settings.model for snapshot in snapshots} == {
        "deepseek/concurrent"
    }
    assert dotenv_values_from_file(managed)["FCC_CONFIG_SCHEMA"] == "1"
    assert list(managed.parent.glob("*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_managed_config_is_owner_only_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)

    consolidate_managed_config({})

    assert managed.stat().st_mode & 0o777 == 0o600


def test_atomic_replace_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    original = "MODEL=deepseek/original\n"
    managed.write_text(original, encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        consolidate_managed_config({})

    assert managed.read_text(encoding="utf-8") == original
    assert list(managed.parent.glob("*.tmp")) == []


def test_newer_managed_schema_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    managed, _legacy = _paths(monkeypatch, tmp_path)
    managed.parent.mkdir(parents=True)
    managed.write_text("FCC_CONFIG_SCHEMA=2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema"):
        consolidate_managed_config({})
