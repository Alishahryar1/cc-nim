from pathlib import Path

from config.env_migrations import (
    HUGGINGFACE_API_KEY_ENV,
    HUGGINGFACE_TOKEN_MIGRATION,
    LEGACY_HUGGINGFACE_TOKEN_ENV,
    env_text_needs_migration,
    explicit_env_file_huggingface_warning,
    migrate_env_key_in_file,
    migrate_env_key_in_text,
    migrate_owned_env_files,
)


def test_migrate_env_key_in_text_renames_legacy_hf_token() -> None:
    text = "# comment\nHF_TOKEN=old-token\nMODEL=nvidia_nim/model\n"

    migrated, changed = migrate_env_key_in_text(text, HUGGINGFACE_TOKEN_MIGRATION)

    assert changed is True
    assert migrated == (
        "# comment\nHUGGINGFACE_API_KEY=old-token\nMODEL=nvidia_nim/model\n"
    )


def test_migrate_env_key_in_text_preserves_existing_huggingface_api_key() -> None:
    text = "HF_TOKEN=old-token\nHUGGINGFACE_API_KEY=new-token\n"

    migrated, changed = migrate_env_key_in_text(text, HUGGINGFACE_TOKEN_MIGRATION)

    assert changed is False
    assert migrated == text


def test_migrate_env_key_in_text_ignores_comments() -> None:
    text = "# HF_TOKEN=old-token\n"

    migrated, changed = migrate_env_key_in_text(text, HUGGINGFACE_TOKEN_MIGRATION)

    assert changed is False
    assert migrated == text
    assert not env_text_needs_migration(text, HUGGINGFACE_TOKEN_MIGRATION)


def test_migrate_env_key_in_file_rewrites_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("export HF_TOKEN = quoted-token\n", encoding="utf-8")

    assert migrate_env_key_in_file(env_file, HUGGINGFACE_TOKEN_MIGRATION) is True

    assert env_file.read_text(encoding="utf-8") == (
        "export HUGGINGFACE_API_KEY = quoted-token\n"
    )


def test_migrate_owned_env_files_rewrites_repo_and_managed_env(
    monkeypatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    managed = tmp_path / ".fcc" / ".env"
    managed.parent.mkdir()
    (repo / ".env").write_text("HF_TOKEN=repo-token\n", encoding="utf-8")
    managed.write_text("HF_TOKEN=managed-token\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    migrated = migrate_owned_env_files()

    assert migrated == (repo / ".env", managed)
    assert (repo / ".env").read_text(encoding="utf-8") == (
        "HUGGINGFACE_API_KEY=repo-token\n"
    )
    assert managed.read_text(encoding="utf-8") == (
        "HUGGINGFACE_API_KEY=managed-token\n"
    )


def test_explicit_env_file_huggingface_warning_does_not_rewrite(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "custom.env"
    explicit.write_text("HF_TOKEN=explicit-token\n", encoding="utf-8")

    warning = explicit_env_file_huggingface_warning({"FCC_ENV_FILE": str(explicit)})

    assert warning is not None
    assert str(explicit) in warning
    assert LEGACY_HUGGINGFACE_TOKEN_ENV in warning
    assert HUGGINGFACE_API_KEY_ENV in warning
    assert explicit.read_text(encoding="utf-8") == "HF_TOKEN=explicit-token\n"
