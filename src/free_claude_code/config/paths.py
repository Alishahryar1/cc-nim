"""Shared filesystem paths for Free Claude Code configuration."""

import os
from pathlib import Path

FCC_CONFIG_DIRNAME = ".fcc"
FCC_ENV_FILENAME = ".env"
LEGACY_REPO_DIRNAME = "free-claude-code"
LEGACY_XDG_CONFIG_DIRNAME = ".config"
MESSAGING_STATE_DIRNAME = "agent_workspace"
FCC_LOGS_DIRNAME = "logs"
SERVER_LOG_FILENAME = "server.log"
CODEX_MODEL_CATALOG_FILENAME = "codex-model-catalog.json"
AUTH_DIRNAME = "auth"
OPENAI_AUTH_FILENAME = "openai.json"
OPENAI_AUTH_LOCK_FILENAME = "openai.lock"
CONFIG_LOCK_FILENAME = "config.lock"
FCC_TEMP_DIRNAME = "tmp"
AIDER_TEMP_DIRNAME = "aider"
CHAT_STATE_DIRNAME = "chat"
CHAT_DATABASE_FILENAME = "chat.db"
CHAT_LOCK_FILENAME = "chat.lock"
TERMINAL_DIRNAME = "terminal"
TERMINAL_BIN_DIRNAME = "bin"
TERMINAL_SOCKET_DIRNAME = "sockets"
TERMINAL_RUNTIME_DIRNAME = "runtime"
TERMINAL_LOCK_FILENAME = "terminal.lock"
TERMINAL_CONFIG_FILENAME = "config.kdl"


def config_dir_path() -> Path:
    """Return the default user config directory."""

    return Path.home() / FCC_CONFIG_DIRNAME


def managed_env_path() -> Path:
    """Return the default user-managed env file path."""

    return config_dir_path() / FCC_ENV_FILENAME


def config_lock_path() -> Path:
    """Return the cross-process managed-config migration lock path."""

    return config_dir_path() / CONFIG_LOCK_FILENAME


def aider_temp_dir_path() -> Path:
    """Return the base directory for managed per-launch Aider files."""

    return config_dir_path() / FCC_TEMP_DIRNAME / AIDER_TEMP_DIRNAME


def chat_state_dir_path() -> Path:
    """Return the managed Chat Sessions state directory."""

    return config_dir_path() / CHAT_STATE_DIRNAME


def chat_database_path() -> Path:
    """Return the managed Chat Sessions database path."""

    return chat_state_dir_path() / CHAT_DATABASE_FILENAME


def chat_lock_path() -> Path:
    """Return the exclusive Chat Sessions process-lock path."""

    return chat_state_dir_path() / CHAT_LOCK_FILENAME


def terminal_state_dir_path() -> Path:
    """Return the private managed terminal-engine directory."""

    return config_dir_path() / TERMINAL_DIRNAME


def terminal_binary_path() -> Path:
    """Return the managed Zellij executable path for this platform."""

    executable = "zellij.exe" if os.name == "nt" else "zellij"
    return terminal_state_dir_path() / TERMINAL_BIN_DIRNAME / executable


def terminal_socket_dir_path() -> Path:
    """Return FCC's isolated Zellij socket namespace."""

    return terminal_state_dir_path() / TERMINAL_SOCKET_DIRNAME


def terminal_runtime_dir_path() -> Path:
    """Return the ephemeral managed terminal runtime directory."""

    return terminal_state_dir_path() / TERMINAL_RUNTIME_DIRNAME


def terminal_config_path() -> Path:
    """Return the generated FCC-only Zellij configuration path."""

    return terminal_runtime_dir_path() / TERMINAL_CONFIG_FILENAME


def terminal_lock_path() -> Path:
    """Return the exclusive terminal-engine owner lock path."""

    return terminal_state_dir_path() / TERMINAL_LOCK_FILENAME


def legacy_env_paths() -> tuple[Path, ...]:
    """Return legacy user env paths that can be migrated to ~/.fcc/.env."""

    home = Path.home()
    return (
        home / LEGACY_REPO_DIRNAME / FCC_ENV_FILENAME,
        home / LEGACY_XDG_CONFIG_DIRNAME / LEGACY_REPO_DIRNAME / FCC_ENV_FILENAME,
    )


def messaging_state_dir_path() -> Path:
    """Return the managed messaging state directory."""

    return config_dir_path() / MESSAGING_STATE_DIRNAME


def server_log_path() -> Path:
    """Return the canonical server log path."""

    return config_dir_path() / FCC_LOGS_DIRNAME / SERVER_LOG_FILENAME


def codex_model_catalog_path() -> Path:
    """Return the generated Codex model catalog path."""

    return config_dir_path() / CODEX_MODEL_CATALOG_FILENAME


def openai_auth_path() -> Path:
    """Return FCC's private ChatGPT credential file path."""

    return config_dir_path() / AUTH_DIRNAME / OPENAI_AUTH_FILENAME


def openai_auth_lock_path() -> Path:
    """Return the cross-process lock path for ChatGPT credentials."""

    return config_dir_path() / AUTH_DIRNAME / OPENAI_AUTH_LOCK_FILENAME
