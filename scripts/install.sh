#!/bin/sh
set -eu

REPO_GIT_URL="git+https://github.com/Alishahryar1/free-claude-code.git"
PYTHON_VERSION="3.14.0"
MIN_UV_VERSION="0.11.0"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"

dry_run=0
voice_nim=0
voice_local=0
voice_all=0
torch_backend=""

show_usage() {
    cat <<'USAGE'
Usage: install.sh [options]

Installs Claude Code and Codex if missing, installs or updates uv, Python 3.14.0, and Free Claude Code.

Options:
  --voice-nim              Install NVIDIA NIM voice transcription support.
  --voice-local            Install local Whisper voice transcription support.
  --voice-all              Install all voice transcription backends.
  --torch-backend VALUE    Use a uv PyTorch backend, such as cu130. Requires local voice.
  --dry-run                Print commands without running them.
  --help                   Show this help text.
USAGE
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

step() {
    printf '\n==> %s\n' "$1"
}

quote_arg() {
    case "$1" in
        *[!A-Za-z0-9_./:@%+=,-]*|"")
            escaped=$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')
            printf '"%s"' "$escaped"
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

print_command() {
    printf '+'
    for arg in "$@"; do
        printf ' '
        quote_arg "$arg"
    done
    printf '\n'
}

run() {
    print_command "$@"
    if [ "$dry_run" -eq 0 ]; then
        "$@"
    fi
}

run_uv_installer() {
    printf '+ curl -LsSf %s | sh\n' "$UV_INSTALL_URL"
    if [ "$dry_run" -eq 0 ]; then
        command -v curl >/dev/null 2>&1 || fail "curl is required to install uv."
        curl -LsSf "$UV_INSTALL_URL" | sh
    fi
}

add_path_entry() {
    [ -n "$1" ] || return 0
    case ":$PATH:" in
        *":$1:"*) ;;
        *) PATH="$1:$PATH" ;;
    esac
}

add_uv_to_path() {
    if [ -n "${XDG_BIN_HOME:-}" ]; then
        add_path_entry "$XDG_BIN_HOME"
    fi

    if [ -n "${HOME:-}" ]; then
        add_path_entry "$HOME/.local/bin"
        add_path_entry "$HOME/.cargo/bin"
    fi

    export PATH
}

require_command() {
    if [ "$dry_run" -eq 0 ] && ! command -v "$1" >/dev/null 2>&1; then
        fail "$1 is required. Install it first, then rerun this installer."
    fi
}

current_uv_version() {
    version=$(uv self version --short 2>/dev/null || true)
    if [ -z "$version" ]; then
        version=$(uv --version 2>/dev/null | sed 's/^uv //; s/ .*//' || true)
    fi

    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

version_ge() {
    current=${1%%[-+]*}
    minimum=${2%%[-+]*}

    old_ifs=$IFS
    IFS=.
    set -- $current
    current_major=${1:-0}
    current_minor=${2:-0}
    current_patch=${3:-0}
    set -- $minimum
    minimum_major=${1:-0}
    minimum_minor=${2:-0}
    minimum_patch=${3:-0}
    IFS=$old_ifs

    [ "$current_major" -gt "$minimum_major" ] && return 0
    [ "$current_major" -lt "$minimum_major" ] && return 1
    [ "$current_minor" -gt "$minimum_minor" ] && return 0
    [ "$current_minor" -lt "$minimum_minor" ] && return 1
    [ "$current_patch" -ge "$minimum_patch" ]
}

uv_version_satisfies_minimum() {
    version=$(current_uv_version) || return 1
    version_ge "$version" "$MIN_UV_VERSION"
}

validate_uv_version() {
    [ "$dry_run" -eq 1 ] && return 0

    version=$(current_uv_version) || fail "Unable to determine uv version."
    if ! version_ge "$version" "$MIN_UV_VERSION"; then
        fail "uv $MIN_UV_VERSION or newer is required; found uv $version. Upgrade uv with its installer or package manager, then rerun this installer."
    fi
}

uv_self_update_supported() {
    uv self update --dry-run >/dev/null 2>&1
}

uv_installed_by_homebrew() {
    command -v brew >/dev/null 2>&1 && brew list --versions uv >/dev/null 2>&1
}

uv_installed_by_pipx() {
    command -v pipx >/dev/null 2>&1 && pipx list 2>/dev/null | grep -Eq '(^|[[:space:]])package uv([[:space:]]|$)'
}

uv_installed_in_active_virtualenv() {
    [ -n "${VIRTUAL_ENV:-}" ] || return 1

    uv_path=$(command -v uv)
    case "$uv_path" in
        "$VIRTUAL_ENV"/*) return 0 ;;
        *) return 1 ;;
    esac
}

update_existing_uv() {
    if uv_self_update_supported; then
        run uv self update
        return 0
    fi

    if uv_installed_by_homebrew; then
        run brew upgrade uv
        return 0
    fi

    if uv_installed_by_pipx; then
        run pipx upgrade uv
        return 0
    fi

    if uv_installed_in_active_virtualenv; then
        run python -m pip install --upgrade uv
        return 0
    fi

    if uv_version_satisfies_minimum; then
        printf 'uv is already installed and satisfies >=%s; skipping automatic uv update because the install source was not detected.\n' "$MIN_UV_VERSION"
        return 0
    fi

    version=$(current_uv_version 2>/dev/null || printf 'unknown')
    fail "uv $MIN_UV_VERSION or newer is required; found uv $version. The existing uv install source was not detected. Upgrade uv manually with the package manager that installed it, then rerun this installer."
}

install_claude_if_missing() {
    if command -v claude >/dev/null 2>&1; then
        printf 'Claude Code already found on PATH; skipping install.\n'
        return 0
    fi

    require_command npm
    run npm install -g @anthropic-ai/claude-code
}

install_codex_if_missing() {
    if command -v codex >/dev/null 2>&1; then
        printf 'Codex already found on PATH; skipping install.\n'
        return 0
    fi

    require_command npm
    run npm install -g @openai/codex
}

install_or_update_uv() {
    add_uv_to_path

    if command -v uv >/dev/null 2>&1; then
        update_existing_uv
        validate_uv_version
        return 0
    fi

    run_uv_installer
    add_uv_to_path

    if [ "$dry_run" -eq 0 ] && ! command -v uv >/dev/null 2>&1; then
        fail "uv was installed, but it is not available on PATH. Open a new terminal or add uv's bin directory to PATH."
    fi

    validate_uv_version
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --voice-nim)
                voice_nim=1
                ;;
            --voice-local)
                voice_local=1
                ;;
            --voice-all)
                voice_all=1
                ;;
            --torch-backend)
                shift
                [ "$#" -gt 0 ] || fail "--torch-backend requires a value."
                torch_backend="$1"
                [ -n "$torch_backend" ] || fail "--torch-backend requires a non-empty value."
                ;;
            --torch-backend=*)
                torch_backend="${1#*=}"
                [ -n "$torch_backend" ] || fail "--torch-backend requires a non-empty value."
                ;;
            --dry-run)
                dry_run=1
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                show_usage >&2
                fail "unknown option: $1"
                ;;
        esac
        shift
    done
}

validate_args() {
    include_local=$voice_local

    if [ "$voice_all" -eq 1 ]; then
        include_local=1
    fi

    if [ -n "$torch_backend" ] && [ "$include_local" -ne 1 ]; then
        fail "--torch-backend requires --voice-local or --voice-all."
    fi
}

package_spec() {
    include_nim=$voice_nim
    include_local=$voice_local

    if [ "$voice_all" -eq 1 ]; then
        include_nim=1
        include_local=1
    fi

    if [ -n "$torch_backend" ] && [ "$include_local" -ne 1 ]; then
        fail "--torch-backend requires --voice-local or --voice-all."
    fi

    if [ "$include_nim" -eq 1 ] && [ "$include_local" -eq 1 ]; then
        printf 'free-claude-code[voice,voice_local] @ %s' "$REPO_GIT_URL"
    elif [ "$include_nim" -eq 1 ]; then
        printf 'free-claude-code[voice] @ %s' "$REPO_GIT_URL"
    elif [ "$include_local" -eq 1 ]; then
        printf 'free-claude-code[voice_local] @ %s' "$REPO_GIT_URL"
    else
        printf '%s' "$REPO_GIT_URL"
    fi
}

install_free_claude_code() {
    spec=$(package_spec)

    if [ -n "$torch_backend" ]; then
        run uv tool install --force --torch-backend "$torch_backend" "$spec"
    else
        run uv tool install --force "$spec"
    fi
}

parse_args "$@"
validate_args

step "Installing Claude Code if missing"
install_claude_if_missing

step "Installing Codex if missing"
install_codex_if_missing

step "Installing uv if missing, updating if present"
install_or_update_uv

step "Installing Python $PYTHON_VERSION"
run uv python install "$PYTHON_VERSION"

step "Installing or updating Free Claude Code"
install_free_claude_code

printf '\nFree Claude Code is installed. Start the proxy with: fcc-server\n'
printf 'Run Claude Code with: fcc-claude\n'
printf 'Run Codex with: fcc-codex\n'
