#!/bin/sh
set -eu

REPO_ARCHIVE_URL="https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
PYTHON_VERSION="3.14.0"
MIN_UV_VERSION="0.11.16"
CLAUDE_INSTALL_URL="https://claude.ai/install.sh"
CODEX_INSTALL_URL="https://chatgpt.com/codex/install.sh"
PI_INSTALL_URL="https://pi.dev/install.sh"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
FCC_MACOS_BUNDLE_ID="io.github.alishahryar1.free-claude-code"
FCC_MACOS_OWNER_FILE=".free-claude-code-owner"
# Include retired entry points so updates reject older FCC processes before replacement.
FCC_COMMANDS="fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-init free-claude-code"

dry_run=0
voice_nim=0
voice_local=0
voice_all=0
torch_backend=""
temporary_script=""
tool_bin=""

show_usage() {
    cat <<'USAGE'
Usage: install.sh [options]

Installs Claude Code, Codex, and Pi if missing, ensures a compatible uv, and installs or updates Free Claude Code.

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
    if [ "$dry_run" -eq 1 ]; then
        return 0
    fi

    if "$@"; then
        return 0
    else
        status=$?
    fi

    fail "Command failed with exit code $status: $1"
}

cleanup() {
    if [ -n "$temporary_script" ] && [ -e "$temporary_script" ]; then
        rm -f "$temporary_script"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

add_path_entry() {
    [ -n "$1" ] || return 0
    case ":$PATH:" in
        *":$1:"*) ;;
        *) PATH="$1:$PATH" ;;
    esac
}

add_known_bin_directories() {
    if [ -n "${XDG_BIN_HOME:-}" ]; then
        add_path_entry "$XDG_BIN_HOME"
    fi

    if [ -n "${HOME:-}" ]; then
        add_path_entry "$HOME/.local/bin"
        add_path_entry "$HOME/.cargo/bin"
        add_path_entry "${XDG_DATA_HOME:-$HOME/.local/share}/pi-node/current/bin"
    fi

    export PATH
    hash -r 2>/dev/null || true
}

add_pi_bin_directories() {
    [ "$dry_run" -eq 0 ] || return 0
    add_known_bin_directories
    if command -v npm >/dev/null 2>&1; then
        pi_npm_prefix=$(npm prefix -g 2>/dev/null || npm config get prefix 2>/dev/null || true)
        if [ -n "$pi_npm_prefix" ]; then
            add_path_entry "$pi_npm_prefix/bin"
            export PATH
            hash -r 2>/dev/null || true
        fi
    fi
}

fcc_process_ids() {
    command_name=$1

    if command -v pgrep >/dev/null 2>&1; then
        {
            pgrep -x "$command_name" 2>/dev/null || true
            pgrep -f "(^|/)${command_name}([[:space:]]|$)" 2>/dev/null || true
        } | sort -nu
        return 0
    fi

    ps -A -o pid= -o args= 2>/dev/null |
        awk -v command_name="$command_name" '
            BEGIN {
                pattern = "(^|/)" command_name "([[:space:]]|$)"
            }
            {
                process_id = $1
                sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "")
                if ($0 ~ pattern) {
                    print process_id
                }
            }
        ' || true
}

assert_no_fcc_processes_running() {
    running=""
    for command_name in $FCC_COMMANDS; do
        process_ids=$(fcc_process_ids "$command_name")
        [ -n "$process_ids" ] || continue

        for process_id in $process_ids; do
            process="$command_name (PID $process_id)"
            if [ -n "$running" ]; then
                running="$running, $process"
            else
                running=$process
            fi
        done
    done

    if [ -n "$running" ]; then
        fail "Free Claude Code is still running ($running). Stop those processes, then rerun the installer."
    fi
}

require_command() {
    if [ "$dry_run" -eq 0 ] && ! command -v "$1" >/dev/null 2>&1; then
        fail "$1 is required. Install it first, then rerun this installer."
    fi
}

download_and_run() {
    url=$1
    interpreter=$2
    label=$3
    non_interactive=${4:-0}

    if [ "$dry_run" -eq 1 ]; then
        print_command curl -fsSL "$url" -o "<temporary-script>"
        if [ "$non_interactive" -eq 1 ]; then
            printf '+ CODEX_NON_INTERACTIVE=1 '
            quote_arg "$interpreter"
            printf ' <temporary-script>\n'
        else
            print_command "$interpreter" "<temporary-script>"
        fi
        return 0
    fi

    temporary_script=$(mktemp "${TMPDIR:-/tmp}/fcc-install.XXXXXX") || fail "Unable to create a temporary file for $label."
    print_command curl -fsSL "$url" -o "$temporary_script"
    if curl -fsSL "$url" -o "$temporary_script"; then
        :
    else
        status=$?
        fail "Could not download the $label installer (curl exit code $status)."
    fi

    if [ ! -s "$temporary_script" ]; then
        fail "The downloaded $label installer was empty."
    fi

    if [ "$non_interactive" -eq 1 ]; then
        printf '+ CODEX_NON_INTERACTIVE=1 '
        quote_arg "$interpreter"
        printf ' '
        quote_arg "$temporary_script"
        printf '\n'
        if CODEX_NON_INTERACTIVE=1 "$interpreter" "$temporary_script"; then
            :
        else
            status=$?
            fail "$label installation failed with exit code $status."
        fi
    else
        print_command "$interpreter" "$temporary_script"
        if "$interpreter" "$temporary_script"; then
            :
        else
            status=$?
            fail "$label installation failed with exit code $status."
        fi
    fi

    rm -f "$temporary_script"
    temporary_script=""
}

verify_command() {
    command_name=$1
    display_name=$2

    if [ "$dry_run" -eq 1 ]; then
        print_command "$command_name" --version
        return 0
    fi

    command_path=$(command -v "$command_name" 2>/dev/null) || fail "$display_name was installed, but '$command_name' is not available on PATH."
    run "$command_path" --version
}

pi_command_is_compatible() {
    pi_command_path=$(command -v pi 2>/dev/null) || return 1
    pi_help=$("$pi_command_path" --help 2>/dev/null) || return 1
    case "$pi_help" in
        *--extension*) ;;
        *) return 1 ;;
    esac
    case "$pi_help" in
        *--models*) return 0 ;;
        *) return 1 ;;
    esac
}

verify_pi_command() {
    if [ "$dry_run" -eq 1 ]; then
        printf '+ pi --help (verify --extension and --models support)\n'
        print_command pi --version
        return 0
    fi

    pi_command_path=$(command -v pi 2>/dev/null) || fail "Pi was installed, but 'pi' is not available on PATH."
    pi_command_is_compatible || fail "The 'pi' command at $pi_command_path is not a compatible Pi Coding Agent."
    run "$pi_command_path" --version
}

ensure_claude() {
    if command -v claude >/dev/null 2>&1; then
        printf 'Claude Code already found on PATH; verifying it.\n'
    else
        download_and_run "$CLAUDE_INSTALL_URL" bash "Claude Code"
        add_known_bin_directories
    fi

    verify_command claude "Claude Code"
}

ensure_codex() {
    if command -v codex >/dev/null 2>&1; then
        printf 'Codex already found on PATH; verifying it.\n'
    else
        download_and_run "$CODEX_INSTALL_URL" sh "Codex" 1
        add_known_bin_directories
    fi

    verify_command codex "Codex"
}

ensure_pi() {
    if [ "$dry_run" -eq 1 ] && command -v pi >/dev/null 2>&1; then
        printf 'Pi already found on PATH; verifying it.\n'
    elif pi_command_is_compatible; then
        printf 'Pi already found on PATH; verifying it.\n'
    else
        if existing_pi_path=$(command -v pi 2>/dev/null); then
            printf "The existing 'pi' command at %s is not Pi Coding Agent; installing Pi.\n" "$existing_pi_path"
        fi
        download_and_run "$PI_INSTALL_URL" sh "Pi"
        add_pi_bin_directories
    fi

    verify_pi_command
}

current_uv_version() {
    if output=$(uv --version); then
        :
    else
        return 1
    fi

    case "$output" in
        uv\ *) version=${output#uv } ;;
        *) version=$output ;;
    esac
    version=${version%% *}

    case "$version" in
        [0-9]*.[0-9]*.[0-9]*) printf '%s\n' "$version" ;;
        *) return 1 ;;
    esac
}

uv_version_is_supported() {
    case "$1" in
        *-*) return 1 ;;
    esac

    current=${1%%+*}
    minimum=${2%%+*}

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

    case "$current_major$current_minor$current_patch$minimum_major$minimum_minor$minimum_patch" in
        *[!0-9]*) return 1 ;;
    esac

    [ "$current_major" -gt "$minimum_major" ] && return 0
    [ "$current_major" -lt "$minimum_major" ] && return 1
    [ "$current_minor" -gt "$minimum_minor" ] && return 0
    [ "$current_minor" -lt "$minimum_minor" ] && return 1
    [ "$current_patch" -ge "$minimum_patch" ]
}

verify_uv() {
    if [ "$dry_run" -eq 1 ]; then
        print_command uv --version
        return 0
    fi

    command -v uv >/dev/null 2>&1 || fail "uv was installed, but it is not available on PATH."
    version=$(current_uv_version) || fail "uv is present, but 'uv --version' did not return a valid version."
    if ! uv_version_is_supported "$version" "$MIN_UV_VERSION"; then
        fail "Stable uv $MIN_UV_VERSION or newer is required; found uv $version after installation."
    fi

    printf 'Verified uv %s.\n' "$version"
}

ensure_uv() {
    if [ "$dry_run" -eq 1 ]; then
        if command -v uv >/dev/null 2>&1; then
            print_command uv --version
            printf 'A compatible existing uv will be left unchanged; an obsolete one will be replaced by the standalone installer.\n'
        else
            printf 'uv is not installed; the current standalone uv would be installed.\n'
            download_and_run "$UV_INSTALL_URL" sh "uv"
            verify_uv
        fi
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        version=$(current_uv_version) || fail "uv is present, but 'uv --version' did not return a valid version."
        if uv_version_is_supported "$version" "$MIN_UV_VERSION"; then
            printf 'uv %s already satisfies >=%s; leaving it unchanged.\n' "$version" "$MIN_UV_VERSION"
            return 0
        fi
        printf 'uv %s does not satisfy stable >=%s; installing the current standalone uv.\n' "$version" "$MIN_UV_VERSION"
    else
        printf 'uv is not installed; installing the current standalone uv.\n'
    fi

    download_and_run "$UV_INSTALL_URL" sh "uv"
    add_known_bin_directories
    verify_uv
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
                torch_backend=$1
                [ -n "$torch_backend" ] || fail "--torch-backend requires a non-empty value."
                ;;
            --torch-backend=*)
                torch_backend=${1#*=}
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

    if [ "$include_nim" -eq 1 ] && [ "$include_local" -eq 1 ]; then
        printf 'free-claude-code[voice,voice_local] @ %s' "$REPO_ARCHIVE_URL"
    elif [ "$include_nim" -eq 1 ]; then
        printf 'free-claude-code[voice] @ %s' "$REPO_ARCHIVE_URL"
    elif [ "$include_local" -eq 1 ]; then
        printf 'free-claude-code[voice_local] @ %s' "$REPO_ARCHIVE_URL"
    else
        printf 'free-claude-code @ %s' "$REPO_ARCHIVE_URL"
    fi
}

install_free_claude_code() {
    assert_no_fcc_processes_running
    spec=$(package_spec)

    if [ -n "$torch_backend" ]; then
        run uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" --torch-backend "$torch_backend" "$spec"
    else
        run uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" "$spec"
    fi
}

configure_and_verify_free_claude_code() {
    run uv tool update-shell

    if [ "$dry_run" -eq 1 ]; then
        print_command uv tool dir --bin
        printf '+ verify fcc-desktop, fcc-server, fcc-claude, fcc-codex, and fcc-pi in the uv tool bin directory\n'
        print_command fcc-server --version
        return 0
    fi

    print_command uv tool dir --bin
    if tool_bin=$(uv tool dir --bin); then
        :
    else
        status=$?
        fail "Could not determine the uv tool bin directory (exit code $status)."
    fi
    [ -n "$tool_bin" ] || fail "uv returned an empty tool bin directory."

    add_path_entry "$tool_bin"
    export PATH
    hash -r 2>/dev/null || true

    for command_name in fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi; do
        [ -x "$tool_bin/$command_name" ] || fail "Free Claude Code installation did not create $tool_bin/$command_name."
    done

    run "$tool_bin/fcc-server" --version
}

shell_quote() {
    escaped=$(printf '%s' "$1" | sed "s/'/'\\\\''/g")
    printf "'%s'" "$escaped"
}

macos_app_is_fcc_owned() {
    app_dir=$1
    owner_file="$app_dir/Contents/$FCC_MACOS_OWNER_FILE"
    [ -d "$app_dir" ] &&
        [ ! -L "$app_dir" ] &&
        [ -f "$owner_file" ] &&
        [ "$(cat "$owner_file")" = "$FCC_MACOS_BUNDLE_ID" ]
}

install_macos_desktop_app() {
    [ "$(uname -s)" = "Darwin" ] || return 0

    app_dir="$HOME/Applications/Free Claude Code.app"
    contents_dir="$app_dir/Contents"
    owner_file="$contents_dir/$FCC_MACOS_OWNER_FILE"
    executable_dir="$contents_dir/MacOS"
    executable_path="$executable_dir/fcc-desktop"
    desktop_dir="$HOME/Desktop"
    desktop_link="$desktop_dir/Free Claude Code.app"

    if [ -e "$app_dir" ] || [ -L "$app_dir" ]; then
        macos_app_is_fcc_owned "$app_dir" ||
            fail "An app not managed by Free Claude Code already exists at $app_dir. Move it, then rerun the installer."
    fi

    if [ "$dry_run" -eq 1 ]; then
        print_command mkdir -p "$executable_dir" "$desktop_dir"
        printf '+ write %s, %s, and %s\n' "$owner_file" "$contents_dir/Info.plist" "$executable_path"
        print_command ln -s "$app_dir" "$desktop_link"
        return 0
    fi

    mkdir -p "$executable_dir" "$desktop_dir"
    printf '%s\n' "$FCC_MACOS_BUNDLE_ID" > "$owner_file"
    cat > "$contents_dir/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Free Claude Code</string>
    <key>CFBundleExecutable</key>
    <string>fcc-desktop</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.alishahryar1.free-claude-code</string>
    <key>CFBundleName</key>
    <string>Free Claude Code</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMultipleInstancesProhibited</key>
    <true/>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
PLIST
    desktop_command=$(shell_quote "$tool_bin/fcc-desktop")
    {
        printf '%s\n' '#!/bin/sh'
        printf 'exec %s\n' "$desktop_command"
    } > "$executable_path"
    chmod +x "$executable_path"

    if [ -L "$desktop_link" ]; then
        if [ "$(readlink "$desktop_link")" = "$app_dir" ]; then
            rm -f "$desktop_link"
        else
            printf 'A non-FCC link already exists at %s; leaving it unchanged.\n' "$desktop_link"
            return 0
        fi
    elif [ -e "$desktop_link" ]; then
        printf 'A non-FCC item already exists at %s; leaving it unchanged.\n' "$desktop_link"
        return 0
    fi
    ln -s "$app_dir" "$desktop_link"
}

parse_args "$@"
validate_args
add_known_bin_directories

step "Checking for running Free Claude Code processes"
assert_no_fcc_processes_running

step "Checking installation prerequisites"
require_command curl
require_command bash
require_command sh
require_command mktemp

step "Ensuring Claude Code is installed"
ensure_claude

step "Ensuring Codex is installed"
ensure_codex

step "Ensuring Pi is installed"
ensure_pi

step "Ensuring uv $MIN_UV_VERSION or newer is installed"
ensure_uv

step "Installing or updating Free Claude Code"
install_free_claude_code

step "Configuring PATH and verifying Free Claude Code"
configure_and_verify_free_claude_code

if [ "$(uname -s)" = "Darwin" ]; then
    step "Installing the Free Claude Code desktop launcher"
    install_macos_desktop_app
fi

if [ "$dry_run" -eq 1 ]; then
    printf '\nDry run complete. No changes were made.\n'
else
    if [ "$(uname -s)" = "Darwin" ]; then
        printf '\nFree Claude Code is installed and verified. Open Free Claude Code from Applications or the desktop to run it in the background.\n'
        printf 'For terminal use, start the proxy with: fcc-server\n'
    else
        printf '\nFree Claude Code is installed and verified. Start the proxy with: fcc-server\n'
    fi
    printf 'Run Claude Code with: fcc-claude\n'
    printf 'Run Codex with: fcc-codex\n'
    printf 'Run Pi with: fcc-pi\n'
fi
