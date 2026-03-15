#!/usr/bin/env bash
# install.sh — set up free-claude-code and register claude-free / claude-free-server commands
set -euo pipefail

# ── colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}${BOLD}[•]${NC} $*"; }
success() { echo -e "${GREEN}${BOLD}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[!]${NC} $*"; }
die()     { echo -e "${RED}${BOLD}[✗]${NC} $*" >&2; exit 1; }

INSTALL_DIR="${FREE_CLAUDE_INSTALL_DIR:-$HOME/Applications/free-claude-code}"
BIN_DIR="$HOME/.local/bin"
PORT="${FREE_CLAUDE_PORT:-8082}"
REPO_URL="https://github.com/Alishahryar1/free-claude-code.git"

echo -e "\n${BOLD}free-claude-code installer${NC}\n"

# ── 1. prerequisites ────────────────────────────────────────────────────────────
info "Checking prerequisites..."

if ! command -v git &>/dev/null; then
    die "git not found. Install git and re-run."
fi

if ! command -v claude &>/dev/null; then
    die "claude (Claude Code CLI) not found.\nInstall it from https://github.com/anthropics/claude-code and re-run."
fi

if ! command -v uv &>/dev/null; then
    warn "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # make uv available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        die "uv install failed. Install manually from https://github.com/astral-sh/uv and re-run."
    fi
    success "uv installed."
fi

UV_BIN="$(command -v uv)"
success "Prerequisites OK  (uv: $UV_BIN)"

# ── 2. clone or update ──────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing repo at $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    # If the script is being run from inside the repo (not via curl|bash), install in-place
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/server.py" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
        info "Running from repo — installing in-place at $INSTALL_DIR ..."
        if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
            mkdir -p "$(dirname "$INSTALL_DIR")"
            cp -r "$SCRIPT_DIR" "$INSTALL_DIR"
        fi
    else
        info "Cloning repo to $INSTALL_DIR ..."
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi
fi
success "Repo at $INSTALL_DIR"

# ── 3. python venv ──────────────────────────────────────────────────────────────
info "Setting up Python virtual environment..."
( cd "$INSTALL_DIR" && "$UV_BIN" sync )
success "Virtual environment ready"

# ── 4. .env ─────────────────────────────────────────────────────────────────────
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    warn ".env created from .env.example — edit $INSTALL_DIR/.env to set your API key."
else
    info ".env already exists, skipping."
fi

# ── 5. ~/.local/bin scripts ─────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"

info "Writing $BIN_DIR/claude-free-server ..."
cat > "$BIN_DIR/claude-free-server" <<SCRIPT
#!/usr/bin/env bash
cd "${INSTALL_DIR}" && .venv/bin/uvicorn server:app --host 0.0.0.0 --port ${PORT}
SCRIPT
chmod +x "$BIN_DIR/claude-free-server"

info "Writing $BIN_DIR/claude-free ..."
cat > "$BIN_DIR/claude-free" <<SCRIPT
#!/usr/bin/env bash
_SERVER_STARTED=0

if ! curl -s --max-time 1 http://localhost:${PORT}/health >/dev/null 2>&1; then
    echo "Starting free-claude-code server in background..."
    ( cd "${INSTALL_DIR}" && \\
        exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port ${PORT} \\
        >> "${INSTALL_DIR}/server.log" 2>&1 ) &
    _SERVER_PID=\$!
    _SERVER_STARTED=1

    for i in \$(seq 1 15); do
        sleep 1
        if curl -s --max-time 1 http://localhost:${PORT}/health >/dev/null 2>&1; then
            echo "Server ready."
            break
        fi
        if [ "\$i" -eq 15 ]; then
            echo "Warning: server may not be ready yet, proceeding anyway..."
        fi
    done
fi

_cleanup() {
    if [ "\$_SERVER_STARTED" -eq 1 ] && kill -0 "\$_SERVER_PID" 2>/dev/null; then
        echo "Stopping free-claude-code server..."
        kill "\$_SERVER_PID" 2>/dev/null
    fi
}
trap _cleanup EXIT INT TERM

ANTHROPIC_AUTH_TOKEN=freecc ANTHROPIC_BASE_URL=http://localhost:${PORT} claude "\$@"
SCRIPT
chmod +x "$BIN_DIR/claude-free"

success "Scripts installed to $BIN_DIR"

# ── 6. PATH ─────────────────────────────────────────────────────────────────────
add_to_path() {
    local rc_file="$1"
    local line='export PATH="$HOME/.local/bin:$PATH"'
    if [ -f "$rc_file" ] && grep -q '\.local/bin' "$rc_file" 2>/dev/null; then
        return 0
    fi
    if [ -f "$rc_file" ] || [ "${2:-}" = "create" ]; then
        echo -e "\n# free-claude-code\n$line" >> "$rc_file"
        echo "$rc_file"
    fi
}

PATCHED=""
[ -n "$(add_to_path "$HOME/.zshrc")"       ] && PATCHED="$HOME/.zshrc"
[ -n "$(add_to_path "$HOME/.bashrc")"      ] && PATCHED="${PATCHED:+$PATCHED, }$HOME/.bashrc"
[ -n "$(add_to_path "$HOME/.bash_profile")"  ] && PATCHED="${PATCHED:+$PATCHED, }$HOME/.bash_profile"

if [ -n "$PATCHED" ]; then
    success "Added ~/.local/bin to PATH in: $PATCHED"
else
    info "~/.local/bin already in PATH."
fi

# make commands available in the current session
export PATH="$BIN_DIR:$PATH"

# ── 7. summary ──────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${GREEN}${BOLD} Installation complete!${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo
echo -e "  ${BOLD}Next step:${NC} edit your API key in"
echo -e "    ${CYAN}$INSTALL_DIR/.env${NC}"
echo
echo -e "  ${BOLD}Commands:${NC}"
echo -e "    ${CYAN}claude-free${NC}         — start Claude Code (auto-starts server)"
echo -e "    ${CYAN}claude-free-server${NC}  — run the proxy server in the foreground"
echo
if [ -n "$PATCHED" ]; then
    echo -e "  ${YELLOW}Reload your shell or run:${NC}  source ~/.zshrc"
    echo
fi
