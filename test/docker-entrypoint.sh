#!/usr/bin/env bash
# docker-entrypoint.sh — post-install verification suite
set -euo pipefail

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}${BOLD}PASS${NC}  $*"; (( PASS++ )) || true; }
fail() { echo -e "  ${RED}${BOLD}FAIL${NC}  $*"; (( FAIL++ )) || true; }
section() { echo -e "\n${BOLD}$*${NC}"; }

INSTALL_DIR="/opt/free-claude-code"
BIN_DIR="$HOME/.local/bin"
PORT=8082

echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   free-claude-code install verification  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

# ── 1. files ────────────────────────────────────────────────────────────────────
section "1. Installed files"

[ -f "$INSTALL_DIR/server.py" ]         && ok "server.py present"           || fail "server.py missing"
[ -f "$INSTALL_DIR/pyproject.toml" ]    && ok "pyproject.toml present"      || fail "pyproject.toml missing"
[ -f "$INSTALL_DIR/.env" ]              && ok ".env created"                 || fail ".env missing"
[ -f "$INSTALL_DIR/.env.example" ]      && ok ".env.example present"        || fail ".env.example missing"

# ── 2. scripts ──────────────────────────────────────────────────────────────────
section "2. CLI scripts"

[ -f "$BIN_DIR/claude-free" ]           && ok "claude-free exists"          || fail "claude-free missing"
[ -f "$BIN_DIR/claude-free-server" ]    && ok "claude-free-server exists"   || fail "claude-free-server missing"
[ -x "$BIN_DIR/claude-free" ]           && ok "claude-free is executable"   || fail "claude-free not executable"
[ -x "$BIN_DIR/claude-free-server" ]    && ok "claude-free-server is executable" || fail "claude-free-server not executable"

# scripts reference correct install dir
grep -q "$INSTALL_DIR" "$BIN_DIR/claude-free"        && ok "claude-free references correct INSTALL_DIR"        || fail "claude-free has wrong path"
grep -q "$INSTALL_DIR" "$BIN_DIR/claude-free-server" && ok "claude-free-server references correct INSTALL_DIR" || fail "claude-free-server has wrong path"

# scripts reference correct port
grep -q "$PORT" "$BIN_DIR/claude-free"        && ok "claude-free references port $PORT"        || fail "claude-free wrong port"
grep -q "$PORT" "$BIN_DIR/claude-free-server" && ok "claude-free-server references port $PORT" || fail "claude-free-server wrong port"

# auto-start logic present
grep -q "_SERVER_STARTED" "$BIN_DIR/claude-free"  && ok "auto-start logic present"  || fail "auto-start logic missing"
grep -q "_cleanup"         "$BIN_DIR/claude-free"  && ok "auto-stop trap present"    || fail "auto-stop trap missing"

# ── 3. venv ─────────────────────────────────────────────────────────────────────
section "3. Python virtual environment"

[ -d "$INSTALL_DIR/.venv" ]                       && ok ".venv directory exists"         || fail ".venv missing"
[ -f "$INSTALL_DIR/.venv/bin/python" ]            && ok ".venv/bin/python exists"        || fail ".venv/bin/python missing"
[ -f "$INSTALL_DIR/.venv/bin/uvicorn" ]           && ok ".venv/bin/uvicorn exists"       || fail ".venv/bin/uvicorn missing"

# shebang must point to new install dir (not stale)
SHEBANG=$(head -1 "$INSTALL_DIR/.venv/bin/uvicorn")
[[ "$SHEBANG" == *"$INSTALL_DIR"* ]] \
    && ok "uvicorn shebang points to correct path ($SHEBANG)" \
    || fail "uvicorn shebang is stale: $SHEBANG"

# key python packages importable
"$INSTALL_DIR/.venv/bin/python" -c "import loguru"   && ok "loguru importable"   || fail "loguru missing"
"$INSTALL_DIR/.venv/bin/python" -c "import fastapi"  && ok "fastapi importable"  || fail "fastapi missing"
"$INSTALL_DIR/.venv/bin/python" -c "import uvicorn"  && ok "uvicorn importable"  || fail "uvicorn missing"
"$INSTALL_DIR/.venv/bin/python" -c "import httpx"    && ok "httpx importable"    || fail "httpx missing"

# ── 4. server startup ───────────────────────────────────────────────────────────
section "4. Server startup & health"

# Use lmstudio provider so no API key is required for server to boot
cat > "$INSTALL_DIR/.env" <<EOF
PROVIDER_TYPE=lmstudio
MODEL=lmstudio/test-model
LM_STUDIO_BASE_URL=http://localhost:1234/v1
MESSAGING_PLATFORM=discord
DISCORD_BOT_TOKEN=
ALLOWED_DISCORD_CHANNELS=
CLAUDE_WORKSPACE=./agent_workspace
MAX_CLI_SESSIONS=10
EOF

# Start server in background
cd "$INSTALL_DIR"
.venv/bin/uvicorn server:app --host 0.0.0.0 --port $PORT >> server.log 2>&1 &
SERVER_PID=$!

# Wait up to 15s for /health
HEALTHY=0
for i in $(seq 1 15); do
    sleep 1
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        HEALTHY=1; break
    fi
done

if [ $HEALTHY -eq 1 ]; then
    ok "Server started (PID $SERVER_PID)"

    # /health returns correct JSON
    RESP=$(curl -sf "http://localhost:$PORT/health")
    [[ "$RESP" == *"healthy"* ]] \
        && ok "/health returns {status: healthy}" \
        || fail "/health unexpected response: $RESP"

    # / root endpoint accessible
    curl -sf "http://localhost:$PORT/" >/dev/null 2>&1 \
        && ok "/ root endpoint responds" \
        || fail "/ root endpoint unreachable"
else
    fail "Server did not start within 15s"
    echo "--- server.log ---"
    cat "$INSTALL_DIR/server.log" | tail -30
fi

kill $SERVER_PID 2>/dev/null || true
sleep 1
kill -9 $SERVER_PID 2>/dev/null || true
# Wait for port to be fully released before auto-start test
for i in $(seq 1 15); do
    sleep 1
    if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
done

# ── 5. auto-start check (claude-free without server) ───────────────────────────
section "5. claude-free auto-start logic"

# Verify server is down
sleep 1
if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    warn "Server still up, skipping auto-start test"
else
    # Run claude-free in background; it should start the server then call claude
    # We stub 'claude' to exit immediately so the test doesn't hang
    mkdir -p /tmp/stub-bin
    printf '#!/bin/sh\nexit 0\n' > /tmp/stub-bin/claude && chmod +x /tmp/stub-bin/claude

    OUTPUT=$(PATH="/tmp/stub-bin:$BIN_DIR:$PATH" timeout 35 bash "$BIN_DIR/claude-free" --version 2>&1 || true)
    echo "$OUTPUT" | grep -q "Starting free-claude-code server" \
        && ok "claude-free auto-starts server when not running" \
        || { fail "claude-free did not print startup message"; echo "    output was: $(echo "$OUTPUT" | head -3)"; }

    # Server should have been killed after claude-free exited
    sleep 2
    if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        ok "Server auto-stopped after claude-free exited"
    else
        fail "Server still running after claude-free exited"
        # kill server using ss (lsof not available in minimal containers)
        ss -tlnp "sport = :$PORT" 2>/dev/null | awk 'NR>1 {print $6}' | \
            grep -oP 'pid=\K[0-9]+' | xargs -r kill 2>/dev/null || true
    fi
fi

# ── summary ─────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}══════════════════════════════════════════${NC}"
if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}ALL $PASS TESTS PASSED${NC}"
else
    echo -e "  ${GREEN}${BOLD}$PASS passed${NC}  ${RED}${BOLD}$FAIL failed${NC}"
fi
echo -e "${BOLD}══════════════════════════════════════════${NC}\n"

exit $FAIL
