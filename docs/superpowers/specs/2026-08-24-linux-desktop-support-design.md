# Linux Desktop Support for `fcc-desktop`

Date: 2026-08-24
Status: Approved

## Goal

Enable the existing `fcc-desktop` tray shell on Linux with feature parity to
Windows/macOS, plus graceful degradation to a foreground console mode when no
tray backend is available.

Non-goal: picker aliasing — upstream's `/v1/models` already advertises canonical
Anthropic model IDs, and `ModelRouter` keyword-routes them to settings slots, so
no client-side aliases are needed.

## Claude Desktop rerouting (follow-up)

Claude Desktop is rerouted through the FCC server without a new entrypoint;
everything rides on the existing `fcc-desktop` command.

### Components

| Component | Change |
|---|---|
| `cli/claude_desktop.py` (new) | `configure`/`unconfigure` merge of `modelDiscoveryEnabled` + `inference` gateway block into `claude_desktop_config.json`; atomic write; aborts with log on malformed JSON; preserves foreign keys. Gateway URL and auth key derive from live `Settings` (`local_proxy_root_url`, `proxy_auth_token`) — never hardcoded. Also locates the binary (PATH + platform candidates) and spawns it with `--ignore-certificate-errors`. `python -m free_claude_code.cli.claude_desktop --configure|--unconfigure` for explicit runs |
| `cli/desktop.py` | `launch_desktop()` best-effort merges the routing block after preflight; failure is logged, never fatal |
| `cli/desktop_tray.py` | "Launch Claude Desktop" menu item: re-merges config then spawns the binary; tray notification when the binary is missing or spawn fails |
| `cli/desktop_console.py` | Prints a hint that Claude Desktop is auto-routed when installed |

### Failure handling

| Failure | Behavior |
|---|---|
| Config malformed | Merge skipped with warning; server continues |
| Binary not found | Tray notification; no crash |
| Spawn failure | Tray notification |

## HTTPS front proxy (Claude Desktop requires TLS)

Claude Desktop refuses plain-HTTP gateway URLs. `cli/tls_proxy.py` provides:

| Piece | Behavior |
|---|---|
| `probe_https` | Unverified-context loopback probe; **any** HTTP status (incl. 404/401) counts as "front alive" — verified against a real caddy during live smoke |
| `resolve_gateway_base_url` | `https://localhost:<TLS_PROXY_PORT>` when something answers there, else the plain HTTP root |
| `CaddyTlsProxy` | Reuses an external front (system Caddy service) without spawning; otherwise generates `~/.fcc/caddy/Caddyfile` (`admin off`, `auto_https disable_redirects`, `tls internal`, `flush_interval -1`) and runs a sandboxed `caddy run` child, stopped alongside the server |

Settings: `TLS_PROXY_ENABLED` (default true), `TLS_PROXY_PORT` (default 8443).

Caddyfile notes from live testing:
- `admin off` is mandatory — a system Caddy may own port 2019.
- `auto_https disable_redirects` (not `off`): plain `off` skips certificate
  provisioning entirely (handshake fails), while the default redirect binds
  privileged :80. Disabling just redirects keeps the internal CA working.

## Testing

## Background

Upstream `fcc-desktop` is a system-tray wrapper around the local FCC proxy
server. Architecture:

- `cli/desktop.py` — platform-neutral `DesktopController` + `launch_desktop(tray_factory)`
- `cli/desktop_tray.py` — pystray adapter (already platform-agnostic code)
- `cli/desktop_entrypoint.py` — hard gate rejects every platform except
  `darwin`/`win32`
- `pyproject.toml` — `pystray`/`Pillow` markers exclude linux

Only the entrypoint gate, dependency markers, and install script treat Linux
as unsupported.

## Design

### Runtime flow

```
fcc-desktop (linux)
└─ desktop_entrypoint.launch()
   ├─ --export-icon → unchanged
   ├─ darwin/win32 → tray directly (unchanged)
   ├─ linux        → tray_is_available() probe:
   │                  ├─ ok   → PystrayDesktopTray (unchanged behavior)
   │                  └─ fail → print reason + hint, console fallback
   └─ other platforms → unchanged "supported on Windows and macOS" error
```

### Components

| Component | Change |
|---|---|
| `desktop_tray.py` | Add `tray_is_available() -> tuple[bool, str]`: attempt minimal `Icon` construction inside try/except; return reason text naming the missing backend on failure |
| `desktop_console.py` (new) | Null tray: `run()` blocks on a `threading.Event`, `stop()` sets it. Lets the unchanged `DesktopController.run()` lifecycle drive server start/stop in tray-less mode |
| `desktop_entrypoint.py` | Replace platform gate: allow linux via probe; keep other-platform rejection |
| `pyproject.toml` | Extend `pystray`/`Pillow` markers to include linux |
| `scripts/install.sh` | Write XDG `~/.local/share/applications/free-claude-code.desktop` + export PNG icon; owned-file guard mirrors macOS bundle logic |

### Console fallback

Not a separate code path through the controller: it passes a null tray to the
same `launch_desktop()`. SIGINT raises `KeyboardInterrupt` out of
`Event.wait()`, and the controller's existing `finally` chain performs clean
shutdown.

### Error handling

| Failure | Behavior |
|---|---|
| pystray import fails | Probe returns reason "pystray not installed" |
| Icon construct fails (no AppIndicator/GTK, Wayland w/o extension) | Reason includes package hint (`gir1.2-ayatanaappindicator3-0.1`) |
| Tray dies mid-run | Unchanged controller `finally` chain stops server |
| Singleton lock held | Unchanged: open admin browser and exit |

## Testing

1. Probe: import failure / construct failure / success → correct tuples
2. Null tray: run blocks, stop releases, full controller lifecycle completes
3. Entrypoint branch table per platform × probe result
4. Config merge: missing file / idempotency / foreign-key preservation / partial
   inference overwrite / malformed abort / roundtrip unconfigure
5. Binary spawn: certificate flag forwarded, FileNotFoundError surfaced,
   configure-then-spawn ordering, `--configure` never spawns
6. Desktop integration: `launch_desktop` merges once with live settings and
   swallows failures; tray item spawns or notifies; console hint printed
7. install.sh dry-run on Linux: `.desktop` contents, owned-file guard
8. TLS proxy unit tests: probe semantics (plain-HTTP listener, error status,
   silence), Caddyfile invariants (admin off / auto_https disable_redirects /
   tls internal), lifecycle (reuse external front, spawn, early exit, stop)
9. Live smoke on real Linux desktop: tray appears or fallback engages cleanly;
   managed caddy fronts a throwaway server instance and tears down cleanly
