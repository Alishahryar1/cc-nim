# Linux Desktop Support for `fcc-desktop`

Date: 2026-08-24
Status: Approved

## Goal

Enable the existing `fcc-desktop` tray shell on Linux with feature parity to
Windows/macOS, plus graceful degradation to a foreground console mode when no
tray backend is available.

Non-goals: Claude Desktop integration (picker aliasing, `claude_desktop_config.json`
merge) — parked as separate future work.

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
4. install.sh dry-run on Linux: `.desktop` contents, owned-file guard
5. Live smoke on real Linux desktop: tray appears or fallback engages cleanly
