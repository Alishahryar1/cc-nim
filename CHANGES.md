# Changelog — Luiz-crypto-cmd fork

## Fix: AVG HTTPS-scanning breaks DeepSeek streams — "empty or malformed response" (2026-07-22)

### Problem

Claude Code suddenly started failing with:

```
API Error: API returned an empty or malformed response (HTTP 200) — check for a proxy or gateway intercepting the request
```

Symptoms:
- Small/fast requests mostly worked; large conversations (200+ messages) reliably stalled.
- A stalled request produced **zero log output** in `server.log` — no error, no exception —
  the client just retried the same message count ~5 minutes later (matching `HTTP_READ_TIMEOUT`).
- Restarting the proxy made it worse: freshly-spawned `python.exe` crashed on startup with only
  `OPENSSL_Uplink(...): no OPENSSL_Applink` on stderr and no Python traceback at all — even
  though the *already-running* process had been serving requests fine for hours.

### Root cause

**AVG Antivirus's HTTPS Scanning (Web Shield) does full TLS interception (MITM) on this
machine**, including on connections to `api.deepseek.com`. Two distinct failures stack on top
of each other:

1. **`SSLKEYLOGFILE` crash.** AVG sets this env var system-wide to a named pipe
   (`\\.\avgMonFltProxy\<id>`) so its driver can log TLS session keys. OpenSSL's Uplink
   mechanism tries to open that path as a regular file during process init and aborts
   immediately — before Python's own exception machinery ever gets control (no traceback,
   exit code 1). The long-running process that was already up had been started before this
   condition applied (or before AVG re-armed it), so it never hit this path — only *fresh*
   `python.exe` starts crash.
2. **Malformed MITM root certificate.** AVG's interception CA
   (`C:\ProgramData\AVG\Antivirus\wscert.pem`) has its `Basic Constraints` extension marked
   non-critical, which is out of spec (RFC 5280). Windows' own certificate chain building is
   lenient and accepts it (so `curl`/Node — via `NODE_EXTRA_CA_CERTS` — work fine), but
   **OpenSSL 3.x's `VERIFY_X509_STRICT` flag, on by default in `ssl.create_default_context()`
   since Python 3.13**, rejects it outright: `CERTIFICATE_VERIFY_FAILED`. This project pins
   Python 3.14, so every fresh TLS handshake through `httpx`'s default context is rejected.
   On a long SSE stream this doesn't always surface as a clean exception — it can just stall
   until the read times out, which is why nothing was ever logged.

Confirmed by reproducing directly: `httpx.get("https://api.deepseek.com/...")` failed with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`, and after adding AVG's CA
to `certifi`'s bundle, with `Basic Constraints of CA cert not marked critical` — i.e. rejected
for being present-but-non-compliant, not for being untrusted.

### Fix (2 places)

**`providers/anthropic_messages.py`** — `AnthropicMessagesTransport`

- Added `_build_relaxed_ssl_context()`: builds the same context httpx would build by default
  (`ssl.create_default_context(cafile=certifi.where())`) but clears
  `ssl.VERIFY_X509_STRICT`. Hostname verification and normal chain-of-trust checks stay fully
  enabled — only the Basic-Constraints-criticality nitpick is relaxed. Passed as
  `verify=` to the provider's `httpx.AsyncClient`. (AVG's CA also needs to be present in the
  trust store for this to matter — see workaround below.)
- Added an explicit `except asyncio.CancelledError:` branch in `stream_response()` before the
  general `except Exception`, since `CancelledError` is `BaseException` in Python 3.8+ and was
  silently swallowing client-disconnect/stream-cancellation events with zero log trace. This
  doesn't change behavior (still re-raises) — it just makes future stalls diagnosable instead
  of leaving a completely empty `server.log`.

**Per-machine workaround (not code, do this once):**
1. Append AVG's root cert to the venv's `certifi` bundle so it's trusted at all:
   `cat /c/ProgramData/AVG/Antivirus/wscert.pem >> .venv/Lib/site-packages/certifi/cacert.pem`
   (gets wiped on `certifi` reinstall/upgrade — reapply if that ever happens).
2. `Start-FreeCC` (PowerShell profile) now clears `SSLKEYLOGFILE` for the spawned proxy
   process before launching uvicorn, to avoid the startup crash.

**Better long-term fix:** disable AVG's "HTTPS Scanning" / "Web Shield" SSL inspection
entirely (AVG settings → Protections → Web/Email Shield). This removes both failure modes at
the source for every app on the machine, not just this proxy, and stops AVG from
man-in-the-middling your dev traffic in general.

### Affected versions

- Any machine with AVG (or likely Avast, same engine) HTTPS Scanning enabled
- Python 3.13+ (`VERIFY_X509_STRICT` default) — Python ≤3.12 masked this bug
- Any provider using `AnthropicMessagesTransport` (DeepSeek, Kimi, Z.ai, Zhipu)

## Fix: Claude Code 2.1.152 + DeepSeek thinking — multi-turn 400 error (2026-05-27)

### Problem

After Claude Code auto-updated to **2.1.152** on May 27, 2026, all multi-turn conversations with
DeepSeek (deepseek-v4-pro / deepseek-reasoner) started failing with:

```
● Invalid request sent to provider.
  Request ID: req_xxxxxxxx
```

The underlying HTTP error from DeepSeek:
```json
{"error": {"message": "The `content[].thinking` in the thinking mode must be passed back to the API.", "type": "invalid_request_error"}}
```

### Root cause

Claude Code **2.1.152** added a cryptographic signature requirement for thinking blocks:
it only stores thinking blocks in conversation history if they carry a valid `signature` field
signed by Anthropic's infrastructure.

DeepSeek's Anthropic-compatible API does **not** emit `signature_delta` events (it's not
Anthropic's real infrastructure). So:

1. Turn 1: DeepSeek generates thinking → proxy forwards it → Claude Code **discards it** (no signature)
2. Turn 2: Claude Code sends history without thinking blocks → DeepSeek requires them → **HTTP 400**

deepseek-v4-pro is an always-on reasoning model: it requires prior thinking blocks in **all**
assistant turns with tool_use, regardless of whether thinking is explicitly requested in the
current turn.

### Fix (2 files)

**`providers/deepseek/request.py`** — `_inject_placeholder_thinking_blocks()`

When the proxy detects assistant messages in history that have `tool_use` blocks but no
`thinking` block (because Claude Code dropped them), it injects a minimal placeholder:

```python
{"type": "thinking", "thinking": "(prior reasoning not available)"}
```

This satisfies DeepSeek's history validation. The model still generates full real thinking
on every new response.

**`core/anthropic/native_sse_block_policy.py`** — synthetic `signature_delta`

Injects a `signature_delta` SSE event before each `content_block_stop` for thinking blocks.
This causes Claude Code to store thinking blocks in history when the signature validation
is not strictly enforced (forward-compatibility fix).

**`config/logging_config.py`** — Windows restart fix

Replaced manual `Path.write_text("")` truncation with `mode="w"` on the loguru file sink,
fixing a `PermissionError` on Windows when restarting the gateway while the previous
process still held the log file handle.

### Affected versions

- Claude Code ≥ 2.1.152
- DeepSeek provider (deepseek-v4-pro, deepseek-reasoner, any always-on reasoning model)
- All other providers unaffected

### Upstream

Based on [free-claude-code](https://github.com/Alishahryar1/free-claude-code) by Ali Khokhar (MIT).
