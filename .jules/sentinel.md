## 2026-05-11 - Adding Security Headers
**Vulnerability:** Missing security headers (CSP, X-Frame-Options, etc.).
**Learning:** FastAPI does not include these by default. Even for local-only apps, these provide defense-in-depth against browser-based attacks.
**Prevention:** Always add a middleware to set standard security headers in FastAPI applications.

## 2026-05-11 - Secure File Permissions for Config
**Vulnerability:** Sensitive `.env` files created with default system permissions may be world-readable.
**Learning:** Default `umask` often allows other users to read files.
**Prevention:** Explicitly set permissions to `0600` for files containing secrets.

## 2026-05-12 - Rate Limiting & Denial of Service Mitigation
**Vulnerability:** Brute-force attacks on API keys or exhaustion of resources via large payloads.
**Learning:** Rate limiting is essential even for local-only interfaces to prevent misconfigured scripts or malicious local processes from crashing the server.
**Prevention:** Implemented a sliding window rate limiter middleware for both Admin and Main API paths.

## 2026-05-12 - Admin UI Input Sanitization
**Vulnerability:** Potential Self-XSS via `innerHTML` and IP spoofing via `X-Forwarded-For`.
**Learning:** Hostname validation is not enough if the server is behind a proxy that forwards headers.
**Prevention:** Blocked `X-Forwarded-For` for admin routes and replaced `innerHTML` with `textContent` in the UI logic.

## 2026-05-12 - Resource Exhaustion in CLI Subprocesses
**Vulnerability:** Excessively large prompts could crash the `claude` binary or consume all system memory.
**Learning:** Subprocess arguments have limits, and large strings can cause significant latency.
**Prevention:** Added a 120KB limit to prompts before spawning the CLI session.

## 2026-05-12 - Availability: Resilience Against Upstream Timeout Stalls
**Vulnerability:** Upstream API stalls (e.g. Nvidia NIM) can cause local server hangs, leading to resource exhaustion or denial of service for the CLI client.
**Learning:** Relying on default HTTP timeouts (120s) for AI inference is insufficient for large-context tasks. Unhandled stalls block concurrency slots.
**Prevention:** Increased global read timeouts to 600s and implemented proactive retries for timeout exceptions to ensure service availability during upstream instability.
## 2026-05-13 - Secure Initialization File Permissions
**Vulnerability:** Initial `.env` scaffolding created by `fcc-init` generates sensitive API key placeholders with default system permissions, which might be world-readable.
**Learning:** Even placeholder configuration files should be restricted immediately upon creation to prevent later added secrets from being exposed.
**Prevention:** Apply `os.chmod` to restrict file permissions to `0o600` immediately after file creation, using `contextlib.suppress(Exception)` to gracefully handle filesystems or environments where chmod fails.
