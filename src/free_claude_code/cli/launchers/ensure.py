"""Install missing Muse Code / Prime Agent CLIs the first time FCC launches them."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

NO_AUTO_INSTALL_ENV = "FCC_NO_AUTO_INSTALL"
USER_AGENT = "free-claude-code"
MUSE_INSTALL_URL = "https://dev.meta.ai/install.sh"
MUSE_DOCS_URL = "https://dev.meta.ai/docs/muse-code/"
PRIME_STABLE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/latest/download/stable"
)
PRIME_RELEASE_API_URL = (
    "https://api.github.com/repos/PrimeIntellect-ai/prime-agent/releases/latest"
)
PRIME_TGZ_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/"
    "v{version}/prime-agent-{version}.tgz"
)
PRIME_SHA_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/"
    "v{version}/SHA256SUMS"
)
PRIME_DOCS_URL = "https://github.com/PrimeIntellect-ai/prime-agent"
DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class ClientSpec:
    """How to invoke an installed client, including WSL-wrapped Muse on Windows."""

    kind: str
    binary: str

    def map_path(self, path: Path) -> str:
        if self.kind == "wsl":
            return windows_path_to_wsl(path)
        return str(path)

    def build(self, args: Sequence[str]) -> list[str]:
        if self.kind == "wsl":
            return wsl_exec(self.binary, args)
        return [self.binary, *args]


def auto_install_enabled(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(NO_AUTO_INSTALL_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def extra_bin_directories() -> list[Path]:
    """Return common install locations that may not yet be on PATH."""

    directories: list[Path] = []
    home = Path.home()
    directories.append(home / ".local" / "bin")
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        directories.append(Path(appdata) / "npm")
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        directories.append(Path(local_appdata) / "pi-node" / "current")
    npm = shutil.which("npm")
    if npm:
        prefix = npm_global_prefix(npm)
        if prefix:
            prefix_path = Path(prefix)
            directories.append(prefix_path)
            directories.append(prefix_path / "bin")
    return directories


def which_command(name: str) -> str | None:
    """Resolve a command on PATH or in FCC-known install directories."""

    found = shutil.which(name)
    if found:
        return found
    suffixes = ("", ".cmd", ".exe", ".bat") if os.name == "nt" else ("",)
    for directory in extra_bin_directories():
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                prepend_path_directory(candidate.parent)
                return str(candidate)
    return None


def muse_install_hint() -> str:
    if sys.platform == "win32":
        return (
            "Muse Code has no native Windows installer. FCC can install it in WSL2 "
            f"from {MUSE_INSTALL_URL} ({MUSE_DOCS_URL})."
        )
    return f"Install Muse Code from {MUSE_DOCS_URL}"


def prime_install_hint() -> str:
    return (
        "Install Prime Agent with Node.js 20.6+ and npm from "
        f"{PRIME_DOCS_URL}, or rerun fcc-prime and let FCC install it."
    )


def ensure_muse_client() -> ClientSpec:
    """Return Muse Code, installing it when missing unless auto-install is disabled."""

    existing = find_muse_client()
    if existing is not None:
        return existing
    print("Could not find Muse Code command: muse", file=sys.stderr)
    if not auto_install_enabled():
        print(muse_install_hint(), file=sys.stderr)
        raise SystemExit(127)
    print("fcc-muse: installing Muse Code...", file=sys.stderr)
    install_muse()
    installed = find_muse_client()
    if installed is None:
        print(
            "Muse Code install finished, but 'muse' is still not available.",
            file=sys.stderr,
        )
        print(muse_install_hint(), file=sys.stderr)
        raise SystemExit(127)
    return installed


def ensure_prime_binary() -> str:
    """Return Prime Agent, installing it when missing unless auto-install is disabled."""

    existing = find_prime_binary()
    if existing is not None:
        return existing
    print("Could not find Prime Agent command: prime-agent", file=sys.stderr)
    if not auto_install_enabled():
        print(prime_install_hint(), file=sys.stderr)
        raise SystemExit(127)
    print("fcc-prime: installing Prime Agent...", file=sys.stderr)
    install_prime()
    installed = find_prime_binary()
    if installed is None:
        print(
            "Prime Agent install finished, but 'prime-agent' is still not available.",
            file=sys.stderr,
        )
        print(prime_install_hint(), file=sys.stderr)
        raise SystemExit(127)
    return installed


def find_muse_client() -> ClientSpec | None:
    native = which_command("muse")
    if native:
        return ClientSpec(kind="native", binary=native)
    if sys.platform == "win32" and wsl_has_muse():
        return ClientSpec(kind="wsl", binary="muse")
    return None


def find_prime_binary() -> str | None:
    for name in ("prime-agent", "prime"):
        found = which_command(name)
        if found:
            return found
    return None


def install_muse() -> None:
    """Install Muse Code using Meta's official installer, via WSL on Windows."""

    if sys.platform == "win32":
        wsl = shutil.which("wsl")
        if wsl is None or not wsl_is_usable(wsl):
            print(
                "Muse Code does not support native Windows. "
                "Install WSL2, then rerun fcc-muse.",
                file=sys.stderr,
            )
            print(f"Docs: {MUSE_DOCS_URL}", file=sys.stderr)
            raise SystemExit(127)
        print(
            "fcc-muse: Meta's installer is Unix-only; running it inside WSL from "
            f"{MUSE_INSTALL_URL}",
            file=sys.stderr,
        )
        run_checked(
            [
                wsl,
                "bash",
                "-lc",
                f'curl -fsSL --proto "=https" --tlsv1.2 {shlex.quote(MUSE_INSTALL_URL)} | bash',
            ]
        )
        return

    bash = shutil.which("bash")
    if bash is None:
        print("Installing Muse Code requires bash.", file=sys.stderr)
        print(muse_install_hint(), file=sys.stderr)
        raise SystemExit(127)
    print(
        f"fcc-muse: running official installer from {MUSE_INSTALL_URL}",
        file=sys.stderr,
    )
    run_posix_installer(MUSE_INSTALL_URL, bash)
    prepend_path_directory(Path.home() / ".local" / "bin")


def install_prime() -> None:
    """Install Prime Agent with npm from the latest GitHub release tarball."""

    npm = shutil.which("npm")
    if npm is None:
        print(
            "Prime Agent needs Node.js 20.6+ and npm. Install Node.js, then rerun fcc-prime.",
            file=sys.stderr,
        )
        print(prime_install_hint(), file=sys.stderr)
        raise SystemExit(127)

    version = resolve_prime_version()
    tarball_url = PRIME_TGZ_URL.format(version=version)
    print(
        f"fcc-prime: downloading Prime Agent v{version} and running npm install -g",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="fcc-prime-install-") as raw:
        directory = Path(raw)
        tarball = directory / f"prime-agent-{version}.tgz"
        download_file(tarball_url, tarball)
        expected = download_prime_tarball_sha256(version)
        if expected:
            actual = sha256_file(tarball)
            if actual != expected:
                print(
                    "Prime Agent download failed checksum verification. "
                    f"Expected {expected}, got {actual}.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        env = os.environ.copy()
        env["PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL"] = "0"
        env["PRIME_AGENT_BOOTSTRAP_TOOLS_ON_INSTALL"] = "1"
        run_checked(
            [npm, "install", "-g", "--no-fund", "--no-audit", str(tarball)],
            env=env,
        )
    prefix = npm_global_prefix(npm)
    if prefix:
        prepend_path_directory(Path(prefix))
        prepend_path_directory(Path(prefix) / "bin")


def resolve_prime_version() -> str:
    try:
        text = download_text(PRIME_STABLE_URL).strip().lstrip("v")
        if text and all(ch.isalnum() or ch in ".-" for ch in text):
            return text
    except (HTTPError, URLError, OSError, TimeoutError):
        pass
    payload = json.loads(download_text(PRIME_RELEASE_API_URL))
    tag = str(payload.get("tag_name") or "").strip().lstrip("v")
    if not tag:
        raise RuntimeError("Could not resolve the latest Prime Agent version.")
    return tag


def download_prime_tarball_sha256(version: str) -> str | None:
    try:
        body = download_text(PRIME_SHA_URL.format(version=version))
    except (HTTPError, URLError, OSError, TimeoutError):
        return None
    wanted = f"prime-agent-{version}.tgz"
    for line in body.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].endswith(wanted):
            digest = parts[0].strip().lower()
            if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest):
                return digest
    return None


def windows_path_to_wsl(path: Path) -> str:
    return windows_path_text_to_wsl(str(path.resolve()))


def windows_path_text_to_wsl(resolved: str) -> str:
    if len(resolved) >= 2 and resolved[1] == ":":
        drive = resolved[0].lower()
        rest = resolved[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return resolved.replace("\\", "/")


def wsl_exec(binary: str, args: Sequence[str]) -> list[str]:
    inner = 'export PATH="$HOME/.local/bin:$PATH"; exec ' + " ".join(
        shlex.quote(part) for part in [binary, *args]
    )
    return ["wsl", "bash", "-lc", inner]


def wsl_is_usable(wsl: str | None = None) -> bool:
    command = wsl or shutil.which("wsl")
    if command is None:
        return False
    try:
        completed = subprocess.run(
            [command, "-e", "true"],
            check=False,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def wsl_has_muse() -> bool:
    wsl = shutil.which("wsl")
    if wsl is None or not wsl_is_usable(wsl):
        return False
    try:
        completed = subprocess.run(
            [
                wsl,
                "bash",
                "-lc",
                'export PATH="$HOME/.local/bin:$PATH"; command -v muse',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def npm_global_prefix(npm: str) -> str | None:
    try:
        completed = subprocess.run(
            [npm, "prefix", "-g"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    prefix = completed.stdout.strip()
    return prefix or None


def prepend_path_directory(directory: Path) -> None:
    resolved = str(directory)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if resolved not in parts:
        os.environ["PATH"] = os.pathsep.join([resolved, *parts]) if parts else resolved


def run_posix_installer(url: str, bash: str) -> None:
    with tempfile.TemporaryDirectory(prefix="fcc-client-install-") as raw:
        script = Path(raw) / "install.sh"
        download_file(url, script)
        run_checked([bash, str(script)])


def run_checked(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), file=sys.stderr)
    completed = subprocess.run(list(command), check=False, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def download_file(url: str, dest: Path) -> None:
    dest.write_bytes(download_bytes(url))


def download_text(url: str) -> str:
    return download_bytes(url).decode("utf-8")


def download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
