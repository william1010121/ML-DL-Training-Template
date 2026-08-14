"""Runpod SSH endpoint discovery and checksummed file transport.

This module is opt-in.  The training lifecycle does not import it, which keeps
``mltrain`` portable when Runpod is not used.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import re
import select
import shlex
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

API_ROOT = "https://api.runpod.io"
DEFAULT_PROXY_TRANSFER_LIMIT = 512 * 1024 * 1024
POD_ID = re.compile(r"^[a-z0-9]{6,64}$")
ENDPOINT_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")
REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RunpodTransportError(RuntimeError):
    """A Runpod API, SSH, transfer, or integrity failure."""


@dataclass(frozen=True)
class SshEndpoint:
    """One exact SSH endpoint returned by Runpod's Pod API."""

    mode: Literal["direct", "proxy"]
    host: str
    port: int
    username: str


@dataclass(frozen=True)
class RemoteResult:
    """Captured result from a remote shell script."""

    endpoint: SshEndpoint
    stdout: str
    returncode: int


def resolve_api_key(
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> str:
    """Resolve a Runpod key without recording or printing it."""

    values = os.environ if environ is None else environ
    from_environment = values.get("RUNPOD_API_KEY", "").strip()
    if from_environment:
        return from_environment

    path = config_path or Path.home() / ".runpod/config.toml"
    try:
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RunpodTransportError(
            "RUNPOD_API_KEY is unset and ~/.runpod/config.toml is unavailable"
        ) from error
    candidates: list[object] = [config.get("apikey")]
    default = config.get("default")
    if isinstance(default, Mapping):
        candidates.append(default.get("api_key"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise RunpodTransportError("Runpod API key is not configured")


def fetch_pod(
    pod_id: str,
    *,
    api_key: str | None = None,
    api_root: str = API_ROOT,
    timeout: float = 20,
) -> dict[str, Any]:
    """Fetch a Pod from the v2 API, which exposes direct and proxy SSH endpoints."""

    if not POD_ID.fullmatch(pod_id):
        raise ValueError("invalid Runpod pod id")
    key = api_key or resolve_api_key()
    request = urllib.request.Request(
        f"{api_root.rstrip('/')}/v2/pods/{pod_id}",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        raise RunpodTransportError(f"Runpod Pod API returned HTTP {error.code}") from error
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        raise RunpodTransportError("Runpod Pod API request failed") from error
    if not isinstance(value, dict) or value.get("id") != pod_id:
        raise RunpodTransportError("Runpod Pod API returned an unexpected payload")
    return value


def _endpoint(mode: Literal["direct", "proxy"], value: object) -> SshEndpoint | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RunpodTransportError(f"Runpod ssh.{mode} must be an object or null")
    host, port, username = value.get("host"), value.get("port"), value.get("username")
    if not isinstance(host, str) or not host or any(character.isspace() for character in host):
        raise RunpodTransportError(f"Runpod ssh.{mode}.host is invalid")
    if mode == "proxy" and host != "ssh.runpod.io":
        raise RunpodTransportError("Runpod proxy host must be ssh.runpod.io")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise RunpodTransportError(f"Runpod ssh.{mode}.port is invalid")
    if not isinstance(username, str) or not ENDPOINT_TOKEN.fullmatch(username):
        raise RunpodTransportError(f"Runpod ssh.{mode}.username is invalid")
    return SshEndpoint(mode=mode, host=host, port=port, username=username)


def ssh_endpoints(pod: Mapping[str, Any]) -> tuple[SshEndpoint, ...]:
    """Return direct-first candidates without waiting for a public IP."""

    ssh = pod.get("ssh")
    if ssh is None:
        return ()
    if not isinstance(ssh, Mapping):
        raise RunpodTransportError("Runpod ssh details must be an object")
    candidates = (
        _endpoint("direct", ssh.get("direct")),
        _endpoint("proxy", ssh.get("proxy")),
    )
    return tuple(candidate for candidate in candidates if candidate is not None)


def wait_for_ssh(
    pod_id: str,
    *,
    api_key: str | None = None,
    timeout: float = 300,
    interval: float = 3,
) -> tuple[SshEndpoint, ...]:
    """Poll until either direct or Basic SSH is available."""

    deadline = time.monotonic() + timeout
    while True:
        pod = fetch_pod(pod_id, api_key=api_key)
        endpoints = ssh_endpoints(pod)
        if endpoints:
            return endpoints
        if time.monotonic() >= deadline:
            raise RunpodTransportError("timed out waiting for a Runpod SSH endpoint")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def _identity_arguments(identity: Path | None) -> list[str]:
    if identity is None:
        return []
    path = identity.expanduser()
    if not path.is_file():
        raise RunpodTransportError(f"SSH identity is not a regular file: {path}")
    return ["-i", str(path)]


def ssh_argv(
    endpoint: SshEndpoint,
    *,
    identity: Path | None = None,
    tty: bool | None = None,
) -> list[str]:
    """Build a non-interactive, bounded SSH invocation."""

    allocate_tty = endpoint.mode == "proxy" if tty is None else tty
    command = [
        "ssh",
        "-p",
        str(endpoint.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        *_identity_arguments(identity),
    ]
    command.append("-tt" if allocate_tty else "-T")
    command.append(f"{endpoint.username}@{endpoint.host}")
    return command


def _remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not REMOTE_PATH.fullmatch(value)
        or not path.is_absolute()
        or value == "/"
        or ".." in path.parts
    ):
        raise ValueError("remote path must be a simple absolute POSIX path")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _LineReader:
    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.buffer = bytearray()

    def readline(self, deadline: float) -> bytes | None:
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[: newline + 1])
                del self.buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RunpodTransportError("timed out reading from Runpod SSH proxy")
            readable, _, _ = select.select([self.stream.fileno()], [], [], remaining)
            if not readable:
                raise RunpodTransportError("timed out reading from Runpod SSH proxy")
            chunk = os.read(self.stream.fileno(), 65536)
            if not chunk:
                if self.buffer:
                    line = bytes(self.buffer)
                    self.buffer.clear()
                    return line
                return None
            self.buffer.extend(chunk)


class _ProxySession:
    def __init__(self, endpoint: SshEndpoint, identity: Path | None, timeout: float) -> None:
        if endpoint.mode != "proxy":
            raise ValueError("proxy session requires a proxy endpoint")
        self.endpoint = endpoint
        self.timeout = timeout
        self.deadline = time.monotonic() + timeout
        self.token = uuid.uuid4().hex
        self.process = subprocess.Popen(
            ssh_argv(endpoint, identity=identity, tty=True),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.abort()
            raise RunpodTransportError("could not open Runpod SSH proxy pipes")
        self.stdin = self.process.stdin
        self.reader = _LineReader(self.process.stdout)
        ready = f"__MLTRAIN_READY_{self.token}__"
        self.write(f"stty -echo\nprintf '%s\\n' {shlex.quote(ready)}\n".encode())
        self._read_until(ready)

    def write(self, value: bytes) -> None:
        try:
            self.stdin.write(value)
            self.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.abort()
            raise RunpodTransportError("Runpod SSH proxy closed its input") from error

    def line(self) -> str | None:
        raw = self.reader.readline(self.deadline)
        return None if raw is None else raw.decode("utf-8", errors="replace").rstrip("\r\n")

    def _read_until(self, marker: str, *, limit: int = 1024 * 1024) -> list[str]:
        output: list[str] = []
        size = 0
        while True:
            line = self.line()
            if line is None:
                self.abort()
                raise RunpodTransportError("Runpod SSH proxy closed before its marker")
            if line == marker:
                return output
            size += len(line)
            if size > limit:
                self.abort()
                raise RunpodTransportError("Runpod SSH proxy output exceeded its limit")
            output.append(line)

    def finish(self, done: str) -> tuple[int, list[str]]:
        output: list[str] = []
        returncode: int | None = None
        prefix = f"{done}:"
        while True:
            line = self.line()
            if line is None:
                break
            if line.startswith(prefix) and line[len(prefix) :].isdigit():
                returncode = int(line[len(prefix) :])
                continue
            output.append(line)
        try:
            process_code = self.process.wait(timeout=max(0.1, self.deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            self.abort()
            raise RunpodTransportError("Runpod SSH proxy did not exit") from error
        if returncode is None:
            raise RunpodTransportError("Runpod SSH proxy returned no completion marker")
        if process_code not in {0, returncode} and returncode == 0:
            raise RunpodTransportError(f"Runpod SSH proxy exited with status {process_code}")
        return returncode, output

    def close_input(self) -> None:
        with contextlib.suppress(OSError):
            self.stdin.close()

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _proxy_script(
    endpoint: SshEndpoint,
    script: str,
    *,
    identity: Path | None,
    timeout: float,
) -> RemoteResult:
    session = _ProxySession(endpoint, identity, timeout)
    delimiter = f"__MLTRAIN_SCRIPT_{session.token}__"
    done = f"__MLTRAIN_DONE_{session.token}__"
    try:
        payload = (
            f"bash -s <<'{delimiter}'\n{script.rstrip()}\n{delimiter}\n"
            f"rc=$?\nprintf '%s:%s\\n' {shlex.quote(done)} \"$rc\"\nexit \"$rc\"\n"
        )
        session.write(payload.encode())
        session.close_input()
        returncode, output = session.finish(done)
    except OSError as error:
        session.abort()
        raise RunpodTransportError("Runpod proxy upload stream failed") from error
    except BaseException:
        session.abort()
        raise
    return RemoteResult(endpoint=endpoint, stdout="\n".join(output), returncode=returncode)


def run_remote(
    endpoint: SshEndpoint,
    script: str,
    *,
    identity: Path | None = None,
    timeout: float = 120,
) -> RemoteResult:
    """Run a shell script through direct SSH or the interactive Basic SSH proxy."""

    if "\x00" in script:
        raise ValueError("remote script contains NUL")
    if endpoint.mode == "proxy":
        result = _proxy_script(endpoint, script, identity=identity, timeout=timeout)
    else:
        try:
            process = subprocess.run(
                [*ssh_argv(endpoint, identity=identity, tty=False), "bash", "-s"],
                input=script,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RunpodTransportError("direct Runpod SSH command failed to start") from error
        result = RemoteResult(
            endpoint=endpoint,
            stdout=(process.stdout + process.stderr).strip(),
            returncode=process.returncode,
        )
    if result.returncode != 0:
        detail = result.stdout[-2000:]
        raise RunpodTransportError(
            f"Runpod {endpoint.mode} SSH command failed with status {result.returncode}: {detail}"
        )
    return result


def _rsync_transport(endpoint: SshEndpoint, identity: Path | None) -> str:
    if endpoint.mode != "direct":
        raise ValueError("rsync requires direct SSH")
    command = ssh_argv(endpoint, identity=identity, tty=False)
    return shlex.join(command[:-2])


def _direct_upload(
    endpoint: SshEndpoint,
    source: Path,
    remote: str,
    digest: str,
    *,
    identity: Path | None,
    timeout: float,
) -> None:
    temporary = f"{remote}.mltrain-{digest}.part"
    destination = f"{endpoint.username}@{endpoint.host}:{temporary}"
    command = [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--protect-args",
        "-e",
        _rsync_transport(endpoint, identity),
        str(source),
        destination,
    ]
    try:
        process = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RunpodTransportError("direct Runpod rsync upload failed to start") from error
    if process.returncode:
        raise RunpodTransportError(
            f"direct Runpod rsync upload failed: {(process.stderr or process.stdout).strip()}"
        )
    quoted_remote = shlex.quote(remote)
    quoted_temporary = shlex.quote(temporary)
    run_remote(
        endpoint,
        "\n".join(
            [
                "set -eu",
                f"test ! -L {quoted_remote}",
                f"actual=$(sha256sum {quoted_temporary} | awk '{{print $1}}')",
                f"test \"$actual\" = {shlex.quote(digest)}",
                f"if test -f {quoted_remote}; then",
                f"  current=$(sha256sum {quoted_remote} | awk '{{print $1}}')",
                f"  test \"$current\" = {shlex.quote(digest)}",
                f"  rm -f {quoted_temporary}",
                "else",
                f"  mv {quoted_temporary} {quoted_remote}",
                "fi",
            ]
        ),
        identity=identity,
        timeout=timeout,
    )


def _proxy_upload(
    endpoint: SshEndpoint,
    source: Path,
    remote: str,
    digest: str,
    *,
    identity: Path | None,
    timeout: float,
) -> None:
    session = _ProxySession(endpoint, identity, timeout)
    script_delimiter = f"__MLTRAIN_SCRIPT_{session.token}__"
    data_delimiter = f"__MLTRAIN_DATA_{session.token}__"
    done = f"__MLTRAIN_DONE_{session.token}__"
    temporary = f"{remote}.mltrain-{digest}.part"
    parent = str(PurePosixPath(remote).parent)
    try:
        header = "\n".join(
            [
                f"bash -s <<'{script_delimiter}'",
                "set -eu",
                f"mkdir -p {shlex.quote(parent)}",
                f"test ! -L {shlex.quote(remote)}",
                f"if test -f {shlex.quote(remote)}; then",
                f"  current=$(sha256sum {shlex.quote(remote)} | awk '{{print $1}}')",
                f"  test \"$current\" = {shlex.quote(digest)}",
                "  exit 0",
                "fi",
                f"test ! -L {shlex.quote(temporary)}",
                f"base64 -d > {shlex.quote(temporary)} <<'{data_delimiter}'",
                "",
            ]
        ).encode()
        session.write(header)
        with source.open("rb") as stream:
            base64.encode(stream, session.stdin)
        footer = "\n".join(
            [
                data_delimiter,
                f"actual=$(sha256sum {shlex.quote(temporary)} | awk '{{print $1}}')",
                f"test \"$actual\" = {shlex.quote(digest)}",
                f"mv {shlex.quote(temporary)} {shlex.quote(remote)}",
                script_delimiter,
                "rc=$?",
                f"printf '%s:%s\\n' {shlex.quote(done)} \"$rc\"",
                "exit \"$rc\"",
                "",
            ]
        )
        session.write(footer.encode())
        session.close_input()
        returncode, output = session.finish(done)
    except OSError as error:
        session.abort()
        raise RunpodTransportError("Runpod proxy download stream failed") from error
    except BaseException:
        session.abort()
        raise
    if returncode:
        raise RunpodTransportError(
            f"Runpod proxy upload failed with status {returncode}: {' '.join(output[-20:])}"
        )


def upload_file(
    endpoints: Sequence[SshEndpoint],
    source: Path,
    remote: str,
    *,
    identity: Path | None = None,
    timeout: float = 900,
    max_proxy_bytes: int = DEFAULT_PROXY_TRANSFER_LIMIT,
) -> SshEndpoint:
    """Upload atomically, preferring resumable direct rsync and falling back to proxy."""

    source = source.expanduser()
    remote = _remote_path(remote)
    if not source.is_file() or source.is_symlink():
        raise ValueError("upload source must be a regular non-symlink file")
    source = source.resolve()
    digest = _sha256(source)
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            if endpoint.mode == "direct":
                _direct_upload(
                    endpoint, source, remote, digest, identity=identity, timeout=timeout
                )
            else:
                if source.stat().st_size > max_proxy_bytes:
                    raise RunpodTransportError(
                        f"file exceeds proxy transfer limit of {max_proxy_bytes} bytes"
                    )
                _proxy_upload(
                    endpoint, source, remote, digest, identity=identity, timeout=timeout
                )
            return endpoint
        except RunpodTransportError as error:
            errors.append(f"{endpoint.mode}: {error}")
    raise RunpodTransportError("all Runpod upload transports failed: " + "; ".join(errors))


def _metadata_script(remote: str, marker: str) -> str:
    quoted = shlex.quote(remote)
    return "\n".join(
        [
            "set -eu",
            f"test -f {quoted}",
            f"test ! -L {quoted}",
            f"digest=$(sha256sum {quoted} | awk '{{print $1}}')",
            f"size=$(stat -c %s {quoted})",
            f"printf '%s:%s:%s\\n' {shlex.quote(marker)} \"$digest\" \"$size\"",
        ]
    )


def _parse_metadata(line: str, marker: str, max_bytes: int) -> tuple[str, int]:
    prefix = f"{marker}:"
    if not line.startswith(prefix):
        raise RunpodTransportError("remote file metadata marker is missing")
    fields = line[len(prefix) :].split(":")
    if len(fields) != 2 or not SHA256.fullmatch(fields[0]) or not fields[1].isdigit():
        raise RunpodTransportError("remote file metadata is invalid")
    size = int(fields[1])
    if size > max_bytes:
        raise RunpodTransportError(f"remote file exceeds transfer limit of {max_bytes} bytes")
    return fields[0], size


def _finish_download(temporary: Path, destination: Path, digest: str, size: int) -> None:
    if temporary.stat().st_size != size or _sha256(temporary) != digest:
        raise RunpodTransportError("downloaded Runpod artifact failed SHA-256 verification")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file() or _sha256(destination) != digest:
            raise FileExistsError(
                f"refusing to overwrite different artifact: {destination}"
            ) from None
    temporary.unlink()


def _direct_download(
    endpoint: SshEndpoint,
    remote: str,
    destination: Path,
    *,
    identity: Path | None,
    timeout: float,
    max_bytes: int,
) -> None:
    marker = f"__MLTRAIN_META_{uuid.uuid4().hex}__"
    result = run_remote(
        endpoint,
        _metadata_script(remote, marker),
        identity=identity,
        timeout=timeout,
    )
    metadata = next((line for line in result.stdout.splitlines() if line.startswith(marker)), "")
    digest, size = _parse_metadata(metadata, marker, max_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        command = [
            "rsync",
            "-a",
            "--partial",
            "--append-verify",
            "--protect-args",
            "-e",
            _rsync_transport(endpoint, identity),
            f"{endpoint.username}@{endpoint.host}:{remote}",
            str(temporary),
        ]
        process = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        if process.returncode:
            raise RunpodTransportError(
                f"direct Runpod rsync download failed: {(process.stderr or process.stdout).strip()}"
            )
        _finish_download(temporary, destination, digest, size)
    finally:
        temporary.unlink(missing_ok=True)


def _proxy_download(
    endpoint: SshEndpoint,
    remote: str,
    destination: Path,
    *,
    identity: Path | None,
    timeout: float,
    max_bytes: int,
) -> None:
    session = _ProxySession(endpoint, identity, timeout)
    script_delimiter = f"__MLTRAIN_SCRIPT_{session.token}__"
    metadata_marker = f"__MLTRAIN_META_{session.token}__"
    data_end = f"__MLTRAIN_DATA_END_{session.token}__"
    done = f"__MLTRAIN_DONE_{session.token}__"
    quoted = shlex.quote(remote)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        script = "\n".join(
            [
                f"bash -s <<'{script_delimiter}'",
                "set -eu",
                f"test -f {quoted}",
                f"test ! -L {quoted}",
                f"digest=$(sha256sum {quoted} | awk '{{print $1}}')",
                f"size=$(stat -c %s {quoted})",
                f"printf '%s:%s:%s\\n' {shlex.quote(metadata_marker)} \"$digest\" \"$size\"",
                f"base64 {quoted}",
                f"printf '%s\\n' {shlex.quote(data_end)}",
                script_delimiter,
                "rc=$?",
                f"printf '%s:%s\\n' {shlex.quote(done)} \"$rc\"",
                "exit \"$rc\"",
                "",
            ]
        )
        session.write(script.encode())
        session.close_input()
        metadata_line = session.line()
        while metadata_line is not None and not metadata_line.startswith(metadata_marker):
            metadata_line = session.line()
        if metadata_line is None:
            raise RunpodTransportError("Runpod proxy returned no file metadata")
        digest, size = _parse_metadata(metadata_line, metadata_marker, max_bytes)
        with temporary.open("wb") as stream:
            while True:
                line = session.line()
                if line is None:
                    raise RunpodTransportError("Runpod proxy closed during download")
                if line == data_end:
                    break
                try:
                    stream.write(base64.b64decode(line, validate=True))
                except ValueError as error:
                    raise RunpodTransportError("Runpod proxy returned invalid base64") from error
        returncode, output = session.finish(done)
        if returncode:
            raise RunpodTransportError(
                f"Runpod proxy download failed with status {returncode}: {' '.join(output[-20:])}"
            )
        _finish_download(temporary, destination, digest, size)
    except BaseException:
        session.abort()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def download_file(
    endpoints: Sequence[SshEndpoint],
    remote: str,
    destination: Path,
    *,
    identity: Path | None = None,
    timeout: float = 900,
    max_bytes: int = DEFAULT_PROXY_TRANSFER_LIMIT,
) -> SshEndpoint:
    """Download and verify, preferring direct rsync and falling back to Basic SSH."""

    remote = _remote_path(remote)
    destination = destination.expanduser().resolve()
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            if endpoint.mode == "direct":
                _direct_download(
                    endpoint,
                    remote,
                    destination,
                    identity=identity,
                    timeout=timeout,
                    max_bytes=max_bytes,
                )
            else:
                _proxy_download(
                    endpoint,
                    remote,
                    destination,
                    identity=identity,
                    timeout=timeout,
                    max_bytes=max_bytes,
                )
            return endpoint
        except (RunpodTransportError, OSError) as error:
            errors.append(f"{endpoint.mode}: {error}")
    raise RunpodTransportError("all Runpod download transports failed: " + "; ".join(errors))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mltrain-runpod", description=__doc__)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=300)
    commands = parser.add_subparsers(dest="command", required=True)
    endpoint = commands.add_parser("endpoint", help="show ordered SSH endpoint candidates")
    endpoint.add_argument("pod_id")
    execute = commands.add_parser("exec", help="run a command through direct or Basic SSH")
    execute.add_argument("pod_id")
    execute.add_argument("arguments", nargs=argparse.REMAINDER)
    upload = commands.add_parser("upload", help="upload a checksummed regular file")
    upload.add_argument("pod_id")
    upload.add_argument("source", type=Path)
    upload.add_argument("remote")
    download = commands.add_parser("download", help="download a checksummed regular file")
    download.add_argument("pod_id")
    download.add_argument("remote")
    download.add_argument("destination", type=Path)
    return parser


def _first_working_remote(
    endpoints: Sequence[SshEndpoint],
    script: str,
    *,
    identity: Path | None,
) -> RemoteResult:
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            return run_remote(endpoint, script, identity=identity)
        except RunpodTransportError as error:
            errors.append(f"{endpoint.mode}: {error}")
    raise RunpodTransportError("all Runpod SSH transports failed: " + "; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        endpoints = wait_for_ssh(args.pod_id, timeout=args.wait_seconds)
        if args.command == "endpoint":
            print(json.dumps([asdict(endpoint) for endpoint in endpoints], indent=2))
        elif args.command == "exec":
            arguments = list(args.arguments)
            if arguments and arguments[0] == "--":
                arguments.pop(0)
            if not arguments:
                raise ValueError("exec requires a command after --")
            result = _first_working_remote(
                endpoints, "exec " + shlex.join(arguments), identity=args.identity
            )
            if result.stdout:
                print(result.stdout)
        elif args.command == "upload":
            endpoint = upload_file(
                endpoints, args.source, args.remote, identity=args.identity
            )
            print(json.dumps({"transport": endpoint.mode, "remote": args.remote}))
        elif args.command == "download":
            endpoint = download_file(
                endpoints, args.remote, args.destination, identity=args.identity
            )
            print(json.dumps({"transport": endpoint.mode, "destination": str(args.destination)}))
    except (FileExistsError, OSError, RunpodTransportError, ValueError) as error:
        raise SystemExit(f"mltrain-runpod: error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
