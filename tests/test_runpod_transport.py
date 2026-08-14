from __future__ import annotations

import json
import os
import stat
import urllib.error
from pathlib import Path

import pytest

from mltrain import runpod_transport as transport


def _pod(*, direct: bool = True, proxy: bool = True) -> dict[str, object]:
    return {
        "id": "pod123456",
        "ssh": {
            "direct": (
                {
                    "host": "192.0.2.10",
                    "port": 23022,
                    "username": "root",
                    "command": "ssh root@192.0.2.10 -p 23022",
                }
                if direct
                else None
            ),
            "proxy": (
                {
                    "host": "ssh.runpod.io",
                    "port": 22,
                    "username": "pod123456-6441103b",
                    "command": "ssh pod123456-6441103b@ssh.runpod.io",
                }
                if proxy
                else None
            ),
        },
    }


def test_endpoint_candidates_prefer_direct_but_accept_proxy_without_public_ip() -> None:
    both = transport.ssh_endpoints(_pod())
    proxy_only = transport.ssh_endpoints(_pod(direct=False))

    assert [endpoint.mode for endpoint in both] == ["direct", "proxy"]
    assert [endpoint.mode for endpoint in proxy_only] == ["proxy"]
    assert proxy_only[0].username == "pod123456-6441103b"


def test_endpoint_parser_rejects_non_runpod_proxy() -> None:
    pod = _pod(direct=False)
    assert isinstance(pod["ssh"], dict)
    pod["ssh"]["proxy"]["host"] = "attacker.example"  # type: ignore[index]

    with pytest.raises(transport.RunpodTransportError, match="proxy host"):
        transport.ssh_endpoints(pod)


def test_wait_for_ssh_returns_proxy_without_waiting_for_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fetch(pod_id: str, *, api_key: str | None = None) -> dict[str, object]:
        nonlocal calls
        del pod_id, api_key
        calls += 1
        return _pod(direct=False)

    monkeypatch.setattr(transport, "fetch_pod", fetch)

    endpoints = transport.wait_for_ssh("pod123456", api_key="test", timeout=0)

    assert calls == 1
    assert endpoints[0].mode == "proxy"


def test_api_key_prefers_environment_then_runpod_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config_value = "config" + "-secret"
    environment_value = "environment" + "-secret"
    config.write_text(f'apikey = "{config_value}"\n', encoding="utf-8")

    assert (
        transport.resolve_api_key(
            {"RUNPOD_API" + "_KEY": f" {environment_value} "}, config
        )
        == environment_value
    )
    assert transport.resolve_api_key({}, config) == config_value


def test_fetch_pod_http_error_redacts_key_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_value = "api" + "-secret"
    response_value = "response-contained" + "-secret"

    def deny(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise urllib.error.HTTPError(
            "https://api.runpod.io/v2/pods/pod123456",
            403,
            response_value,
            None,
            None,
        )

    monkeypatch.setattr(transport.urllib.request, "urlopen", deny)

    with pytest.raises(transport.RunpodTransportError) as captured:
        transport.fetch_pod("pod123456", api_key=api_value)

    message = str(captured.value)
    assert message == "Runpod Pod API returned HTTP 403"
    assert api_value not in message
    assert response_value not in message


def test_ssh_argv_allocates_tty_only_for_proxy(tmp_path: Path) -> None:
    identity = tmp_path / "key"
    identity.write_text("not-a-real-private-key", encoding="utf-8")
    direct, proxy = transport.ssh_endpoints(_pod())

    direct_command = transport.ssh_argv(direct, identity=identity)
    proxy_command = transport.ssh_argv(proxy, identity=identity)

    assert "-T" in direct_command
    assert "-tt" not in direct_command
    assert "-tt" in proxy_command
    assert proxy_command[-1] == "pod123456-6441103b@ssh.runpod.io"


def _fake_remote_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    ssh = tools / "ssh"
    ssh.write_text("#!/bin/sh\nexec bash\n", encoding="utf-8")
    ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
    stat_command = tools / "stat"
    stat_command.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -c ] && [ \"$2\" = %s ]; then\n"
        "  shift 2\n"
        "  wc -c < \"$1\" | tr -d ' '\n"
        "else\n"
        "  exec /usr/bin/stat \"$@\"\n"
        "fi\n",
        encoding="utf-8",
    )
    stat_command.chmod(stat_command.stat().st_mode | stat.S_IXUSR)
    base64_command = tools / "base64"
    base64_command.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -d ]; then\n"
        "  exec /usr/bin/base64 -d\n"
        "fi\n"
        "exec /usr/bin/base64 -i \"$1\"\n",
        encoding="utf-8",
    )
    base64_command.chmod(base64_command.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")


def test_proxy_upload_and_download_are_checksummed_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_remote_tools(tmp_path, monkeypatch)
    endpoint = transport.ssh_endpoints(_pod(direct=False))[0]
    source = tmp_path / "source.bin"
    source.write_bytes((b"proxy-transfer\x00" * 1000) + b"end")
    remote = tmp_path / "remote/archive.bin"
    destination = tmp_path / "download/archive.bin"

    used_upload = transport.upload_file([endpoint], source, str(remote), timeout=10)
    used_download = transport.download_file([endpoint], str(remote), destination, timeout=10)

    assert used_upload.mode == "proxy"
    assert used_download.mode == "proxy"
    assert remote.read_bytes() == source.read_bytes()
    assert destination.read_bytes() == source.read_bytes()

    # Retrying the same immutable object is allowed and does not rewrite history.
    transport.upload_file([endpoint], source, str(remote), timeout=10)
    transport.download_file([endpoint], str(remote), destination, timeout=10)


def test_proxy_upload_refuses_different_existing_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_remote_tools(tmp_path, monkeypatch)
    endpoint = transport.ssh_endpoints(_pod(direct=False))[0]
    source = tmp_path / "source.bin"
    source.write_bytes(b"new")
    remote = tmp_path / "remote.bin"
    remote.write_bytes(b"old")

    with pytest.raises(transport.RunpodTransportError, match="all Runpod upload"):
        transport.upload_file([endpoint], source, str(remote), timeout=10)

    assert remote.read_bytes() == b"old"


def test_upload_falls_back_from_direct_to_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    calls: list[str] = []

    def direct(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("direct")
        raise transport.RunpodTransportError("no public TCP route")

    def proxy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("proxy")

    monkeypatch.setattr(transport, "_direct_upload", direct)
    monkeypatch.setattr(transport, "_proxy_upload", proxy)

    endpoint = transport.upload_file(
        transport.ssh_endpoints(_pod()), source, "/workspace/source.bin"
    )

    assert endpoint.mode == "proxy"
    assert calls == ["direct", "proxy"]


def test_cli_endpoint_reports_both_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        transport,
        "wait_for_ssh",
        lambda *_args, **_kwargs: transport.ssh_endpoints(_pod()),
    )

    assert transport.main(["endpoint", "pod123456"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert [value["mode"] for value in output] == ["direct", "proxy"]
