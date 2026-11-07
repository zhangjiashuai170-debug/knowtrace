"""Connection lifecycle uses only mocks; no keys, inference, servers or real kills."""
from contextlib import contextmanager
from datetime import datetime
import asyncio
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess

import httpx
import pytest

from knowtrace.generation import ModelConfig
from scripts import connect_deepseek as connect

FAKE_KEY = "unit-test-only-not-a-real-deepseek-key"


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda *_: pytest.fail("No real network in connection tests"))


def config():
    return ModelConfig(provider="openai_compatible", base_url="https://api.deepseek.com",
                       model=connect.DEFAULT_MODEL, api_key=FAKE_KEY)


def passed():
    return {"status": "protocol_checks_passed", "model_inference_observed": True,
            "provider_requests_forwarded": 1, "private_case_provider_requests": 0,
            "automatic_retries": 0, "semantic_verified": False,
            "cases": [{"answer": "Retained for 14 days. [S1]"}]}


def status(directory):
    return json.loads((directory / "status.json").read_text(encoding="utf-8"))


def test_official_defaults_and_api_key_not_in_repr():
    value = connect.load_deepseek_config({"KNOWTRACE_MODEL_API_KEY": FAKE_KEY})
    assert value.provider == "openai_compatible"
    assert value.base_url == "https://api.deepseek.com"
    assert value.model == "deepseek-flash"
    assert value.api_key == FAKE_KEY and FAKE_KEY not in repr(value)


@pytest.mark.parametrize("base", ["https://api.deepseek.com/v1", "https://api.deepseek.com:443"])
def test_known_official_paths_are_allowed(base):
    assert connect.load_deepseek_config({"KNOWTRACE_MODEL_API_KEY": FAKE_KEY,
           "KNOWTRACE_MODEL_BASE_URL": base}).base_url == base


@pytest.mark.parametrize("fields", [
    {"KNOWTRACE_MODEL_BASE_URL": "https://api.deepseek.com.evil.invalid"},
    {"KNOWTRACE_MODEL_BASE_URL": "https://api.deepseek.com/other"},
    {"KNOWTRACE_MODEL_BASE_URL": "https://api.deepseek.com:444"},
    {"KNOWTRACE_MODEL_BASE_URL": "http://api.deepseek.com"},
    {"KNOWTRACE_MODEL_BASE_URL": "https://user:pass@api.deepseek.com"},
    {"KNOWTRACE_MODEL_BASE_URL": "https://api.deepseek.com?"},
    {"KNOWTRACE_MODEL_NAME": "private-key-in-model-field"},
    {"KNOWTRACE_MODEL_PROVIDER": "ollama"},
    {"KNOWTRACE_MODEL_API_KEY": ""},
    {"KNOWTRACE_MODEL_API_KEY": FAKE_KEY + "\n"},
    {"KNOWTRACE_MODEL_API_KEY": "非ASCII密钥"},
])
def test_invalid_configuration_fails_closed_without_echo(fields):
    with pytest.raises(connect.ConnectionFailure) as caught:
        connect.load_deepseek_config({"KNOWTRACE_MODEL_API_KEY": FAKE_KEY, **fields})
    assert str(caught.value) == "ConfigurationError"
    assert FAKE_KEY not in str(caught.value)


def test_real_loader_scrubs_its_own_key_environment(monkeypatch):
    for name in ("PROVIDER", "BASE_URL", "NAME", "TIMEOUT_SECONDS"):
        monkeypatch.delenv("KNOWTRACE_MODEL_" + name, raising=False)
    monkeypatch.setenv("KNOWTRACE_MODEL_API_KEY", FAKE_KEY)
    value = connect.load_deepseek_config()
    assert value.api_key == FAKE_KEY
    assert "KNOWTRACE_MODEL_API_KEY" not in os.environ


def test_failed_preflight_never_stops_existing_offline_service(tmp_path):
    calls = []
    def preflight(**kwargs):
        calls.append("preflight")
        assert kwargs["run_live"] is True and kwargs["config_loader"]().api_key == FAKE_KEY
        return {"status": "checks_failed_or_unknown", "model_inference_observed": None,
                "provider_requests_forwarded": 1, "automatic_retries": 0}
    result = connect.run_connection(directory=tmp_path, config_loader=config, preflight=preflight,
              stopper=lambda: pytest.fail("Preflight failure stopped old server"),
              service_runner=lambda *_: pytest.fail("Preflight failure activated service"))
    assert result == 2 and calls == ["preflight"]
    assert status(tmp_path)["status"] == "preflight_failed"
    assert FAKE_KEY not in (tmp_path / "preflight.json").read_text()


@pytest.mark.parametrize("change", [{"model_inference_observed": None},
                                    {"model_inference_observed": False},
                                    {"provider_requests_forwarded": 2},
                                    {"private_case_provider_requests": 1},
                                    {"automatic_retries": 1}])
def test_success_label_without_required_observations_cannot_activate(tmp_path, change):
    value = {**passed(), **change}
    assert connect.run_connection(directory=tmp_path, config_loader=config,
        preflight=lambda **_: value, stopper=lambda: pytest.fail("Unsafe activation")) == 2
    assert status(tmp_path)["status"] == "preflight_failed"


def test_preflight_exception_is_redacted_and_no_retry(tmp_path):
    calls = []
    def failing(**_):
        calls.append(1)
        raise RuntimeError(FAKE_KEY + " https://user:password@private.invalid")
    assert connect.run_connection(directory=tmp_path, config_loader=config, preflight=failing,
           stopper=lambda: pytest.fail("Must preserve offline server")) == 2
    assert calls == [1] and status(tmp_path)["status"] == "preflight_failed"
    encoded = "".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert FAKE_KEY not in encoded and "password@" not in encoded


def test_success_stops_owned_service_then_uses_same_configuration(tmp_path):
    calls = []
    def stop():
        calls.append("stop")
        assert status(tmp_path)["status"] == "activating"
    def serve(actual, finished_at, writer):
        calls.append("serve")
        assert actual.api_key == FAKE_KEY and actual.model == connect.DEFAULT_MODEL
        datetime.fromisoformat(finished_at)
        assert status(tmp_path)["status"] == "activating"
        writer.write("serving", preflight_finished_at=finished_at)
        assert status(tmp_path)["status"] == "serving"
        return True  # A mock orderly shutdown after readiness.
    assert connect.run_connection(directory=tmp_path, config_loader=config,
           preflight=lambda **_: passed(), stopper=stop, service_runner=serve) == 0
    assert calls == ["stop", "serve"] and status(tmp_path)["status"] == "stopped"
    assert status(tmp_path)["preflight_finished_at"]


def test_unknown_port_owner_preserved_and_never_starts_new_service(tmp_path):
    def stop():
        raise connect.ConnectionFailure("UnknownPortOwner")
    assert connect.run_connection(directory=tmp_path, config_loader=config,
        preflight=lambda **_: passed(), stopper=stop,
        service_runner=lambda *_: pytest.fail("Unknown owner must be preserved")) == 2
    assert status(tmp_path)["status"] == "error"
    assert status(tmp_path)["error_class"] == "UnknownPortOwner"


def test_preflight_report_redacts_reflected_secret_and_url_credentials(tmp_path):
    report = passed()
    report["cases"][0]["answer"] += " " + FAKE_KEY + " https://user:password@private.invalid"
    report["headers"] = {"Authorization": "Bearer " + FAKE_KEY}
    assert connect.run_connection(directory=tmp_path, config_loader=config,
        preflight=lambda **_: report, stopper=lambda: None,
        service_runner=lambda *_: True) == 0
    encoded = (tmp_path / "preflight.json").read_text()
    assert FAKE_KEY not in encoded and "password@" not in encoded and "Authorization" not in encoded
    assert "[REDACTED]" in encoded and "[URL omitted]" in encoded


def test_singleton_duplicate_exits_before_config_provider_or_status_change(tmp_path):
    (tmp_path / "status.json").write_text('{"status":"serving","pid":123}', encoding="utf-8")
    with connect.single_worker(tmp_path):
        assert connect.run_connection(directory=tmp_path,
            config_loader=lambda: pytest.fail("Duplicate read config"),
            preflight=lambda **_: pytest.fail("Duplicate inference")) == 3
    assert status(tmp_path) == {"status": "serving", "pid": 123}


def test_lock_file_can_be_reused_after_worker_exit(tmp_path):
    with connect.single_worker(tmp_path):
        pass
    with connect.single_worker(tmp_path):
        pass


def test_configuration_exception_is_class_only_and_no_preflight(tmp_path):
    def bad():
        raise RuntimeError(FAKE_KEY)
    assert connect.run_connection(directory=tmp_path, config_loader=bad,
          preflight=lambda **_: pytest.fail("Invalid config called provider")) == 2
    assert status(tmp_path)["error_class"] == "ConfigurationError"
    assert FAKE_KEY not in (tmp_path / "status.json").read_text()


def test_server_never_declares_serving_before_socket_ready(tmp_path, monkeypatch):
    import uvicorn
    async def not_ready(self, sockets=None):
        self.started = False
    monkeypatch.setattr(uvicorn.Server, "startup", not_ready)
    writer = connect.StatusWriter(tmp_path)
    writer.write("activating")
    server = connect.audited_server(object(), writer, "2026-11-07T00:00:00+00:00")
    asyncio.run(server.startup())
    assert status(tmp_path)["status"] == "activating"
    async def ready(self, sockets=None):
        self.started = True
    monkeypatch.setattr(uvicorn.Server, "startup", ready)
    asyncio.run(server.startup())
    assert status(tmp_path)["status"] == "serving"


def test_uvicorn_bind_failure_is_error_not_false_serving(tmp_path):
    def cannot_bind(*_):
        raise SystemExit(1)
    assert connect.run_connection(directory=tmp_path, config_loader=config,
          preflight=lambda **_: passed(), stopper=lambda: None, service_runner=cannot_bind) == 2
    assert status(tmp_path)["status"] == "error"
    assert status(tmp_path)["error_class"] == "ActivationError"


def test_stop_helper_output_is_parsed_without_inheriting_key(monkeypatch):
    monkeypatch.setenv("KNOWTRACE_MODEL_API_KEY", FAKE_KEY)
    def run(arguments, **kwargs):
        assert "KNOWTRACE_MODEL_API_KEY" not in kwargs["env"]
        assert FAKE_KEY not in json.dumps(arguments)
        return subprocess.CompletedProcess(arguments, 0, '{"status":"no_service"}', "")
    monkeypatch.setattr(connect.subprocess, "run", run)
    connect.stop_owned_demo()


@pytest.mark.parametrize("code,body,expected", [(2, '{"status":"blocked_unknown_owner"}', "UnknownPortOwner"),
    (0, '{"status":"owned_service_found"}', "StopServiceError"),
    (1, FAKE_KEY, "StopServiceError")])
def test_stop_helper_failure_is_public_class_only(monkeypatch, code, body, expected):
    monkeypatch.setattr(connect.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, body, FAKE_KEY))
    with pytest.raises(connect.ConnectionFailure) as caught:
        connect.stop_owned_demo()
    assert caught.value.code == expected and FAKE_KEY not in str(caught.value)


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell.exe") is None, reason="Windows PowerShell guard")
@pytest.mark.parametrize("scenario,expected,code", [
    ("owned", "stopped", 0), ("unknown", "blocked_unknown_owner", 2),
    ("lookalike", "blocked_unknown_owner", 2), ("reused", "blocked_unknown_owner", 2),
    ("check", "owned_service_found", 0), ("empty", "no_service", 0),
])
def test_powershell_process_guard_with_fake_os_inventory(scenario, expected, code):
    # Shadow every OS process/network/kill command in the child PowerShell.
    # The script under test can only inspect synthetic records; no actual PID is
    # touched. The numeric IDs below are fake and Stop-Process is a mock function.
    source = r'''
$global:Scenario = '__SCENARIO__'
$global:Python = '__PYTHON__'
$global:Command = '"' + $global:Python + '" -m knowtrace.cli serve --port 8766'
$global:FakeProcesses = @(
    [PSCustomObject]@{ProcessId=900001; ParentProcessId=900000; ExecutablePath=$global:Python; CommandLine=$global:Command; CreationDate='one'},
    [PSCustomObject]@{ProcessId=900002; ParentProcessId=900001; ExecutablePath='C:\synthetic\python.exe'; CommandLine='"C:\synthetic\python.exe" -m knowtrace.cli serve --port 8766'; CreationDate='two'}
)
$global:FakeListeners = @([PSCustomObject]@{LocalPort=8766; OwningProcess=900002})
if ($global:Scenario -eq 'unknown') {$global:FakeListeners[0].OwningProcess=900099}
if ($global:Scenario -eq 'lookalike') {$global:FakeProcesses[0].ExecutablePath+='-other'; $global:FakeProcesses[0].CommandLine='"C:\other\python.exe" -m knowtrace.cli serve --port 8766'}
if ($global:Scenario -eq 'empty') {$global:FakeProcesses=@(); $global:FakeListeners=@()}
function Get-CimInstance {
    [CmdletBinding()]param($ClassName, $Filter)
    if (-not $Filter) {return $global:FakeProcesses}
    $Record = $global:FakeProcesses | Where-Object {$Filter -eq ('ProcessId=' + $_.ProcessId)} | Select-Object *
    if ($Record -and $global:Scenario -eq 'reused') {$Record.CreationDate='changed'}
    return $Record
}
function Get-NetTCPConnection {[CmdletBinding()]param($State) return $global:FakeListeners}
function Stop-Process {
    [CmdletBinding()]param([int]$Id)
    if ($global:Scenario -in @('unknown','lookalike','reused','check','empty')) {throw 'A guarded case attempted a kill'}
    if ($Id -notin @(900001,900002)) {throw 'An unowned PID was selected'}
    $global:FakeListeners=@($global:FakeListeners | Where-Object {$_.OwningProcess -ne $Id})
    $global:FakeProcesses=@($global:FakeProcesses | Where-Object {$_.ProcessId -ne $Id})
}
& '__SCRIPT__' __CHECK__
exit $LASTEXITCODE
'''
    source = source.replace("__SCENARIO__", scenario).replace("__PYTHON__", str(connect.ROOT / ".venv/Scripts/python.exe").replace("'", "''"))
    source = source.replace("__SCRIPT__", str(connect.ROOT / "scripts/stop_owned_demo.ps1").replace("'", "''"))
    source = source.replace("__CHECK__", "-CheckOnly" if scenario == "check" else "")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True, encoding="utf-8", timeout=15)
    assert result.returncode == code, result.stdout
    assert json.loads(result.stdout)["status"] == expected
