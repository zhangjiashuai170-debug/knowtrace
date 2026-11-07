"""Controlled replacement tests: fake credentials, model, service and OS calls."""
from contextlib import ExitStack
from datetime import datetime
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from knowtrace.generation import ModelConfig
from scripts import connect_deepseek as connect

FAKE_KEY = "upgrade-unit-test-only-not-a-real-key"
OLD_STATUS = {"status": "serving", "pid": 900002, "host": "127.0.0.1", "port": 8766}


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda *_: pytest.fail("No real provider network in upgrade tests"))


def config():
    return ModelConfig(provider="openai_compatible", base_url="https://api.deepseek.com",
                       model="deepseek-flash", api_key=FAKE_KEY)


def passed():
    return {"status": "protocol_checks_passed", "model_inference_observed": True,
            "provider_requests_forwarded": 1, "private_case_provider_requests": 0,
            "automatic_retries": 0, "semantic_verified": False,
            "cases": [{"answer": "The snapshots remain for 14 days [S1]."}]}


def read(directory, name):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def old_files(directory):
    (directory / "status.json").write_text(json.dumps(OLD_STATUS), encoding="utf-8")
    (directory / "preflight.json").write_text('{"old_evidence":"preserve"}', encoding="utf-8")
    return {name: (directory / name).read_bytes() for name in ("status.json", "preflight.json")}


def assert_old_preserved(directory, originals):
    assert all((directory / name).read_bytes() == contents for name, contents in originals.items())


def test_failed_upgrade_keeps_current_worker_lock_and_evidence(tmp_path):
    originals = old_files(tmp_path)
    calls = []
    def preflight(**kwargs):
        calls.append("preflight")
        assert kwargs["run_live"] and kwargs["config_loader"]().api_key == FAKE_KEY
        return {"status": "checks_failed_or_unknown", "model_inference_observed": None,
                "provider_requests_forwarded": 1, "automatic_retries": 0}
    with connect.single_worker(tmp_path):
        assert connect.run_upgrade(directory=tmp_path, config_loader=config,
            preflight=preflight, stopper=lambda: pytest.fail("Failed preflight stopped old worker"),
            service_runner=lambda *_: pytest.fail("Failed preflight started service")) == 2
        with pytest.raises(connect.AlreadyRunning):
            with connect.single_worker(tmp_path):
                pass
    assert calls == ["preflight"]
    assert_old_preserved(tmp_path, originals)
    assert read(tmp_path, "upgrade-status.json")["status"] == "preflight_failed"


@pytest.mark.parametrize("change", [{"model_inference_observed": False},
    {"model_inference_observed": None}, {"provider_requests_forwarded": 2},
    {"private_case_provider_requests": 1}, {"automatic_retries": 1}])
def test_upgrade_cannot_activate_from_labels_without_actual_one_request(tmp_path, change):
    originals = old_files(tmp_path)
    report = {**passed(), **change}
    assert connect.run_upgrade(directory=tmp_path, config_loader=config,
        preflight=lambda **_: report, stopper=lambda: pytest.fail("Unverified activation")) == 2
    assert_old_preserved(tmp_path, originals)


def test_config_failure_does_not_read_old_credentials_or_preflight(tmp_path):
    originals = old_files(tmp_path)
    def bad_config():
        raise RuntimeError(FAKE_KEY)
    assert connect.run_upgrade(directory=tmp_path, config_loader=bad_config,
        preflight=lambda **_: pytest.fail("Invalid config called model"),
        stopper=lambda: pytest.fail("Invalid config stopped worker")) == 2
    assert_old_preserved(tmp_path, originals)
    encoded = (tmp_path / "upgrade-status.json").read_text()
    assert FAKE_KEY not in encoded and '"error_class": "ConfigurationError"' in encoded


def test_duplicate_upgrade_never_reads_key_dispatches_or_overwrites_audit(tmp_path):
    originals = old_files(tmp_path)
    (tmp_path / "upgrade-status.json").write_text('{"existing_upgrade":true}', encoding="utf-8")
    with connect.single_upgrade(tmp_path):
        assert connect.run_upgrade(directory=tmp_path,
            config_loader=lambda: pytest.fail("Duplicate read new key"),
            preflight=lambda **_: pytest.fail("Duplicate model request")) == 3
    assert read(tmp_path, "upgrade-status.json") == {"existing_upgrade": True}
    assert_old_preserved(tmp_path, originals)


def test_upgrade_preflight_exception_is_safe_and_not_retried(tmp_path):
    originals = old_files(tmp_path)
    calls = []
    def fail(**_):
        calls.append(1)
        raise RuntimeError(FAKE_KEY + " https://user:password@provider.invalid")
    assert connect.run_upgrade(directory=tmp_path, config_loader=config, preflight=fail,
        stopper=lambda: pytest.fail("Exception stopped current service")) == 2
    assert calls == [1]
    assert_old_preserved(tmp_path, originals)
    text = "".join(p.read_text() for p in tmp_path.glob("upgrade-*.json"))
    assert FAKE_KEY not in text and "password@" not in text


def test_unknown_owner_after_good_preflight_keeps_old_status_and_model(tmp_path):
    originals = old_files(tmp_path)
    def stop():
        assert_old_preserved(tmp_path, originals)
        assert read(tmp_path, "upgrade-status.json")["status"] == "activating"
        raise connect.ConnectionFailure("UnknownPortOwner")
    assert connect.run_upgrade(directory=tmp_path, config_loader=config,
        preflight=lambda **_: passed(), stopper=stop,
        service_runner=lambda *_: pytest.fail("Unknown owner was replaced")) == 2
    assert_old_preserved(tmp_path, originals)
    assert read(tmp_path, "upgrade-status.json")["error_class"] == "UnknownPortOwner"


def test_success_upgrades_once_and_releases_transition_lock_only_after_readiness(tmp_path):
    originals = old_files(tmp_path)
    calls = []
    old = ExitStack()
    old.enter_context(connect.single_worker(tmp_path))
    def preflight(**kwargs):
        calls.append("preflight")
        assert_old_preserved(tmp_path, originals)
        assert kwargs["config_loader"]().api_key == FAKE_KEY
        return passed()
    def stop():
        calls.append("stop")
        assert_old_preserved(tmp_path, originals)
        old.close()  # Fake the already-verified old process exiting.
    def serve(actual, finished_at, writer):
        calls.append("serve")
        assert actual.api_key == FAKE_KEY
        datetime.fromisoformat(finished_at)
        assert read(tmp_path, "status.json")["status"] == "activating"
        with pytest.raises(connect.AlreadyRunning):
            with connect.single_upgrade(tmp_path):
                pass
        writer.write("serving", preflight_finished_at=finished_at)
        assert read(tmp_path, "status.json")["status"] == "serving"
        assert read(tmp_path, "upgrade-status.json")["status"] == "serving"
        # Later upgrades must be possible while the new service owns worker.lock.
        with connect.single_upgrade(tmp_path):
            pass
        with pytest.raises(connect.AlreadyRunning):
            with connect.single_worker(tmp_path):
                pass
        return True
    try:
        assert connect.run_upgrade(directory=tmp_path, config_loader=config,
            preflight=preflight, stopper=stop, service_runner=serve) == 0
    finally:
        old.close()
    assert calls == ["preflight", "stop", "serve"]
    assert read(tmp_path, "status.json")["status"] == "stopped"
    assert read(tmp_path, "upgrade-status.json")["status"] == "stopped"
    assert read(tmp_path, "preflight.json")["live_check"]["provider_requests_forwarded"] == 1


def test_racing_worker_is_preserved_and_upgrade_does_not_repreflight(tmp_path):
    originals = old_files(tmp_path)
    calls = []
    def preflight(**_):
        calls.append(1)
        return passed()
    with connect.single_worker(tmp_path):
        assert connect.run_upgrade(directory=tmp_path, config_loader=config,
            preflight=preflight, stopper=lambda: None,
            service_runner=lambda *_: pytest.fail("Must preserve racing worker")) == 3
    assert calls == [1]
    assert_old_preserved(tmp_path, originals)


def test_activation_failure_is_error_never_serving(tmp_path):
    old_files(tmp_path)
    def fail(*_):
        raise SystemExit(1)
    assert connect.run_upgrade(directory=tmp_path, config_loader=config,
        preflight=lambda **_: passed(), stopper=lambda: None, service_runner=fail) == 2
    for filename in ("status.json", "upgrade-status.json"):
        value = read(tmp_path, filename)
        assert value["status"] == "error" and value["error_class"] == "ActivationError"


def test_upgrade_report_redacts_reflected_key_and_raw_headers(tmp_path):
    old_files(tmp_path)
    result = passed()
    result["cases"][0]["answer"] += " " + FAKE_KEY + " https://user:password@provider.invalid"
    result["headers"] = {"Authorization": FAKE_KEY}
    assert connect.run_upgrade(directory=tmp_path, config_loader=config,
        preflight=lambda **_: result, stopper=lambda: None, service_runner=lambda *_: True) == 0
    for filename in ("preflight.json", "upgrade-preflight.json"):
        text = (tmp_path / filename).read_text()
        assert FAKE_KEY not in text and "password@" not in text and "Authorization" not in text


def test_new_service_uses_configured_local_encoder_without_downloading(monkeypatch, tmp_path):
    from knowtrace import app as app_module
    from knowtrace import semantic_config
    encoder = object()
    seen = {}
    application = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setattr(semantic_config, "configured_embedder", lambda: (encoder, True))
    def create(data_dir, **kwargs):
        seen.update(kwargs)
        return application
    monkeypatch.setattr(app_module, "create_app", create)
    server = SimpleNamespace(started=True, run=lambda: None)
    monkeypatch.setattr(connect, "audited_server", lambda *args: server)
    assert connect.serve_app(config(), "2026-11-07T00:00:00+00:00", connect.StatusWriter(tmp_path))
    assert seen["embedder"] is encoder and seen["retrieval_configured"] is True
    assert seen["model_config"].api_key == FAKE_KEY
    assert application.state.last_model_success_scope == "isolated_synthetic_preflight"


def test_cli_upgrade_is_explicit_and_unknown_arguments_never_echo(monkeypatch, capsys):
    monkeypatch.setattr(connect, "run_connection", lambda: 40)
    monkeypatch.setattr(connect, "run_upgrade", lambda: 41)
    assert connect.main([]) == 40 and connect.main(["--upgrade"]) == 41
    assert connect.main([FAKE_KEY]) == 2
    assert capsys.readouterr() == ("", "")


def test_upgrade_stop_helper_drops_key_environment_and_accepts_only_stopped(monkeypatch):
    monkeypatch.setenv("KNOWTRACE_MODEL_API_KEY", FAKE_KEY)
    def run(arguments, **kwargs):
        assert "KNOWTRACE_MODEL_API_KEY" not in kwargs["env"]
        assert "stop_owned_deepseek.ps1" in arguments[-1]
        assert FAKE_KEY not in json.dumps(arguments)
        return subprocess.CompletedProcess(arguments, 0, '{"status":"stopped"}', "")
    monkeypatch.setattr(connect.subprocess, "run", run)
    connect.stop_owned_deepseek()


@pytest.mark.parametrize("code,body,expected", [(2, '{"status":"blocked_unknown_owner"}', "UnknownPortOwner"),
    (0, '{"status":"no_service"}', "StopServiceError"),
    (1, FAKE_KEY, "StopServiceError")])
def test_upgrade_stop_failures_are_class_only(monkeypatch, code, body, expected):
    monkeypatch.setattr(connect.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, body, FAKE_KEY))
    with pytest.raises(connect.ConnectionFailure) as caught:
        connect.stop_owned_deepseek()
    assert caught.value.code == expected and FAKE_KEY not in str(caught.value)


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell.exe") is None, reason="Windows guard")
@pytest.mark.parametrize("scenario,expected,code", [
    ("owned", "stopped", 0), ("upgraded", "stopped", 0),
    ("absolute", "stopped", 0), ("check", "owned_worker_found", 0),
    ("unknown", "blocked_unknown_owner", 2), ("lookalike", "blocked_unknown_owner", 2),
    ("reused", "blocked_unknown_owner", 2), ("extra_args", "blocked_unknown_owner", 2),
    ("parent_args", "blocked_unknown_owner", 2), ("status_changed", "blocked_unknown_owner", 2),
    ("symlink", "inspection_or_stop_failed", 2), ("oversized", "inspection_or_stop_failed", 2),
])
def test_worker_stop_guard_with_entirely_fake_os_state(scenario, expected, code):
    # Every status/process/network/stop operation is shadowed. No real PID,
    # service, production status file or environment key is inspected here.
    source = r'''
$global:Scenario = '__SCENARIO__'
$global:Python = '__PYTHON__'
$global:Script = '__WORKER__'
$global:Arguments = ' -u scripts/connect_deepseek.py'
if ($global:Scenario -eq 'upgraded') {$global:Arguments += ' --upgrade'}
if ($global:Scenario -eq 'absolute') {$global:Arguments = ' -u "' + $global:Script + '"'}
if ($global:Scenario -eq 'extra_args') {$global:Arguments += ' --unsafe'}
$global:Command = '"' + $global:Python + '"' + $global:Arguments
$global:FakeProcesses = @(
    [PSCustomObject]@{ProcessId=900001; ParentProcessId=900000; ExecutablePath=$global:Python; CommandLine=$global:Command; CreationDate='one'},
    [PSCustomObject]@{ProcessId=900002; ParentProcessId=900001; ExecutablePath='C:\synthetic\python.exe'; CommandLine='"C:\synthetic\python.exe"' + $global:Arguments; CreationDate='two'}
)
$global:FakeListeners = @([PSCustomObject]@{LocalPort=8766; OwningProcess=900002})
$global:StatusReads = 0
if ($global:Scenario -eq 'unknown') {$global:FakeListeners += [PSCustomObject]@{LocalPort=8766; OwningProcess=900099}}
if ($global:Scenario -eq 'lookalike') {$global:FakeProcesses[0].ExecutablePath+='-other'}
if ($global:Scenario -eq 'parent_args') {$global:FakeProcesses[0].CommandLine += ' --upgrade'}
function Get-Item {
    [CmdletBinding()]param($LiteralPath)
    $Length = 200; $Attributes = [IO.FileAttributes]::Normal
    if ($global:Scenario -eq 'oversized') {$Length = 9000}
    if ($global:Scenario -eq 'symlink') {$Attributes = [IO.FileAttributes]::ReparsePoint}
    return [PSCustomObject]@{PSIsContainer=$false; Length=$Length; Attributes=$Attributes}
}
function Get-Content {
    [CmdletBinding()]param($LiteralPath, [switch]$Raw, $Encoding)
    $global:StatusReads += 1
    $RecordPid = 900002
    if ($global:Scenario -eq 'status_changed' -and $global:StatusReads -gt 1) {$RecordPid=900099}
    return (@{status='serving'; pid=$RecordPid; host='127.0.0.1'; port=8766} | ConvertTo-Json -Compress)
}
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
    if ($global:Scenario -notin @('owned','upgraded','absolute')) {throw 'Guarded branch attempted stop'}
    if ($Id -notin @(900001,900002)) {throw 'Unowned fake PID selected'}
    $global:FakeListeners=@($global:FakeListeners | Where-Object {$_.OwningProcess -ne $Id})
    $global:FakeProcesses=@($global:FakeProcesses | Where-Object {$_.ProcessId -ne $Id})
}
& '__STOPPER__' __CHECK__
exit $LASTEXITCODE
'''
    source = source.replace("__SCENARIO__", scenario)
    replacements = {"__PYTHON__": connect.ROOT / ".venv/Scripts/python.exe",
                    "__WORKER__": connect.ROOT / "scripts/connect_deepseek.py",
                    "__STOPPER__": connect.ROOT / "scripts/stop_owned_deepseek.ps1"}
    for marker, value in replacements.items():
        source = source.replace(marker, str(value).replace("'", "''"))
    source = source.replace("__CHECK__", "-CheckOnly" if scenario == "check" else "")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True, encoding="utf-8", timeout=15)
    assert result.returncode == code, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == expected


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell.exe") is None, reason="Windows launcher")
def test_upgrade_prompt_check_only_neither_prompts_nor_starts():
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(connect.ROOT / "scripts/connect_deepseek.ps1"), "-Upgrade", "-CheckOnly"],
        capture_output=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0
    value = json.loads(result.stdout)
    assert value["upgrade"] is True
    assert value["starts_service"] is False and value["prompts_for_key"] is False
    assert value["sends_model_request"] is False
