"""One controlled DeepSeek preflight, then a loopback KnowTrace service.

Run only after the local masked-input launcher supplies an API key in this
process's environment. Credentials never enter command arguments or artifacts.
The preflight uses synthetic temporary documents, not the main data directory.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowtrace.config import model_config_from_env
from knowtrace.generation import ModelConfig
from scripts.check_model_live import run_isolated_live, safe_text

DEFAULT_MODEL = "deepseek-flash"
ALLOWED_MODELS = {DEFAULT_MODEL, "deepseek-v4-pro"}
STATUS_DIRECTORY = ROOT / "artifacts" / "deepseek-connection"
HOST, PORT = "127.0.0.1", 8766
STATES = {"validating", "preflight_failed", "activating", "serving", "stopped", "error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConnectionFailure(Exception):
    """Contains a fixed public code, never the underlying exception text."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class AlreadyRunning(ConnectionFailure):
    def __init__(self):
        super().__init__("AlreadyRunning")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class StatusWriter:
    """Only allow known non-secret status fields into the public artifact."""
    def __init__(self, directory: Path, *, filename="status.json"):
        self.directory = Path(directory)
        if filename not in {"status.json", "upgrade-status.json"}:
            raise ValueError("invalid_status_filename")
        self.filename = filename
        self.model = DEFAULT_MODEL
        self.last_state = None

    def write(self, state: str, *, error_class: str | None = None,
              preflight_finished_at: str | None = None) -> None:
        if state not in STATES:
            raise ValueError("invalid_connection_state")
        value = {"status": state, "updated_at": utc_now(), "pid": os.getpid(),
                 "provider": "openai_compatible", "model": self.model,
                 "host": HOST, "port": PORT}
        if error_class is not None:
            # All callers supply fixed codes, never str(error) or provider output.
            allowed = {"ConfigurationError", "PreflightError", "ProtocolCheckFailed",
                       "UnknownPortOwner", "StopServiceError", "ActivationError", "ConnectionError"}
            value["error_class"] = error_class if error_class in allowed else "ConnectionError"
        if preflight_finished_at is not None:
            # The value is generated locally after preflight, not model output.
            value["preflight_finished_at"] = preflight_finished_at
        write_json(self.directory / self.filename, value)
        self.last_state = state


@contextmanager
def _file_lock(directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise AlreadyRunning() from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def single_worker(directory: Path):
    """Hold a process lock throughout preflight and serving; stale files are fine."""
    with _file_lock(directory, "worker.lock"):
        yield


@contextmanager
def single_upgrade(directory: Path):
    """Serialise upgrade preflights without touching an active worker's lock."""
    with _file_lock(directory, "upgrade.lock"):
        yield


def load_deepseek_config(environ=None) -> ModelConfig:
    """Load only explicit model fields; never discover keys in other files."""
    source = os.environ if environ is None else environ
    names = ("KNOWTRACE_MODEL_PROVIDER", "KNOWTRACE_MODEL_BASE_URL", "KNOWTRACE_MODEL_NAME",
             "KNOWTRACE_MODEL_API_KEY", "KNOWTRACE_MODEL_TIMEOUT_SECONDS")
    selected = {name: source[name] for name in names if name in source}
    selected.setdefault("KNOWTRACE_MODEL_PROVIDER", "openai_compatible")
    selected.setdefault("KNOWTRACE_MODEL_BASE_URL", "https://api.deepseek.com")
    selected.setdefault("KNOWTRACE_MODEL_NAME", DEFAULT_MODEL)
    try:
        config = model_config_from_env(selected)
        parsed = urlsplit(config.base_url)
        if (config.provider != "openai_compatible" or parsed.scheme != "https"
                or parsed.hostname != "api.deepseek.com" or parsed.port not in {None, 443}
                or parsed.path not in {"", "/v1"} or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or "?" in config.base_url or "#" in config.base_url or "\\" in config.base_url
                or config.model not in ALLOWED_MODELS or not config.api_key
                or config.api_key != config.api_key.strip()
                or any(ord(char) < 33 or ord(char) > 126 for char in config.api_key)):
            raise ValueError()
        return config
    except Exception:
        raise ConnectionFailure("ConfigurationError") from None
    finally:
        if environ is None:
            # The configuration object retains the key in memory. Subsequent
            # process helpers must not inherit it as an environment variable.
            os.environ.pop("KNOWTRACE_MODEL_API_KEY", None)


def sanitized_report(value, api_key: str):
    """Defence in depth over the live helper's already-redacted report."""
    if isinstance(value, dict):
        return {safe_text(str(key), (api_key,), 120): sanitized_report(item, api_key)
                for key, item in value.items()
                if str(key).lower() not in {"api_key", "authorization", "headers", "token", "base_url"}}
    if isinstance(value, list):
        return [sanitized_report(item, api_key) for item in value[:100]]
    if isinstance(value, str):
        return safe_text(value, (api_key,))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def passed_preflight(report: dict) -> bool:
    return (isinstance(report, dict) and report.get("status") == "protocol_checks_passed"
            and report.get("model_inference_observed") is True
            and report.get("provider_requests_forwarded") == 1
            and report.get("private_case_provider_requests") == 0
            and report.get("automatic_retries") == 0)


def stop_owned_demo() -> None:
    """A separate Windows guard validates ownership before stopping any process."""
    environment = dict(os.environ)
    environment.pop("KNOWTRACE_MODEL_API_KEY", None)
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
             str(ROOT / "scripts" / "stop_owned_demo.ps1")],
            capture_output=True, encoding="utf-8", timeout=20, check=False,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if len(result.stdout) > 10_000:
            raise ValueError()
        value = json.loads(result.stdout)
        if value.get("status") == "blocked_unknown_owner":
            raise ConnectionFailure("UnknownPortOwner")
        if result.returncode != 0 or value.get("status") not in {"no_service", "stopped"}:
            raise ValueError()
    except ConnectionFailure:
        raise
    except Exception:
        raise ConnectionFailure("StopServiceError") from None


def stop_owned_deepseek() -> None:
    """Replace only the status-identified, currently listening project worker."""
    environment = dict(os.environ)
    environment.pop("KNOWTRACE_MODEL_API_KEY", None)
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
             str(ROOT / "scripts" / "stop_owned_deepseek.ps1")],
            capture_output=True, encoding="utf-8", timeout=20, check=False,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if len(result.stdout) > 10_000:
            raise ValueError()
        value = json.loads(result.stdout)
        if value.get("status") == "blocked_unknown_owner":
            raise ConnectionFailure("UnknownPortOwner")
        if result.returncode != 0 or value.get("status") != "stopped":
            raise ValueError()
    except ConnectionFailure:
        raise
    except Exception:
        raise ConnectionFailure("StopServiceError") from None


def serve_app(config: ModelConfig, finished_at: str, status: StatusWriter) -> bool:
    """Set serving only after uvicorn has completed startup and bound its socket."""
    import uvicorn
    from knowtrace.app import create_app

    from knowtrace.semantic_config import configured_embedder

    embedder, retrieval_configured = configured_embedder()
    app = create_app(ROOT / "data", model_config=config, embedder=embedder,
                     retrieval_configured=retrieval_configured)
    app.state.last_model_success_at = finished_at
    app.state.last_model_success_scope = "isolated_synthetic_preflight"
    server = audited_server(app, status, finished_at)
    server.run()
    return bool(server.started)


def audited_server(app, status: StatusWriter, finished_at: str):
    import uvicorn

    class AuditedServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                status.write("serving", preflight_finished_at=finished_at)

    config = uvicorn.Config(app, host=HOST, port=PORT, log_config=None,
                            log_level="error", access_log=False)
    return AuditedServer(config)


def run_connection(*, directory=STATUS_DIRECTORY, config_loader=load_deepseek_config,
                   preflight=run_isolated_live, stopper=stop_owned_demo,
                   service_runner=serve_app, lock=single_worker) -> int:
    """Injectable boundaries keep unit tests entirely offline and non-destructive."""
    directory = Path(directory)
    status = StatusWriter(directory)
    try:
        with lock(directory):
            status.write("validating")
            try:
                config = config_loader()
                status.model = config.model
            except Exception:
                status.write("error", error_class="ConfigurationError")
                return 2
            try:
                result = preflight(run_live=True, config_loader=lambda: config)
            except Exception:
                result = {"status": "error_or_unknown", "model_inference_observed": None,
                          "automatic_retries": 0}
            finished_at = utc_now()
            write_json(directory / "preflight.json", {"finished_at": finished_at,
                       "provider": config.provider, "model": config.model,
                       "live_check": sanitized_report(result, config.api_key),
                       "semantic_verified": False})
            if not passed_preflight(result):
                status.write("preflight_failed", error_class="ProtocolCheckFailed",
                             preflight_finished_at=finished_at)
                return 2
            status.write("activating", preflight_finished_at=finished_at)
            try:
                stopper()
                if not service_runner(config, finished_at, status):
                    raise ConnectionFailure("ActivationError")
            except (Exception, SystemExit) as error:
                public_code = error.code if isinstance(error, ConnectionFailure) else "ActivationError"
                status.write("error", error_class=public_code, preflight_finished_at=finished_at)
                return 2
            status.write("stopped", preflight_finished_at=finished_at)
            return 0
    except AlreadyRunning:
        # Preserve the existing worker's status and never send another request.
        return 3
    except (Exception, SystemExit):
        try:
            status.write("error", error_class="ConnectionError")
        except Exception:
            pass
        return 2


class UpgradeServiceStatus(StatusWriter):
    """After owned replacement, reflect real startup in both safe audit files."""
    def __init__(self, directory: Path, upgrade_status: StatusWriter, *, on_serving=None):
        super().__init__(directory)
        self.upgrade_status = upgrade_status
        self.model = upgrade_status.model
        self.on_serving = on_serving

    def write(self, state: str, **kwargs):
        super().write(state, **kwargs)
        self.upgrade_status.write(state, **kwargs)
        if state == "serving" and self.on_serving is not None:
            callback, self.on_serving = self.on_serving, None
            callback()


def run_upgrade(*, directory=STATUS_DIRECTORY, config_loader=load_deepseek_config,
                preflight=run_isolated_live, stopper=stop_owned_deepseek,
                service_runner=serve_app, lock=single_upgrade, worker_lock=single_worker) -> int:
    """Validate a fresh key once; preserve the old service until checks pass.

    Until the controlled stopper succeeds and worker.lock is acquired, only
    upgrade-* audit files are written. No old process state/credentials are read.
    """
    directory = Path(directory)
    audit = StatusWriter(directory, filename="upgrade-status.json")
    try:
        with ExitStack() as lifetime:
            transition = lifetime.enter_context(ExitStack())
            transition.enter_context(lock(directory))
            audit.write("validating")
            try:
                config = config_loader()
                audit.model = config.model
            except Exception:
                audit.write("error", error_class="ConfigurationError")
                return 2
            try:
                result = preflight(run_live=True, config_loader=lambda: config)
            except Exception:
                result = {"status": "error_or_unknown", "model_inference_observed": None,
                          "automatic_retries": 0}
            finished_at = utc_now()
            report = {"finished_at": finished_at, "provider": config.provider,
                      "model": config.model,
                      "live_check": sanitized_report(result, config.api_key),
                      "semantic_verified": False}
            write_json(directory / "upgrade-preflight.json", report)
            if not passed_preflight(result):
                audit.write("preflight_failed", error_class="ProtocolCheckFailed",
                            preflight_finished_at=finished_at)
                return 2
            audit.write("activating", preflight_finished_at=finished_at)
            try:
                stopper()
            except (Exception, SystemExit) as error:
                code = error.code if isinstance(error, ConnectionFailure) else "StopServiceError"
                audit.write("error", error_class=code, preflight_finished_at=finished_at)
                return 2
            try:
                lifetime.enter_context(worker_lock(directory))
            except AlreadyRunning:
                # A separately started worker won the port/lock transition.
                # Do not replace that worker's status or dispatch again.
                audit.write("error", error_class="ConnectionError", preflight_finished_at=finished_at)
                return 3
            status = UpgradeServiceStatus(directory, audit, on_serving=transition.close)
            status.write("activating", preflight_finished_at=finished_at)
            write_json(directory / "preflight.json", report)
            try:
                if not service_runner(config, finished_at, status):
                    raise ConnectionFailure("ActivationError")
            except (Exception, SystemExit) as error:
                code = error.code if isinstance(error, ConnectionFailure) else "ActivationError"
                status.write("error", error_class=code, preflight_finished_at=finished_at)
                return 2
            status.write("stopped", preflight_finished_at=finished_at)
            return 0
    except AlreadyRunning:
        # A second upgrade never reads the new key, dispatches or overwrites audit.
        return 3
    except (Exception, SystemExit):
        try:
            audit.write("error", error_class="ConnectionError")
        except Exception:
            pass
        return 2


def main(argv=None) -> int:
    # Deliberately no CLI key/config arguments and no console output.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--upgrade"]:
        return run_upgrade()
    if not arguments:
        return run_connection()
    return 2


if __name__ == "__main__":
    # Only the standalone hidden worker redirects streams; imports/tests keep
    # their own stdout. No raw uvicorn/provider exception can reach a console.
    # Keep these streams open through interpreter finalization as well.
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
    sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    raise SystemExit(main())
