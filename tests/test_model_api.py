import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from knowtrace.app import create_app
from knowtrace.config import model_config_from_env
from knowtrace.generation import ModelConfig


def make_client(tmp_path, *, enabled=True, response=None, fail=False):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if fail:
            return httpx.Response(500, text="sensitive-provider-body secret-marker")
        content = response if response is not None else {"answer": "Starling timeout is 37 seconds [S1]", "refused": False, "citations": ["S1"]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})
    config = ModelConfig(provider="openai_compatible", base_url="https://model.example/v1", model="test-only", api_key="secret-marker") if enabled else ModelConfig()
    app = create_app(tmp_path / "data", model_config=config, model_transport=httpx.MockTransport(handler))
    core = app.state.core
    core.create_kb("Engineering", kb_id="kb-engineering")
    core.create_kb("Support", kb_id="kb-support")
    for name, kb, text in [("eng.txt", "kb-engineering", "Starling timeout is 37 seconds."), ("private.txt", "kb-support", "Private support nectar code is C928.")]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        core.ingest_file(path, name, kb)
    client = TestClient(app, base_url="http://127.0.0.1")
    client.post("/api/auth/login", json={"username": "engineer", "password": "engineer-demo"})
    return client, requests


def test_default_offline_no_provider_calls_even_if_configured(tmp_path):
    client, calls = make_client(tmp_path)
    status = client.get("/api/model").json()
    assert status["configured"] is True and status["live_verified"] is False
    assert status["last_success_at"] is None and status["last_success_scope"] is None
    assert "secret-marker" not in json.dumps(status)
    response = client.post("/api/ask", json={"question": "starling timeout"})
    assert response.json()["answer_mode"] == "extractive"
    assert calls == []


def test_generated_receives_only_scoped_sources_and_records_actual_request(tmp_path):
    client, calls = make_client(tmp_path)
    response = client.post("/api/ask", json={"question": "starling timeout", "answer_mode": "generated"})
    assert response.status_code == 200
    value = response.json()
    assert value["answer_mode"] == "generated" and value["model_invoked"] is True
    assert len(calls) == 1
    payload = json.dumps(calls[0])
    assert "Starling" in payload and "C928" not in payload
    assert "secret-marker" not in response.text
    status = client.get("/api/model").json()
    assert status["live_verified"] is True
    assert status["last_success_at"] and status["last_success_scope"] == "current_app_request"


def test_synthetic_preflight_history_is_distinguished_from_app_request(tmp_path):
    client, calls = make_client(tmp_path)
    app = client.app
    app.state.last_model_success_at = "2026-11-07T13:00:00+00:00"
    app.state.last_model_success_scope = "isolated_synthetic_preflight"
    status = client.get("/api/model").json()
    assert status["live_verified"] is True
    assert status["last_success_at"] == "2026-11-07T13:00:00+00:00"
    assert status["last_success_scope"] == "isolated_synthetic_preflight"
    assert calls == []
    assert client.post("/api/ask", json={"question": "starling timeout", "answer_mode": "generated"}).status_code == 200
    assert client.get("/api/model").json()["last_success_scope"] == "current_app_request"


def test_refusal_and_invalid_scope_do_not_call_model(tmp_path):
    client, calls = make_client(tmp_path)
    response = client.post("/api/ask", json={"question": "zebras galaxy pineapple", "answer_mode": "generated"})
    assert response.status_code == 200
    assert response.json()["refused"] and response.json()["model_invoked"] is False
    assert response.json()["answer_mode"] == "extractive"
    assert client.post("/api/ask", json={"question": "code", "kb_ids": ["kb-support"], "answer_mode": "generated"}).status_code == 403
    assert calls == []


def test_disabled_and_failed_generation_are_explicit(tmp_path):
    client, calls = make_client(tmp_path / "disabled", enabled=False)
    assert client.post("/api/ask", json={"question": "starling timeout", "answer_mode": "generated"}).status_code == 409
    assert calls == []
    client, calls = make_client(tmp_path / "failure", fail=True)
    response = client.post("/api/ask", json={"question": "starling timeout", "answer_mode": "generated"})
    assert response.status_code == 502
    assert "sensitive-provider-body" not in response.text and "secret-marker" not in response.text
    assert response.headers["X-KnowTrace-Model-Error"] == "provider_error"
    assert response.headers["X-KnowTrace-Provider-Status"] == "500"
    assert client.get("/api/model").json()["live_verified"] is False


def test_unknown_citation_not_presented_as_answer(tmp_path):
    client, _ = make_client(tmp_path, response={"answer": "No evidence [S999]", "refused": False, "citations": ["S999"]})
    response = client.post("/api/ask", json={"question": "starling timeout", "answer_mode": "generated"})
    assert response.status_code == 502 and "S999" not in response.text


def test_env_configuration_is_explicit_no_key_reads_when_disabled():
    class Disabled(dict):
        def get(self, key, default=None):
            assert key == "KNOWTRACE_MODEL_PROVIDER"
            return "disabled"
    assert model_config_from_env(Disabled()).provider == "disabled"
    for url in ["http://remote.example/v1", "https://user:password@example.org/v1", "https://example.org/v1?secret=x", "https://example.org/v1#fragment", "file:///tmp/x", "http://localhost.evil/v1"]:
        with pytest.raises(ValueError):
            model_config_from_env({"KNOWTRACE_MODEL_PROVIDER": "openai_compatible", "KNOWTRACE_MODEL_BASE_URL": url, "KNOWTRACE_MODEL_NAME": "demo"})
    config = model_config_from_env({"KNOWTRACE_MODEL_PROVIDER": "ollama", "KNOWTRACE_MODEL_BASE_URL": "http://127.0.0.1:11434", "KNOWTRACE_MODEL_NAME": "existing-local-model"})
    assert config.provider == "ollama" and config.api_key == ""
    config = model_config_from_env({"KNOWTRACE_MODEL_PROVIDER": "openai_compatible", "KNOWTRACE_MODEL_BASE_URL": "https://model.example/v1", "KNOWTRACE_MODEL_NAME": "demo", "KNOWTRACE_MODEL_API_KEY": "secret-marker"})
    assert "secret-marker" not in repr(config)
