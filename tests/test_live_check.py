"""The live-check tool is tested only with in-process API and mock providers."""
import json

import httpx
import pytest

from knowtrace.generation import ModelConfig
from scripts import check_model_live as live

FAKE_KEY = "live-check-unit-key-not-real"


@pytest.fixture(autouse=True)
def forbid_actual_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live-check unit tests must never contact a real service")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)


def config():
    return ModelConfig(provider="openai_compatible", base_url="https://mock-provider.invalid/v1",
                       model="mock-model", api_key=FAKE_KEY, timeout_seconds=2)


def model_response(answer="Snapshots are retained for 14 days. [S1]", refused=False, citations=None):
    value = {"answer": answer, "refused": refused,
             "citations": (["S1"] if not refused else []) if citations is None else citations}
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})


@pytest.mark.parametrize("url", ["https://remote.invalid", "http://127.0.0.1:8766/path",
                                 "http://user:password@localhost:8766", "http://localhost:8766?key=secret",
                                 "http://127.0.0.1:8766/#fragment", "file:///tmp/model", "http://localhost.evil.invalid"])
def test_api_target_rejects_remote_credentials_and_path(url):
    with pytest.raises(ValueError):
        live.loopback_url(url)


def test_readiness_uses_demo_engineer_and_never_inference_or_raw_configuration():
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/auth/login":
            assert json.loads(request.content) == {"username": "engineer", "password": "engineer-demo"}
            return httpx.Response(200, json={"user": {"id": "engineer"}})
        assert request.method == "GET" and request.url.path == "/api/model"
        return httpx.Response(200, json={"configured": True, "provider": "ollama", "model": "mock-model",
                                       "live_verified": True, "api_key": FAKE_KEY,
                                       "base_url": "https://user:password@secret.invalid"})
    value = live.probe_readiness(transport=httpx.MockTransport(handler))
    assert paths == ["/api/auth/login", "/api/model"]
    assert value["status"] == "configuration_status_read"
    assert value["configured"] is True and value["previous_live_verified"] is True
    assert value["model_inference_attempted"] is False
    assert FAKE_KEY not in json.dumps(value) and "password@" not in json.dumps(value)


def test_redirect_error_body_is_never_followed_or_recorded():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(302, text=FAKE_KEY, headers={"Location": "https://credential.invalid/steal"})
    value = live.probe_readiness(transport=httpx.MockTransport(handler))
    assert len(seen) == 1 and value["http_status"] == 302
    assert FAKE_KEY not in json.dumps(value)


def test_explicit_live_flag_is_required_before_config_or_app_loading():
    def forbidden_loader():
        pytest.fail("A dry check read model environment configuration")
    value = live.run_isolated_live(config_loader=forbidden_loader)
    assert value["status"] == "not_run" and value["model_inference_observed"] is False


def test_main_dry_run_writes_truthful_preparation_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(live, "probe_readiness", lambda _: {"status": "configuration_status_read", "configured": True,
                                                         "model_inference_attempted": False})
    from knowtrace import config as config_module
    monkeypatch.setattr(config_module, "model_config_from_env", lambda: pytest.fail("No environment secrets in a dry run"))
    output = tmp_path / "report.json"
    assert live.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["run_live_requested"] is False
    assert report["live_check"]["status"] == "not_run"
    assert report["semantic_verified"] is False
    assert "protocol_checks_passed" not in capsys.readouterr().out


def test_three_isolated_cases_and_private_data_never_reach_mock_provider():
    seen = []
    def handler(request):
        seen.append(request)
        content = json.loads(request.content)
        evidence = json.loads(content["messages"][1]["content"])["evidence"]
        assert evidence and all(item["text"] == live.PUBLIC_TEXT for item in evidence)
        assert live.PRIVATE_MARKER not in request.content.decode()
        return model_response()
    result = live.run_isolated_live(run_live=True, config_loader=config, model_transport=httpx.MockTransport(handler))
    assert result["status"] == "protocol_checks_passed"
    assert result["model_inference_observed"] is True
    assert len(seen) == result["provider_requests_forwarded"] == 1
    assert result["private_case_provider_requests"] == 0
    assert result["private_content_blocked_before_dispatch"] is False
    supported, missing, private = result["cases"]
    assert supported["answer"] == "Snapshots are retained for 14 days. [S1]"
    assert supported["actual_cited_ids"] == ["S1"] and supported["actual_document_ids"]
    assert supported["citation_syntactic_gate"]["status"] == "passed"
    assert missing["refused"] is True and missing["model_invoked"] is False
    assert private["status"] == "permission_denied" and private["http_status"] == 403
    assert all(item["latency_ms"] >= 0 for item in result["cases"])
    assert result["semantic_verified"] is False and FAKE_KEY not in json.dumps(result)


def test_model_refusal_is_reported_and_does_not_count_as_success_or_retry():
    requests = []
    def handler(request):
        requests.append(request)
        return model_response("There is insufficient evidence.", True)
    result = live.run_isolated_live(run_live=True, config_loader=config, model_transport=httpx.MockTransport(handler))
    assert len(requests) == 1
    assert result["status"] == "checks_failed_or_unknown" and result["model_inference_observed"] is True
    assert result["cases"][0]["refused"] is True and result["cases"][0]["actual_cited_ids"] == []


def test_timeout_stays_unknown_and_never_retries_or_records_raw_error():
    requests = []
    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout(FAKE_KEY + " https://user:password@provider.invalid raw-private-body", request=request)
    result = live.run_isolated_live(run_live=True, config_loader=config, model_transport=httpx.MockTransport(handler))
    assert len(requests) == 1 and result["automatic_retries"] == 0
    case = result["cases"][0]
    assert case["status"] == "error_or_unknown" and case["model_invoked"] is None
    assert case["provider_request_outcome"] == "unknown" and case["answer"] is None
    assert result["model_inference_observed"] is None
    encoded = json.dumps(result)
    assert FAKE_KEY not in encoded and "password@" not in encoded and "raw-private-body" not in encoded


def test_outer_cleanup_failure_preserves_observed_provider_dispatch(monkeypatch):
    from fastapi.testclient import TestClient
    original_exit = TestClient.__exit__
    def fail_on_close(self, *args):
        original_exit(self, *args)
        raise RuntimeError("Synthetic failure after completed API cases")
    monkeypatch.setattr(TestClient, "__exit__", fail_on_close)
    result = live.run_isolated_live(run_live=True, config_loader=config,
                                    model_transport=httpx.MockTransport(lambda _: model_response()))
    assert result["status"] == "error_or_unknown"
    assert result["provider_requests_forwarded"] == 1
    assert result["model_inference_observed"] is True


def test_redacts_key_and_url_credentials_even_from_valid_model_answer():
    def handler(request):
        return model_response("Retained 14 days. " + FAKE_KEY + " https://user:password@provider.invalid [S1]")
    result = live.run_isolated_live(run_live=True, config_loader=config, model_transport=httpx.MockTransport(handler))
    assert result["cases"][0]["answer_redacted"] is True
    assert "[REDACTED]" in result["cases"][0]["answer"]
    assert FAKE_KEY not in json.dumps(result) and "password@" not in json.dumps(result)


def test_audit_prevents_synthetic_private_bytes_from_leaving_process():
    def forbidden(request):
        pytest.fail("Private synthetic data reached provider dispatch")
    audit = live.PrivacyAuditTransport(httpx.MockTransport(forbidden))
    with pytest.raises(httpx.RequestError):
        audit.handle_request(httpx.Request("POST", "https://mock.invalid", content=live.PRIVATE_TEXT.encode()))
    assert audit.private_content_blocked is True and audit.forwarded == 0


def test_disabled_model_reports_not_configured_without_transport_call():
    result = live.run_isolated_live(run_live=True, config_loader=ModelConfig,
                                    model_transport=httpx.MockTransport(lambda _: pytest.fail("Disabled model inference")))
    assert result["status"] == "not_configured" and result["model_inference_observed"] is False


def test_citation_gate_does_not_claim_semantic_correctness():
    payload = {"answer": "A semantically unrelated assertion. [S1]", "refused": False,
               "answer_mode": "generated", "sources": [{"citation_id": "S1", "document_id": "public-id", "kb_id": "kb-engineering"}],
               "grounding": {"cited_ids": ["S1"]}}
    gate = live.citation_gate(payload, {"public-id"})
    assert gate["status"] == "passed" and gate["semantic_verified"] is False
    payload["answer"] = "Invalid reference [S99]"
    assert live.citation_gate(payload, {"public-id"})["status"] == "failed"
