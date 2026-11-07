import json

import httpx

from knowtrace.generation import ModelConfig
from scripts.check_model_live import run_isolated_live


def test_failed_live_preflight_records_only_safe_category_and_http_status():
    config = ModelConfig(provider="openai_compatible", base_url="https://api.deepseek.com",
                         model="deepseek-flash", api_key="fake-test-only-key")
    seen = []
    def rejected(request):
        seen.append(request)
        return httpx.Response(401, text="private-provider-body fake-test-only-key")
    report = run_isolated_live(run_live=True, config_loader=lambda: config,
                               model_transport=httpx.MockTransport(rejected))
    assert len(seen) == 1
    assert report["model_inference_observed"] is None
    assert report["cases"][0]["model_failure_code"] == "authentication_failed"
    assert report["cases"][0]["provider_http_status"] == 401
    assert "private-provider-body" not in json.dumps(report)
    assert "fake-test-only-key" not in json.dumps(report)
    assert report["automatic_retries"] == 0
