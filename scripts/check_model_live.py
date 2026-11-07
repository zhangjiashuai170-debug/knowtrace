"""Opt-in live protocol smoke check against an isolated synthetic knowledge base.

Default: read loopback /api/model through the demo engineer API session only.
--run-live: additionally load startup model configuration, create a temporary
in-process API instance, and make at most one inference request per case. This
does not certify semantic correctness or modify the running server's documents.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRIVATE_MARKER = "TEST-ONLY-PRIVATE-ALPHA"
PUBLIC_TEXT = "KnowTraceProbe stores project snapshots for 14 days. KnowTraceProbe exports snapshots as ZIP files."
PRIVATE_TEXT = "SealedMemo recovery passphrase is " + PRIVATE_MARKER + "."
_INLINE = re.compile(r"\[(S[1-9][0-9]{0,2})\]")


def loopback_url(value: str) -> str:
    """Never follow a supplied URL outside an unambiguous loopback origin."""
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        url.port
        local = host.lower() == "localhost"
        if not local:
            local = ipaddress.ip_address(host).is_loopback
        if (not local or url.scheme not in {"http", "https"} or url.username is not None
                or url.password is not None or url.query or url.fragment
                or url.path not in {"", "/"} or "\\" in value
                or any(character.isspace() for character in value)):
            raise ValueError()
        return value.rstrip("/")
    except (ValueError, TypeError):
        raise ValueError("KnowTrace API 地址必须是无凭据、无路径参数的本机 loopback 地址") from None


def safe_text(value, secrets=(), limit=8000) -> str:
    text = value if isinstance(value, str) else ""
    for secret in secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    # Synthetic checks never need URL output; omit all URLs, including userinfo.
    text = re.sub(r"(?i)https?://[^\s<>\"']+", "[URL omitted]", text)
    return "".join(character for character in text[:limit] if ord(character) >= 32 or character in "\n\t")


def probe_readiness(base_url="http://127.0.0.1:8766", *, transport=None) -> dict:
    base_url = loopback_url(base_url)
    try:
        with httpx.Client(base_url=base_url, transport=transport, trust_env=False,
                          follow_redirects=False, timeout=5) as client:
            login = client.post("/api/auth/login", json={"username": "engineer", "password": "engineer-demo"})
            if login.status_code != 200:
                return {"status": "unavailable", "http_status": login.status_code, "model_inference_attempted": False}
            response = client.get("/api/model")
            if response.status_code != 200:
                return {"status": "unavailable", "http_status": response.status_code, "model_inference_attempted": False}
            value = response.json()
            if not isinstance(value, dict) or type(value.get("configured")) is not bool:
                raise ValueError()
            return {"status": "configuration_status_read", "configured": value["configured"],
                    "provider": value.get("provider") if value.get("provider") in {"disabled", "ollama", "openai_compatible"} else "unknown",
                    "model_named": isinstance(value.get("model"), str) and bool(value["model"].strip()),
                    "previous_live_verified": value.get("live_verified") is True,
                    "model_inference_attempted": False,
                    "message": "只核对配置状态，历史成功不代表本次已推理或当前健康。"}
    except Exception:
        # Never persist exception text, headers, response body or provider URLs.
        return {"status": "unknown", "model_inference_attempted": False,
                "message": "无法可靠读取本机配置状态；未重试、未请求推理。"}


class PrivacyAuditTransport(httpx.BaseTransport):
    """Count actual dispatches and block synthetic private content pre-dispatch."""
    def __init__(self, delegate=None):
        self.delegate = delegate
        self.attempted = 0
        self.forwarded = 0
        self.private_content_blocked = False

    def handle_request(self, request):
        self.attempted += 1
        body = request.read()
        if PRIVATE_MARKER.encode() in body or PRIVATE_TEXT.encode() in body:
            self.private_content_blocked = True
            raise httpx.RequestError("Synthetic private content blocked before provider dispatch")
        if self.delegate is None:
            # Created lazily only on the explicitly authorized live path.
            self.delegate = httpx.HTTPTransport(retries=0, trust_env=False)
        self.forwarded += 1
        return self.delegate.handle_request(request)

    def close(self):
        if self.delegate is not None:
            self.delegate.close()


def citation_gate(payload: dict, public_document_ids: set[str]) -> dict:
    """Syntax/ID membership only; intentionally not a semantic-quality judge."""
    answer = payload.get("answer")
    sources = payload.get("sources")
    refused = payload.get("refused")
    inline = list(dict.fromkeys(_INLINE.findall(answer))) if isinstance(answer, str) else []
    valid_sources = isinstance(sources, list) and all(isinstance(source, dict) for source in sources)
    cited = [source.get("citation_id") for source in sources] if valid_sources else []
    residual = _INLINE.sub("", answer) if isinstance(answer, str) else ""
    malformed = bool(re.search(r"[\[［【]\s*[sS]", residual))
    allowed = valid_sources and all(source.get("document_id") in public_document_ids
                                   and source.get("kb_id") == "kb-engineering" for source in sources)
    consistency = (not inline and not cited) if refused is True else bool(inline) and set(inline) == set(cited)
    declared = payload.get("grounding", {}).get("cited_ids") if isinstance(payload.get("grounding"), dict) else None
    passed = (isinstance(answer, str) and bool(answer.strip()) and type(refused) is bool
              and allowed and not malformed and consistency and isinstance(declared, list)
              and declared == inline and payload.get("answer_mode") == "generated")
    return {"status": "passed" if passed else "failed", "inline_ids": inline,
            "source_citation_ids": cited, "semantic_verified": False,
            "scope": "JSON字段、正文引用ID与返回的授权合成文档ID一致性；不核验语义支持。"}


def run_isolated_live(*, run_live=False, config_loader=None, model_transport=None) -> dict:
    if not run_live:
        return {"status": "not_run", "model_inference_observed": False,
                "message": "未提供 --run-live，不加载模型凭据、不推理。"}
    audit = None
    cases = []
    try:
        # Import/loading is deliberately behind the explicit live gate.
        if config_loader is None:
            from knowtrace.config import model_config_from_env
            config_loader = model_config_from_env
        config = config_loader()
        if config.provider == "disabled":
            return {"status": "not_configured", "model_inference_observed": False,
                    "message": "隔离实例未配置模型；未请求推理。"}
        from fastapi.testclient import TestClient
        from knowtrace.app import create_app
        audit = PrivacyAuditTransport(model_transport)
        secrets = (config.api_key,)
        with tempfile.TemporaryDirectory(prefix="knowtrace-live-check-") as temporary:
            directory = Path(temporary)
            app = create_app(directory / "data", model_config=config, model_transport=audit)
            core = app.state.core
            core.create_kb("Synthetic public probe", kb_id="kb-engineering")
            core.create_kb("Synthetic private probe", kb_id="kb-support")
            documents = []
            for name, text, kb in (("public-probe.txt", PUBLIC_TEXT, "kb-engineering"),
                                    ("private-probe.txt", PRIVATE_TEXT, "kb-support")):
                source = directory / name
                source.write_text(text, encoding="utf-8")
                documents.append(core.ingest_file(source, name, kb))
            public_ids = {documents[0]["id"]}
            with TestClient(app, base_url="http://127.0.0.1:8768") as client:
                login = client.post("/api/auth/login", json={"username": "engineer", "password": "engineer-demo"})
                if login.status_code != 200:
                    return {"status": "error", "model_inference_observed": False, "message": "隔离演示登录未通过。"}
                status = client.get("/api/model").json()
                definitions = [
                    ("supported", "How many days does KnowTraceProbe store project snapshots?", ["kb-engineering"]),
                    ("no_evidence", "What is the exact default maximum snapshot size in bytes for KnowTraceProbe?", ["kb-engineering"]),
                    ("unauthorized", "What is the SealedMemo recovery passphrase?", ["kb-support"]),
                ]
                for identifier, question, scope in definitions:
                    before = audit.forwarded
                    started = time.perf_counter()
                    try:
                        response = client.post("/api/ask", json={"question": question, "kb_ids": scope,
                                               "answer_mode": "generated", "mode": "hybrid", "top_k": 3})
                        item = {"id": identifier, "http_status": response.status_code}
                        if response.status_code == 200:
                            payload = response.json()
                            if not isinstance(payload, dict):
                                raise ValueError()
                            invoked = payload.get("model_invoked") is True
                            item.update(status="response", answer=safe_text(payload.get("answer"), secrets),
                                        answer_redacted=safe_text(payload.get("answer"), secrets) != payload.get("answer"),
                                        refused=payload.get("refused") if type(payload.get("refused")) is bool else None,
                                        model_invoked=invoked,
                                        generation_status=payload.get("generation_status") if payload.get("generation_status") == "skipped_insufficient_evidence" else None,
                                        actual_cited_ids=[source.get("citation_id") for source in payload.get("sources", []) if isinstance(source, dict)],
                                        actual_document_ids=[source.get("document_id") for source in payload.get("sources", []) if isinstance(source, dict)],
                                        citation_syntactic_gate=citation_gate(payload, public_ids) if invoked else {"status": "not_run", "semantic_verified": False})
                        elif identifier == "unauthorized" and response.status_code == 403:
                            item.update(status="permission_denied", answer=None, refused=None, model_invoked=False,
                                        actual_cited_ids=[], actual_document_ids=[],
                                        citation_syntactic_gate={"status": "not_run", "semantic_verified": False})
                        else:
                            item.update(status="error_or_unknown", answer=None, refused=None, model_invoked=None,
                                        message="未采用错误响应体；结果不明，不自动重试。",
                                        actual_cited_ids=[], actual_document_ids=[],
                                        citation_syntactic_gate={"status": "not_run", "semantic_verified": False})
                            code = response.headers.get("X-KnowTrace-Model-Error")
                            if code in {"authentication_failed", "quota_exhausted", "rate_limited", "provider_error",
                                        "request_rejected", "timeout", "transport_error", "invalid_model_output", "configuration_error"}:
                                item["model_failure_code"] = code
                            provider_status = response.headers.get("X-KnowTrace-Provider-Status", "")
                            if provider_status.isascii() and provider_status.isdecimal() and len(provider_status) == 3 and 100 <= int(provider_status) <= 599:
                                item["provider_http_status"] = int(provider_status)
                    except Exception:
                        item = {"id": identifier, "status": "error_or_unknown", "answer": None, "refused": None,
                                "model_invoked": None, "actual_cited_ids": [], "actual_document_ids": [],
                                "citation_syntactic_gate": {"status": "not_run", "semantic_verified": False}}
                    item["provider_requests_forwarded"] = audit.forwarded - before
                    item["provider_request_outcome"] = ("unknown" if item["status"] == "error_or_unknown" and audit.forwarded > before
                                                        else "response_observed" if item.get("model_invoked") is True
                                                        else "not_forwarded" if audit.forwarded == before else "unknown")
                    item["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    cases.append(item)
            observed = (True if any(item.get("model_invoked") and item["provider_requests_forwarded"] > 0 for item in cases)
                        else None if audit.forwarded else False)
            checked = (cases[0].get("model_invoked") and cases[0].get("refused") is False
                       and cases[0]["citation_syntactic_gate"]["status"] == "passed"
                       and cases[1].get("refused") is True and cases[1]["provider_requests_forwarded"] == 0
                       and cases[2]["status"] == "permission_denied" and cases[2]["provider_requests_forwarded"] == 0
                       and not audit.private_content_blocked)
            return {"status": "protocol_checks_passed" if checked else "checks_failed_or_unknown",
                    "isolation": "temporary in-process API client; no listening 8768 service, no user data changes",
                    "configuration_source": "explicit startup model configuration for the isolated instance; not copied from running server",
                    "configured": status.get("configured") is True,
                    "provider": config.provider, "model_inference_observed": observed,
                    "provider_requests_forwarded": audit.forwarded,
                    "private_content_blocked_before_dispatch": audit.private_content_blocked,
                    "private_case_provider_requests": cases[2]["provider_requests_forwarded"],
                    "semantic_verified": False, "automatic_retries": 0, "cases": cases}
    except Exception:
        forwarded = audit.forwarded if audit is not None else 0
        observed = (True if any(item.get("model_invoked") and item["provider_requests_forwarded"] > 0 for item in cases)
                    else None if forwarded else False)
        return {"status": "error_or_unknown", "model_inference_observed": observed,
                "provider_requests_forwarded": forwarded, "automatic_retries": 0, "cases": cases,
                "message": "隔离检查未可靠完成；不保存原始错误、配置地址或凭据，不自动重试。"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766", type=loopback_url)
    parser.add_argument("--run-live", action="store_true", help="明确允许隔离实例读取模型配置并请求模型")
    parser.add_argument("--output", type=Path, help="JSON报告文件；默认 artifacts/live-model/时间戳.json")
    args = parser.parse_args(argv)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_live_requested": args.run_live,
              "readiness": probe_readiness(args.base_url),
              "live_check": run_isolated_live(run_live=args.run_live),
              "semantic_verified": False,
              "methodology": "配置不等于健康；未调用不算联调成功；引用ID检查不等于语义正确。仅受控合成资料，无自动重试。"}
    destination = args.output or ROOT / "artifacts" / "live-model" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8] + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(destination.resolve())}, ensure_ascii=False, indent=2))
    if args.run_live:
        return 0 if report["live_check"]["status"] == "protocol_checks_passed" else 2
    return 0 if report["readiness"]["status"] == "configuration_status_read" else 2


if __name__ == "__main__":
    raise SystemExit(main())
