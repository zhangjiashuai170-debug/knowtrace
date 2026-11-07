import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowtrace.app import create_app


@pytest.fixture
def app(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "evaluation.json").write_text(json.dumps({
        "id": "test", "name": "Test", "cases": [
            {"id": "a", "question": "如何设置星光超时为37秒？", "expected_document": "engineering.txt", "answerable": True, "split": "test"},
            {"id": "b", "question": "银河大象的饼干价格是多少？", "expected_document": None, "answerable": False, "split": "test"},
        ]}), encoding="utf-8")
    value = create_app(tmp_path / "data", demo)
    core = value.state.core
    core.create_kb("Engineering", kb_id="kb-engineering")
    core.create_kb("Support", kb_id="kb-support")
    for filename, kb, text in [
        ("engineering.txt", "kb-engineering", "# 星光配置\n星光超时设置为37秒。配置重载后生效。"),
        ("support.txt", "kb-support", "# 支持隔离\nSUPPORT-ORCHID-742 专属支持密语。退款期限七天。"),
    ]:
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        core.ingest_file(path, filename, kb)
    return value


def login(client, name):
    response = client.post("/api/auth/login", json={"username": name, "password": name + "-demo"})
    assert response.status_code == 200


def test_login_signed_session_and_logout(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/auth/me").status_code == 401
        assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
        assert client.post("/api/auth/login", json={"username": "admin", "password": "密码错误"}).status_code == 401
        login(client, "engineer")
        assert client.get("/api/auth/me").json()["user"]["kb_ids"] == ["kb-engineering"]
        cookie = client.cookies.get("knowtrace_session")
        client.cookies.clear()
        client.cookies.set("knowtrace_session", cookie + "bad")
        assert client.get("/api/auth/me").status_code == 401
        login(client, "admin")
        client.post("/api/auth/logout")
        assert client.get("/api/auth/me").status_code == 401


def test_acl_applies_to_ranking_source_chunks_delete_and_scope(app):
    support_doc = next(d for d in app.state.core.list_documents(None) if d["kb_id"] == "kb-support")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        login(client, "engineer")
        assert len(client.get("/api/kbs").json()["items"]) == 1
        items = client.post("/api/search", json={"question": "SUPPORT-ORCHID-742", "mode": "hybrid"}).json()["items"]
        assert all(h["kb_id"] == "kb-engineering" for h in items)
        for suffix in ("source", "chunks"):
            assert client.get(f"/api/documents/{support_doc['id']}/{suffix}").status_code == 404
        assert client.delete(f"/api/documents/{support_doc['id']}").status_code == 404
        assert client.post("/api/ask", json={"question": "退款期限", "kb_ids": ["kb-support"]}).status_code == 403
        empty = client.post("/api/search", json={"question": "星光超时", "kb_ids": []}).json()
        assert empty["items"] == []
        assert client.post("/api/kbs", json={"name": "new"}).status_code == 403
        assert client.post("/api/documents", data={"kb_id": "kb-support"}, files={"file": ("a.txt", b"test")}).status_code == 403


def test_upload_duplicate_download_delete_and_validation(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        login(client, "admin")
        for _ in range(2):
            response = client.post("/api/documents", data={"kb_id": "kb-engineering"}, files={"file": ("../safe.txt", "霓虹备份编号为N329。".encode())})
            assert response.status_code == 201
            doc = response.json()["document"]
        assert doc["duplicate"] is True
        assert "path" not in doc
        assert doc["name"] == "safe.txt"
        response = client.get(f"/api/documents/{doc['id']}/source")
        assert "attachment" in response.headers["content-disposition"]
        assert "N329" in response.text
        assert client.post("/api/documents", data={"kb_id": "kb-engineering"}, files={"file": ("bad.exe", b"abc")}).status_code == 400
        assert client.post("/api/documents", data={"kb_id": "kb-engineering"}, files={"file": ("bad.pdf", b"bad pdf")}).status_code == 400
        assert client.post("/api/ask", json={"question": " ", "top_k": 5}).status_code == 400
        assert client.post("/api/ask", json={"question": "x", "top_k": 99}).status_code == 422
        assert client.delete(f"/api/documents/{doc['id']}").status_code == 200
        assert client.get(f"/api/documents/{doc['id']}/source").status_code == 404


def test_evaluation_runs_are_actual_and_scoped(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        login(client, "engineer")
        run = client.post("/api/evaluations/run", json={"dataset_id": "test", "mode": "hybrid", "top_k": 5})
        assert run.status_code == 200
        data = run.json()
        assert len(data["cases"]) == 2
        assert data["metrics"]["recall_at_1"] == 1
        assert data["metrics"]["p95_ms"] >= data["metrics"]["p50_ms"] >= 0
        assert client.get(f"/api/evaluations/runs/{data['id']}/download").status_code == 200
        login(client, "support")
        assert client.get("/api/evaluations/runs").json()["items"] == []
        assert client.get(f"/api/evaluations/runs/{data['id']}").status_code == 404
        assert client.post("/api/evaluations/run", json={"dataset_id": "test"}).status_code == 409
        login(client, "admin")
        assert len(client.get("/api/evaluations/runs").json()["items"]) == 1


def test_cross_origin_mutation_blocked(app):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/auth/login", headers={"Origin": "https://untrusted.example"}, json={"username": "admin", "password": "admin-demo"}).status_code == 403
    with TestClient(app, base_url="http://rebinding.invalid") as client:
        assert client.post("/api/auth/login", headers={"Origin": "http://rebinding.invalid"}, json={"username": "admin", "password": "admin-demo"}).status_code == 400
