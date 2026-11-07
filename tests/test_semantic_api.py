import json

from fastapi.testclient import TestClient

from knowtrace.app import create_app
from knowtrace.embeddings import SemanticError
from knowtrace.generation import ModelConfig


class FakeEncoder:
    dimension = 3
    fingerprint = "unit-test-semantic-v1"
    metadata = {"model_id": "fake-local-encoder", "dimension": 3}

    def embed_documents(self, texts):
        return [[1., 0., 0.] for _ in texts]

    def embed_query(self, text):
        return [1., 0., 0.]


def make_client(tmp_path, embedder=None):
    app = create_app(tmp_path / "data", model_config=ModelConfig(), embedder=embedder)
    core = app.state.core
    core.create_kb("Engineering", kb_id="kb-engineering")
    core.create_kb("Support", kb_id="kb-support")
    source = tmp_path / "source.txt"
    source.write_text("Starling stores backups for 14 days.", encoding="utf-8")
    core.ingest_file(source, source.name, "kb-engineering")
    client = TestClient(app, base_url="http://127.0.0.1")
    return client


def login(client):
    client.post("/api/auth/login", json={"username": "engineer", "password": "engineer-demo"}).raise_for_status()


def test_semantic_configuration_requires_login_and_disabled_mode_is_explicit(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/api/retrieval").status_code == 401
    login(client)
    status = client.get("/api/retrieval").json()
    assert status["available"] is False
    for mode in ["semantic", "semantic_hybrid"]:
        assert client.post("/api/search", json={"question": "backups", "mode": mode}).status_code == 409
        assert client.post("/api/ask", json={"question": "backups", "mode": mode}).status_code == 409
    assert client.post("/api/search", json={"question": "backups"}).status_code == 200


def test_local_candidates_do_not_turn_zero_overlap_into_answer(tmp_path):
    client = make_client(tmp_path, FakeEncoder())
    login(client)
    assert client.get("/api/retrieval").json()["available"] is True
    search = client.post("/api/search", json={"question": "renewal apple harbor", "mode": "semantic"})
    assert search.status_code == 200 and search.json()["items"]
    answer = client.post("/api/ask", json={"question": "renewal apple harbor", "mode": "semantic"})
    assert answer.status_code == 200
    assert answer.json()["refused"] is True and answer.json()["sources"] == []
    assert client.post("/api/search", json={"question": "backups", "mode": "semantic", "kb_ids": ["kb-support"]}).status_code == 403


def test_embedding_failure_is_sanitized_and_lexical_still_works(tmp_path):
    class Broken(FakeEncoder):
        def embed_documents(self, texts):
            raise SemanticError("private-path-and-query")
    client = make_client(tmp_path, Broken())
    login(client)
    response = client.post("/api/search", json={"question": "backups", "mode": "semantic"})
    assert response.status_code == 503 and "private-path-and-query" not in response.text
    assert client.post("/api/search", json={"question": "backups", "mode": "bm25"}).status_code == 200
