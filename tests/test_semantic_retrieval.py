"""Synthetic vector contracts; no model files, optional runtime or network."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from knowtrace.core import KnowledgeCore
from knowtrace.embeddings import SemanticError


class FakeEmbedder:
    def __init__(self, fingerprint="test-model-v1", dimension=3):
        self.fingerprint = fingerprint
        self.dimension = dimension
        self.document_calls = []
        self.query_calls = []
        self.metadata = {"model_id": "synthetic/test-only", "dimension": dimension,
                         "provider": "local-test", "pooling": "mean", "local_only": True,
                         "model_path": "C:/private/model", "cache_dir": "/private/cache"}

    def vector(self, text):
        axis = 1 if "invoices" in text.lower() else 0
        return [1.0 if i == axis else 0.0 for i in range(self.dimension)]

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self.vector(text)


@pytest.fixture
def fixture(tmp_path):
    embedder = FakeEmbedder()
    core = KnowledgeCore(tmp_path / "data", embedder=embedder)
    core.create_kb("Public", kb_id="public")
    core.create_kb("Private", kb_id="private")
    return core, embedder, tmp_path


def add(fixture, text, kb="public", name=None):
    core, _, root = fixture
    name = name or f"document-{len(list(root.glob('*.txt')))}.txt"
    source = root / name
    source.write_text(text, encoding="utf-8")
    return core.ingest_file(source, name, kb)


def cache_rows(core):
    with closing(core._connect()) as db:
        return [dict(row) for row in db.execute("SELECT * FROM chunk_embeddings ORDER BY chunk_id,model_fingerprint")]


def test_lexical_modes_never_access_or_invoke_embedding_runtime(tmp_path):
    class Untouchable:
        def __getattribute__(self, name):
            raise AssertionError("Lexical search touched embedding runtime")
    core = KnowledgeCore(tmp_path / "data", embedder=Untouchable())
    core.create_kb("Public", kb_id="public")
    source = tmp_path / "orchid.txt"
    source.write_text("Orchid roots need moisture.", encoding="utf-8")
    core.ingest_file(source, source.name, "public")
    for mode in ("bm25", "tfidf", "hybrid"):
        assert core.search("orchid roots", ["public"], mode)
        assert not core.ask("orchid roots", ["public"], mode)["refused"]
    assert cache_rows(core) == []


@pytest.mark.parametrize("mode", ["semantic", "semantic_hybrid"])
def test_missing_model_is_explicit_not_lexical_fallback(tmp_path, mode):
    core = KnowledgeCore(tmp_path / "data")
    with pytest.raises(SemanticError):
        core.search("orchid", None, mode)


def test_paraphrase_is_a_candidate_but_similarity_does_not_admit_evidence(fixture):
    expected = add(fixture, "Orchids flourish with moisture.")
    add(fixture, "Invoices record yearly expenses.")
    core, encoder, _ = fixture
    hits = core.search("hydrating blossoms", ["public"], "semantic")
    assert hits[0]["document_id"] == expected["id"]
    assert hits[0]["semantic_score"] == 1
    assert hits[0]["coverage"] == 0
    result = core.ask("hydrating blossoms", ["public"], "semantic")
    assert result["refused"] is True and result["sources"] == []
    assert result["retrieval_candidates"][0]["document_id"] == expected["id"]
    assert result["evidence_decision"]["policy_version"] == "lexical-v2"
    assert all("citation_id" not in hit for hit in result["retrieval_candidates"])
    assert len(encoder.document_calls) == 1


def test_semantic_query_with_no_lexical_tokens_still_returns_candidates(fixture):
    expected = add(fixture, "Orchids flourish with moisture.")
    core, _, _ = fixture
    assert core.search("🌱", ["public"], "semantic")[0]["document_id"] == expected["id"]
    assert core.ask("🌱", ["public"], "semantic")["refused"] is True


@pytest.mark.parametrize("mode", ["semantic", "semantic_hybrid"])
def test_empty_acl_invokes_no_document_or_query_encoding(fixture, mode):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    assert core.search("orchid roots", [], mode) == []
    assert core.ask("orchid roots", [], mode)["retrieval_candidates"] == []
    assert encoder.document_calls == [] and encoder.query_calls == []
    assert cache_rows(core) == []


@pytest.mark.parametrize("mode", ["semantic", "semantic_hybrid"])
def test_authorization_precedes_encoding_statistics_and_ranking(fixture, mode):
    public = add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    before = core.search("orchid roots", ["public"], mode)
    private = add(fixture, "Orchid roots orchid roots PRIVATE-RECOVERY-TOKEN", "private")
    after = core.search("orchid roots", ["public"], mode)
    assert before == after
    assert after[0]["document_id"] == public["id"]
    assert private["id"] not in json.dumps(after)
    assert all("PRIVATE-RECOVERY-TOKEN" not in text for call in encoder.document_calls for text in call)
    with closing(core._connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM chunk_embeddings e JOIN chunks c ON e.chunk_id=c.id WHERE c.kb_id='private'").fetchone()[0] == 0


def test_public_status_contains_only_safe_model_metadata_and_no_encoding(fixture):
    core, encoder, _ = fixture
    value = core.retrieval_status()
    assert value["available"] and value["configured"]
    assert value["modes"] == ["bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid"]
    assert value["fingerprint"] == encoder.fingerprint and value["dimension"] == 3
    assert value["model"]["model_id"] == "synthetic/test-only"
    assert "private" not in json.dumps(value)
    assert encoder.document_calls == [] and encoder.query_calls == []
    assert KnowledgeCore(core.data_dir).retrieval_status()["modes"] == ["bm25", "tfidf", "hybrid"]


def test_durable_vectors_reused_across_core_instances_without_query_persistence(fixture):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    before = core.search("orchid roots", ["public"], "semantic")
    assert len(encoder.document_calls) == 1
    replacement = FakeEmbedder()
    reopened = KnowledgeCore(core.data_dir, embedder=replacement)
    assert reopened.search("orchid roots", ["public"], "semantic") == before
    assert replacement.document_calls == []
    reopened.search("UNIQUE-QUERY-NEVER-PERSIST-947", ["public"], "semantic")
    with closing(core._connect()) as db:
        assert "UNIQUE-QUERY-NEVER-PERSIST-947" not in "\n".join(db.iterdump())


def test_ingest_and_cross_instance_delete_invalidate_dense_rows_and_cascade_cache(fixture):
    first = add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    core.search("orchid", ["public"], "semantic")
    second_encoder = FakeEmbedder()
    second_core = KnowledgeCore(core.data_dir, embedder=second_encoder)
    second_core.search("orchid", ["public"], "semantic")
    second = add(fixture, "Invoices record yearly expenses.")
    hits = second_core.search("invoices", ["public"], "semantic")
    assert hits[0]["document_id"] == second["id"]
    assert second_encoder.document_calls == [["Invoices record yearly expenses."]]
    assert core.delete_document(first["id"], ["public"])
    assert all(hit["document_id"] != first["id"] for hit in second_core.search("orchid", ["public"], "semantic"))
    with closing(core._connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0] == 1
    assert second_core.delete_document(second["id"], ["public"])
    assert core.search("orchid", ["public"], "semantic") == []
    assert cache_rows(core) == []


def test_model_fingerprint_and_dimension_changes_recompute_vectors(fixture):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    core.search("orchid", ["public"], "semantic")
    encoder.fingerprint = "test-model-v2"
    core.search("orchid", ["public"], "semantic")
    assert len(encoder.document_calls) == 2 and len(cache_rows(core)) == 2
    encoder.dimension = 2
    core.search("orchid", ["public"], "semantic")
    assert len(encoder.document_calls) == 3
    assert next(row for row in cache_rows(core) if row["model_fingerprint"] == "test-model-v2")["dimension"] == 2


def test_changed_chunk_text_recomputes_cached_content_hash(fixture):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    core.search("orchid", ["public"], "semantic")
    before = cache_rows(core)[0]["text_sha256"]
    with closing(core._connect()) as db, db:
        db.execute("UPDATE chunks SET text='Invoices record yearly expenses.'")
        db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")
    hits = core.search("invoices", ["public"], "semantic")
    assert hits[0]["semantic_score"] == 1
    assert encoder.document_calls[-1] == ["Invoices record yearly expenses."]
    assert cache_rows(core)[0]["text_sha256"] != before


@pytest.mark.parametrize("encoded", ["{broken", "null", "[NaN,0,1]", "[0,0,0]", "[1,0]", "[2,0,0]", "[true,0,0]"])
def test_damaged_durable_cache_fails_without_silent_recompute_or_fallback(fixture, encoded):
    add(fixture, "Orchid roots need moisture.")
    core, _, _ = fixture
    core.search("orchid", ["public"], "semantic")
    with closing(core._connect()) as db, db:
        db.execute("UPDATE chunk_embeddings SET vector_json=?", (encoded,))
    replacement = FakeEmbedder()
    reopened = KnowledgeCore(core.data_dir, embedder=replacement)
    with pytest.raises(SemanticError):
        reopened.search("orchid", ["public"], "semantic")
    assert replacement.document_calls == [] and replacement.query_calls == []
    assert reopened.search("orchid", ["public"], "bm25")


@pytest.mark.parametrize("value", [[], [[0, 0, 0]], [[float('inf'), 0, 0]], [[1, 0]], [[True, 0, 0]]])
def test_invalid_runtime_document_vectors_are_not_persisted(fixture, value):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    encoder.embed_documents = lambda _: value
    with pytest.raises(SemanticError):
        core.search("orchid", ["public"], "semantic")
    assert cache_rows(core) == []


def test_encoder_error_is_sanitized_and_lexical_remains_usable(fixture):
    add(fixture, "Orchid roots need moisture.")
    core, encoder, _ = fixture
    def failed(_):
        raise RuntimeError("C:/private/model secret-marker")
    encoder.embed_documents = failed
    with pytest.raises(SemanticError) as caught:
        core.search("orchid", ["public"], "semantic_hybrid")
    assert "private" not in str(caught.value) and "secret-marker" not in str(caught.value)
    assert core.search("orchid", ["public"], "hybrid")


def test_semantic_rrf_uses_three_lists_while_legacy_hybrid_stays_two(fixture):
    dense_first = add(fixture, "Orchid roots need moisture.")
    lexical_first = add(fixture, "Invoices record yearly expenses.")
    core, encoder, _ = fixture
    encoder.embed_query = lambda _: [1, 0, 0]
    assert core.search("invoices", ["public"], "semantic")[0]["document_id"] == dense_first["id"]
    old = core.search("invoices", ["public"], "hybrid")
    new = core.search("invoices", ["public"], "semantic_hybrid")
    assert old[0]["document_id"] == lexical_first["id"]
    assert new[0]["document_id"] == lexical_first["id"]
    assert new[0]["score"] == round(2 / 61 + 1 / 62, 8)
    assert next(hit for hit in new if hit["document_id"] == dense_first["id"])["score"] == round(1 / 61, 8)
    assert "semantic_score" not in old[0]


def test_numeric_and_negation_constraints_remain_in_force_with_high_dense_score(fixture):
    add(fixture, "Orchid releases launch on 2026-05-12. Orchid roots require water.")
    core, _, _ = fixture
    for query in ("Orchid release 2029-05-12", "Do orchid roots not require water?"):
        result = core.ask(query, ["public"], "semantic")
        assert result["refused"] is True and result["sources"] == []
        assert result["retrieval_candidates"][0]["semantic_score"] == 1
