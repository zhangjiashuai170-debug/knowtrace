import json

from knowtrace import embeddings
from knowtrace.semantic_config import configured_embedder


def test_missing_local_setting_does_not_load_optional_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "LocalEmbedder", lambda *_: (_ for _ in ()).throw(AssertionError("should not load")))
    assert configured_embedder(project_root=tmp_path, environ={}) == (None, False)


def test_only_named_local_setting_supplies_path(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    (tmp_path / "local-settings.json").write_text(json.dumps({"embedding_model_dir": str(directory)}), encoding="utf-8")
    sentinel = object()
    monkeypatch.setattr(embeddings, "LocalEmbedder", lambda path: sentinel if path == directory else None)
    assert configured_embedder(project_root=tmp_path, environ={}) == (sentinel, True)


def test_invalid_or_unavailable_model_keeps_lexical_configuration_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "LocalEmbedder", lambda *_: (_ for _ in ()).throw(RuntimeError("local-private-path")))
    for fields in [{"embedding_model_dir": "https://not-local.invalid/model"},
                   {"embedding_model_dir": str(tmp_path), "unexpected": "value"},
                   {"embedding_model_dir": str(tmp_path)}]:
        (tmp_path / "local-settings.json").write_text(json.dumps(fields), encoding="utf-8")
        assert configured_embedder(project_root=tmp_path, environ={}) == (None, True)


def test_explicit_embedding_environment_does_not_read_unrelated_values(tmp_path, monkeypatch):
    class OnlyEmbedding(dict):
        def get(self, key, default=None):
            assert key == "KNOWTRACE_EMBEDDING_DIR"
            return str(tmp_path)
    monkeypatch.setattr(embeddings, "LocalEmbedder", lambda path: "local-only")
    assert configured_embedder(project_root=tmp_path, environ=OnlyEmbedding()) == ("local-only", True)
