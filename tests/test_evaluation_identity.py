import hashlib
import json

import pytest

from knowtrace.cli import seed
from knowtrace.core import KnowledgeCore
from knowtrace.evaluation import run_evaluation


def test_same_filename_wrong_bytes_does_not_count_as_retrieval_hit(tmp_path):
    core = KnowledgeCore(tmp_path / "data")
    core.create_kb("Engineering", kb_id="kb-engineering")
    original = tmp_path / "original.md"
    original.write_text("# Invoice policy\nInvoices are due thirty days after delivery.", encoding="utf-8")
    official = core.ingest_file(original, "manual.md", "kb-engineering")
    changed = tmp_path / "changed.md"
    changed.write_text("# Counterfeit\nThe counterfeit code is ZX999.", encoding="utf-8")
    fake = core.ingest_file(changed, "manual.md", "kb-engineering")
    manifest = {"commit": "a" * 40, "documents": [{"filename": "manual.md", "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}]}
    dataset = {"id": "identity", "cases": [{"id": "1", "question": "counterfeit code ZX999", "expected_document": "manual.md", "answerable": True}]}
    report = run_evaluation(core, dataset, ["kb-engineering"], "bm25", 1, manifest=manifest)
    assert report["cases"][0]["retrieved_document_ids"] == [fake["id"]]
    assert report["cases"][0]["hit_rank"] is None
    assert report["metrics"]["recall_at_1"] == 0
    core.delete_document(official["id"], None)
    with pytest.raises(ValueError, match="哈希核验"):
        run_evaluation(core, dataset, ["kb-engineering"], manifest=manifest)


def test_seed_verifies_hashes_before_creating_store(tmp_path):
    demo = tmp_path / "demo"
    (demo / "documents").mkdir(parents=True)
    (demo / "documents" / "manual.md").write_text("tampered", encoding="utf-8")
    (demo / "manifest.json").write_text(json.dumps({"documents": [{"filename": "manual.md", "sha256": "0" * 64}]}), encoding="utf-8")
    data_dir = tmp_path / "data"
    with pytest.raises(ValueError, match="校验失败"):
        seed(data_dir, demo)
    assert not data_dir.exists()
