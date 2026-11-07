"""Data integrity checks, not a test that all authored questions must pass."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_official_corpus_and_labels_are_reproducible():
    manifest = json.loads((ROOT / "demo/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["commit"]) == 40
    documents = manifest["documents"]
    assert len(documents) == 20
    texts = {}
    for document in documents:
        path = ROOT / "demo/documents" / document["filename"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == document["sha256"]
        assert manifest["commit"] in document["source_url"]
        assert "fastapi/fastapi" in document["source_url"]
        texts[document["filename"]] = raw.decode("utf-8")
    dataset = json.loads((ROOT / "demo/evaluation.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    assert len(cases) == 100
    assert len({case["id"] for case in cases}) == 100
    assert Counter(c["split"] for c in cases) == {"dev": 50, "test": 50}
    expected_counts = Counter()
    for case in cases:
        assert case["question"].strip()
        if case["answerable"]:
            expected_counts[case["expected_document"]] += 1
            assert case["expected_evidence"] in texts[case["expected_document"]]
        else:
            assert case["expected_document"] is None
            assert case["expected_evidence"] is None
    assert len(expected_counts) == 20
    assert set(expected_counts.values()) == {4}
