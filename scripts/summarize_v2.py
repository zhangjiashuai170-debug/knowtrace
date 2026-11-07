"""Produce a comparison from immutable v0.1 reports and the frozen v0.2 run."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent


def main():
    old = {r["mode"]: r for path in (ROOT / "docs/reports").glob("*.json") if (r := json.loads(path.read_text(encoding="utf-8"))).get("split") == "test"}
    new = {r["mode"]: r for path in (ROOT / "docs/reports-v0.2").glob("*.json") if (r := json.loads(path.read_text(encoding="utf-8"))).get("split") == "test"}
    assert set(old) == set(new) == {"bm25", "tfidf", "hybrid"}
    notes = json.loads((ROOT / "docs/retrieval-dev-notes.json").read_text(encoding="utf-8"))
    assert notes["final"]["frozen"] is True
    comparison = {
        "status": "frozen-regression-comparison",
        "limits": ["Already inspected regression split, not unseen holdout", "AI-assisted labels, independent human review pending", "Source retrieval/use does not verify semantic answer correctness", "v0.1 sources included unquoted candidates; source-use rate has no comparable v0.1 baseline", "Local warm sequential timing, not throughput"],
        "development": notes,
        "modes": {mode: {"before": {"run_id": old[mode]["id"], "metrics": old[mode]["metrics"]}, "after": {"run_id": new[mode]["id"], "metrics": new[mode]["metrics"], "scope": new[mode]["scope"], "code_sha256": new[mode]["code_sha256"]}} for mode in old},
    }
    (ROOT / "docs/comparison-v0.2.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"hybrid_before": old["hybrid"]["metrics"], "hybrid_after": new["hybrid"]["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
