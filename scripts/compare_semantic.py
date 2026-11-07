"""Frozen retrieval comparison on an isolated, hash-verified public corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowtrace.cli import seed
from knowtrace.core import KnowledgeCore
from knowtrace.evaluation import load_dataset, run_evaluation, save_run
from knowtrace.integrity import read_manifest
from knowtrace.semantic_config import configured_embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.data_dir.resolve() == (ROOT / "data").resolve():
        raise ValueError("Use a separate public benchmark directory")
    seed(args.data_dir)
    encoder, _ = configured_embedder()
    if encoder is None:
        raise ValueError("Prepare the local semantic model first")
    core = KnowledgeCore(args.data_dir, embedder=encoder)
    dataset = load_dataset(ROOT / "demo/evaluation.json")
    manifest = read_manifest(ROOT / "demo")
    comparisons = {}
    for split in ("dev", "test"):
        comparisons[split] = {}
        for mode in ("bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid"):
            run = run_evaluation(core, dataset, ["kb-engineering"], mode, 5, split, manifest=manifest)
            destination = save_run(run, args.output_dir / split, "local-benchmark")
            comparisons[split][mode] = {"metrics": run["metrics"], "scope": run["scope"],
                                       "report": str(destination.relative_to(args.output_dir)),
                                       "code_sha256": run["code_sha256"]}
    summary = {"app_version": "0.3.0", "embedding": encoder.metadata,
               "splits": comparisons, "generated_api_calls": 0, "default_changed": False,
               "note": "Fixed k=60 RRF; no parameter tuning on results. Existing test split was already inspected: regression comparison, not untouched holdout. Dense similarity ranks candidates; lexical evidence admission is unchanged. Latencies are sequential on this machine; no throughput claim."}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "generated_api_calls": 0,
                      "test": {mode: value["metrics"] for mode, value in comparisons["test"].items()}}, indent=2))


if __name__ == "__main__":
    main()
