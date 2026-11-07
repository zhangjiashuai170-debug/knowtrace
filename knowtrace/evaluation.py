"""Reproducible retrieval / refusal evaluation. Never used to build the index."""
from __future__ import annotations

import json
import hashlib
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .integrity import bind_gold_documents, digest_file


def load_dataset(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        value = {"id": "fastapi-zh-100", "name": "FastAPI 中文文档 · 100题", "cases": value}
    cases = value.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集为空或格式不正确")
    value["_dataset_sha256"] = digest_file(path)
    return value


def dataset_summary(dataset: dict) -> dict:
    cases = dataset["cases"]
    return {
        "id": dataset["id"], "name": dataset.get("name", dataset["id"]),
        "size": len(cases), "answerable": sum(bool(c["answerable"]) for c in cases),
        "unanswerable": sum(not c["answerable"] for c in cases),
        "description": dataset.get("description", "AI 辅助来源核对草稿，尚未经独立人工复核。"),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 3)


def run_evaluation(core, dataset: dict, allowed_kb_ids, mode="hybrid", top_k=5, split=None, *, manifest=None) -> dict:
    """Retrieval metrics on answerable questions; refusal on unanswerable ones.

    Ranking is chunk-based. A document hits when ANY retrieved chunk belongs to
    the expected document. Cutoffs beyond requested top_k are reported as null.
    Latency includes search and evidence assembly, sequential in-process only.
    """
    if mode not in {"bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid"} or not 1 <= top_k <= 10:
        raise ValueError("无效的检索参数")
    selected = [c for c in dataset["cases"] if split is None or c.get("split") == split]
    if not selected:
        raise ValueError("评测分组为空")
    if dataset.get("source_commit") and (manifest is None or dataset["source_commit"] != manifest["commit"]):
        raise ValueError("评测集与语料清单提交不匹配")
    gold = bind_gold_documents(core, allowed_kb_ids, selected, manifest)
    inventory = []
    for document in core.list_documents(allowed_kb_ids):
        source = core.source_path(document["id"], allowed_kb_ids)
        inventory.append({"kb_id": document["kb_id"], "name": document["name"],
                          "sha256": digest_file(source) if source else None,
                          "chunk_count": document["chunk_count"]})
    inventory.sort(key=lambda item: (item["kb_id"], item["name"], item["sha256"] or ""))
    index_sha256 = hashlib.sha256(json.dumps(inventory, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    start = time.perf_counter()
    rows = []
    for case in selected:
        search_start = time.perf_counter()
        hits = core.search(case["question"], allowed_kb_ids, mode=mode, top_k=top_k)
        search_latency = (time.perf_counter() - search_start) * 1000
        case_start = time.perf_counter()
        result = core.ask(case["question"], allowed_kb_ids, mode=mode, top_k=top_k)
        latency = (time.perf_counter() - case_start) * 1000
        names = [h["document_name"] for h in hits]
        ids = [h["document_id"] for h in hits]
        expected = case.get("expected_document")
        rank = next((i for i, identifier in enumerate(ids, 1) if identifier in gold.get(expected, set())), None)
        used_ids = [h["document_id"] for h in result.get("sources", [])]
        gold_used = bool(set(used_ids) & gold.get(expected, set()))
        rows.append({
            "id": case["id"], "question": case["question"], "split": case.get("split"),
            "category": case.get("category"), "expected_document": expected,
            "answerable": bool(case["answerable"]), "retrieved_documents": names,
            "retrieved_document_ids": ids,
            "hit_rank": rank, "refused": bool(result["refused"]),
            "used_source_document_ids": used_ids, "expected_source_used": gold_used,
            "evidence_decision": result.get("evidence_decision"),
            "latency_ms": round(latency, 3),
            "search_latency_ms": round(search_latency, 3),
            "passed": (gold_used and not result["refused"]) if case["answerable"] else bool(result["refused"]),
        })
    answerable = [c for c in rows if c["answerable"]]
    unanswerable = [c for c in rows if not c["answerable"]]
    def fraction(count, total):
        return round(count / total, 4) if total else None
    metrics = {
        f"recall_at_{k}": fraction(sum(c["hit_rank"] is not None and c["hit_rank"] <= k for c in answerable), len(answerable)) if top_k >= k else None
        for k in (1, 3, 5)
    }
    metrics.update({
        "mrr": round(sum(1 / c["hit_rank"] if c["hit_rank"] else 0 for c in answerable) / len(answerable), 4) if answerable else None,
        "refusal_accuracy": fraction(sum(c["refused"] for c in unanswerable), len(unanswerable)),
        "false_refusal_rate": fraction(sum(c["refused"] for c in answerable), len(answerable)),
        "expected_source_use_rate": fraction(sum(c["expected_source_used"] and not c["refused"] for c in answerable), len(answerable)),
        "unsupported_acceptance_rate": fraction(sum(not c["refused"] for c in unanswerable), len(unanswerable)),
        "p50_ms": percentile([c["latency_ms"] for c in rows], .5),
        "p95_ms": percentile([c["latency_ms"] for c in rows], .95),
        "search_p50_ms": percentile([c["search_latency_ms"] for c in rows], .5),
        "search_p95_ms": percentile([c["search_latency_ms"] for c in rows], .95),
    })
    return {
        "id": uuid.uuid4().hex, "app_version": "0.3.0", "dataset_id": dataset["id"], "dataset_name": dataset.get("name"),
        "mode": mode, "top_k": top_k, "split": split or "all", "metrics": metrics, "cases": rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
        "scope": {"allowed_kb_ids": allowed_kb_ids, "case_count": len(rows), "answerable": len(answerable), "unanswerable": len(unanswerable),
                  "document_count": len(inventory), "chunk_count": sum(d["chunk_count"] for d in inventory), "index_sha256": index_sha256},
        "dataset_sha256": dataset.get("_dataset_sha256"),
        "manifest_sha256": manifest.get("_manifest_sha256") if manifest else None,
        "corpus_commit": manifest.get("commit") if manifest else None,
        "retrieval_policy": next((r["evidence_decision"].get("policy_version") for r in rows if r.get("evidence_decision")), "legacy"),
        "retrieval_configuration": core.retrieval_status(),
        "code_sha256": {name: digest_file(Path(__file__).parent / name) for name in ("core.py", "parsing.py", "retrieval.py", "evaluation.py", "integrity.py", "embeddings.py")},
        "methodology": "AI-assisted source-checked draft, awaiting independent human review. Regression split already inspected in v0.1, not an unseen holdout. Chunk ranking from search, independent of answer refusal; expected-document IDs bound through verified file hashes. Case pass requires expected source actually used, not just retrieved. This is source selection, not semantic answer correctness; no LLM judge. Sequential ask latency after search warm-up, not cold start or throughput. search_latency_ms includes the per-case search; first search can include authorized document embedding/cache construction, excluded from ask latency. Refusal accuracy uses unanswerable cases only. Evaluation never calls the answer-generation API.",
    }


def save_run(run: dict, directory: Path, user_id: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = {**run, "owner_user_id": user_id}
    path = directory / f"{run['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
