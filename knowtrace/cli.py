from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .core import KnowledgeCore
from .evaluation import load_dataset, run_evaluation, save_run
from .integrity import read_manifest, verify_corpus_files

ROOT = Path(__file__).resolve().parent.parent


def seed(data_dir: Path, demo_dir: Path = ROOT / "demo") -> dict:
    manifest = verify_corpus_files(demo_dir)
    core = KnowledgeCore(data_dir)
    existing = {k["id"] for k in core.list_kbs(None)}
    for kb_id, name, description in [
        ("kb-engineering", "工程文档", "官方 FastAPI 中文文档；公开演示语料"),
        ("kb-support", "客户支持", "合成支持手册，仅用于权限隔离演示"),
    ]:
        if kb_id not in existing:
            core.create_kb(name, description, kb_id)
    imported = duplicate = 0
    for document in manifest["documents"]:
        name = document["filename"]
        result = core.ingest_file(demo_dir / "documents" / name, name, "kb-engineering", document["source_url"], "MIT")
        duplicate += bool(result.get("duplicate"))
        imported += not result.get("duplicate", False)
    support = demo_dir / "support-demo.txt"
    if not support.exists():
        raise ValueError("缺少明确标注的合成支持手册")
    core.ingest_file(support, support.name, "kb-support", license_name="CC0-1.0 (synthetic demo)")
    return {"imported": imported, "duplicates": duplicate, "knowledge_bases": core.list_kbs(None)}


def main():
    parser = argparse.ArgumentParser(description="KnowTrace 离线证据检索")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed", help="幂等导入公开演示文档，不删除现有上传")
    serve = commands.add_parser("serve", help="仅在本机127.0.0.1启动")
    serve.add_argument("--port", type=int, default=8766)
    evaluate = commands.add_parser("evaluate", help="命令行复跑评测")
    evaluate.add_argument("--mode", choices=["bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid", "all"], default="all")
    evaluate.add_argument("--split", choices=["dev", "test", "all"], default="all")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "evaluations")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.command == "seed":
        print(json.dumps(seed(args.data_dir), ensure_ascii=False, indent=2))
    elif args.command == "serve":
        import uvicorn
        from .app import create_app
        from .semantic_config import configured_embedder
        embedder, configured = configured_embedder()
        uvicorn.run(create_app(args.data_dir, embedder=embedder, retrieval_configured=configured), host="127.0.0.1", port=args.port, log_level="info")
    else:
        dataset = load_dataset(ROOT / "demo" / "evaluation.json")
        from .semantic_config import configured_embedder
        embedder, _ = configured_embedder() if args.mode in {"semantic", "semantic_hybrid", "all"} else (None, False)
        core = KnowledgeCore(args.data_dir, embedder=embedder)
        modes = ["bm25", "tfidf", "hybrid"] + (["semantic", "semantic_hybrid"] if embedder is not None else [])
        for mode in (modes if args.mode == "all" else [args.mode]):
            run = run_evaluation(core, dataset, ["kb-engineering"], mode, args.top_k, None if args.split == "all" else args.split, manifest=read_manifest(ROOT / "demo"))
            path = save_run(run, args.output_dir, "cli")
            print(json.dumps({"mode": mode, "metrics": run["metrics"], "report": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
