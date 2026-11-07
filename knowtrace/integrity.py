"""Bind public corpus provenance and scoring to bytes, not filenames alone."""
import hashlib
import json
from pathlib import Path


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(demo_dir: Path) -> dict | None:
    path = demo_dir / "manifest.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_manifest_sha256"] = digest_file(path)
    return value


def verify_corpus_files(demo_dir: Path) -> dict:
    manifest = read_manifest(demo_dir)
    if not manifest or not manifest.get("documents"):
        raise ValueError("缺少演示语料清单")
    for document in manifest["documents"]:
        name = document["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("语料清单文件名无效")
        path = demo_dir / "documents" / name
        if not path.is_file() or digest_file(path) != document["sha256"]:
            raise ValueError(f"演示语料校验失败：{name}，未导入任何文件")
    license_path = demo_dir / "LICENSE"
    if not license_path.is_file() or digest_file(license_path) != manifest["license_sha256"]:
        raise ValueError("上游许可证校验失败")
    return manifest


def bind_gold_documents(core, allowed, cases: list[dict], manifest: dict | None) -> dict[str, set[str]]:
    expected_names = {c["expected_document"] for c in cases if c["answerable"]}
    hashes = {d["filename"]: d["sha256"] for d in manifest["documents"]} if manifest else None
    result = {name: set() for name in expected_names}
    for document in core.list_documents(allowed):
        name = document["name"]
        if name not in result:
            continue
        source = core.source_path(document["id"], allowed)
        if source is None or not source.is_file():
            continue
        if hashes is not None and digest_file(source) != hashes.get(name):
            continue
        result[name].add(document["id"])
    missing = sorted(name for name, ids in result.items() if not ids)
    if missing:
        raise ValueError(f"当前权限或知识库缺少经过哈希核验的评测文档：{', '.join(missing[:4])}。请先导入完整演示语料并使用工程或管理员账号。")
    if hashes is None and any(len(ids) > 1 for ids in result.values()):
        raise ValueError("自定义评测集存在同名文档歧义，请提供带内容哈希的manifest")
    return result
