"""Acquire pinned public FastAPI docs, or verify the bundled demo offline.

Network access happens only with --download. No upstream code is executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
DOCUMENTS = DEMO / "documents"
REPOSITORY = "https://github.com/fastapi/fastapi"
COMMIT = "8e3db5153ec314d6e7775cd0ea39a46e95bfddb0"
RAW_BASE = "https://raw.githubusercontent.com/fastapi/fastapi/" + COMMIT + "/"
SLUGS = (
    "first-steps", "path-params", "query-params", "body",
    "query-params-str-validations", "path-params-numeric-validations",
    "body-multiple-params", "body-fields", "body-nested-models",
    "extra-data-types", "cookie-params", "header-params", "response-model",
    "extra-models", "response-status-code", "request-forms", "request-files",
    "static-files", "handling-errors", "cors",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(source_path: str) -> bytes:
    request = urllib.request.Request(
        RAW_BASE + source_path,
        headers={"User-Agent": "KnowTrace-public-demo-corpus/1.0"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("unexpected_source_size")
    data.decode("utf-8")
    return data


def acquire() -> None:
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    manifest_path = DEMO / "manifest.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["repository"] != REPOSITORY or previous["commit"] != COMMIT:
            raise ValueError("refusing_to_replace_a_different_pinned_corpus")
    known_hashes = {row["filename"]: row["sha256"]
                    for row in (previous or {}).get("documents", [])}
    rows = []
    for slug in SLUGS:
        source_path = "docs/zh/docs/tutorial/" + slug + ".md"
        filename = slug + ".md"
        data = fetch(source_path)
        digest = sha256(data)
        if filename in known_hashes and known_hashes[filename] != digest:
            raise ValueError("pinned_source_hash_changed: " + filename)
        text = data.decode("utf-8")
        (DOCUMENTS / filename).write_bytes(data)
        title = next((line.lstrip("# ").strip() for line in text.splitlines()
                      if line.startswith("# ")), slug)
        rows.append({
            "filename": filename, "title": title, "language": "zh",
            "source_path": source_path, "source_url": RAW_BASE + source_path,
            "browse_url": REPOSITORY + "/blob/" + COMMIT + "/" + source_path,
            "commit": COMMIT, "sha256": digest, "bytes": len(data),
            "license_name": "MIT",
        })
    license_data = fetch("LICENSE")
    license_digest = sha256(license_data)
    if previous and previous["license_sha256"] != license_digest:
        raise ValueError("pinned_license_hash_changed")
    (DEMO / "LICENSE").write_bytes(license_data)
    manifest = {
        "id": "fastapi-zh-tutorial-20", "repository": REPOSITORY,
        "commit": COMMIT, "source_ref_description": "release 0.162.1 commit",
        "license": "MIT", "license_file": "LICENSE",
        "license_source_url": RAW_BASE + "LICENSE", "license_sha256": license_digest,
        "acquired_at": (previous or {}).get("acquired_at")
        or datetime.now(timezone.utc).isoformat(),
        "content_policy": "Unmodified upstream UTF-8 bytes; public documentation only.",
        "documents": rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"downloaded_documents": len(rows), "commit": COMMIT,
                      "license_sha256": license_digest}, ensure_ascii=False))


def verify() -> dict:
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    if manifest["repository"] != REPOSITORY or manifest["commit"] != COMMIT:
        raise ValueError("manifest_source_mismatch")
    if manifest["license"] != "MIT" or manifest["license_source_url"] != RAW_BASE + "LICENSE":
        raise ValueError("license_provenance_mismatch")
    license_data = (DEMO / "LICENSE").read_bytes()
    if sha256(license_data) != manifest["license_sha256"]:
        raise ValueError("license_hash_mismatch")
    if "MIT License" not in license_data.decode("utf-8"):
        raise ValueError("unexpected_license_text")
    expected_names = {slug + ".md" for slug in SLUGS}
    if len(manifest["documents"]) != 20 or {row["filename"] for row in manifest["documents"]} != expected_names:
        raise ValueError("corpus_document_set_mismatch")
    texts = {}
    for row in manifest["documents"]:
        data = (DOCUMENTS / row["filename"]).read_bytes()
        source_path = "docs/zh/docs/tutorial/" + row["filename"]
        if (sha256(data) != row["sha256"] or len(data) != row["bytes"]
                or row["commit"] != COMMIT or row["source_path"] != source_path
                or row["source_url"] != RAW_BASE + source_path):
            raise ValueError("document_integrity_mismatch: " + row["filename"])
        texts[row["filename"]] = data.decode("utf-8")
    result = {"documents": 20, "all_document_hashes_verified": True,
              "license_hash_verified": True, "pinned_commit": COMMIT}
    evaluation_path = DEMO / "evaluation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError("evaluation.json is required for the complete demo check")
    dataset = json.loads(evaluation_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if dataset["id"] != "fastapi-zh-100" or len(cases) != 100:
        raise ValueError("evaluation_size_or_identity_mismatch")
    if len({case["id"] for case in cases}) != 100 or len({case["question"] for case in cases}) != 100:
        raise ValueError("duplicate_case_id_or_question")
    counts = Counter((case["split"], case["answerable"]) for case in cases)
    if counts != Counter({("dev", True): 40, ("dev", False): 10,
                          ("test", True): 40, ("test", False): 10}):
        raise ValueError("evaluation_split_mismatch")
    per_document = Counter()
    for case in cases:
        if not isinstance(case["answerable"], bool) or not case["category"] or not case["question"].strip():
            raise ValueError("invalid_case_fields")
        document, evidence = case["expected_document"], case["expected_evidence"]
        if case["answerable"]:
            if document not in texts or not evidence or evidence not in texts[document]:
                raise ValueError("evidence_not_in_document: " + case["id"])
            per_document[(document, case["split"])] += 1
        elif document is not None or evidence is not None:
            raise ValueError("unanswerable_case_has_gold_evidence")
    if any(per_document[(name, split)] != 2 for name in expected_names for split in ("dev", "test")):
        raise ValueError("expected_two_cases_per_document_per_split")
    result.update(cases=100, answerable=80, unanswerable=20, dev=50, test=50,
                  all_answerable_evidence_verified=True,
                  label_status="AI-assisted source-checked draft; independent human review pending")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch only pinned official raw URLs.")
    args = parser.parse_args()
    if args.download:
        acquire()
    print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
