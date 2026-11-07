"""Explicit public download of pinned E5 files. Inference never calls this tool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_ID = "intfloat/multilingual-e5-small"
FILES = {
    "model_qint8_avx512_vnni.onnx": (118346824, "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"),
    "tokenizer.json": (17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
}


def valid_file(path, size, expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        return False
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return digest == expected


def allowed_download(url):
    value = urlsplit(str(url))
    host = value.hostname or ""
    return (value.scheme == "https" and value.port in {None, 443}
            and value.username is None and value.password is None
            and (host == "huggingface.co" or host.endswith(".huggingface.co")
                 or host.endswith(".hf.co")))


def download(client, url, path, size, digest):
    # Signed public CDN redirects are used only in memory, never in reports.
    for _ in range(6):
        if not allowed_download(url):
            raise ValueError("unapproved_download_host")
        with client.stream("GET", url) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                target = response.headers.get("location")
                if not target:
                    raise ValueError("missing_download_redirect")
                url = response.url.join(target)
                continue
            if response.status_code != 200:
                raise ValueError("download_http_failure")
            temporary = path.with_suffix(path.suffix + ".partial")
            if temporary.is_symlink():
                raise ValueError("invalid_download_target")
            count = 0
            hasher = hashlib.sha256()
            try:
                with temporary.open("wb") as handle:
                    for block in response.iter_bytes(1024 * 1024):
                        count += len(block)
                        if count > size:
                            raise ValueError("model_size_mismatch")
                        hasher.update(block)
                        handle.write(block)
                if count != size or hasher.hexdigest() != digest:
                    raise ValueError("model_integrity_mismatch")
                temporary.replace(path)
                return
            finally:
                temporary.unlink(missing_ok=True)
    raise ValueError("too_many_download_redirects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--activate", action="store_true", help="Save the non-secret model directory for next server startup")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    directory = args.destination.expanduser().resolve()
    results = {}
    if not args.check_only:
        directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=httpx.Timeout(120, connect=15)) as client:
        for filename, (size, digest) in FILES.items():
            path = directory / filename
            cached = valid_file(path, size, digest)
            if not cached and not args.check_only:
                if path.is_symlink():
                    raise ValueError("invalid_model_file")
                download(client, f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/onnx/{filename}", path, size, digest)
            results[filename] = {"bytes": size, "sha256": digest, "verified": valid_file(path, size, digest), "already_present": cached}
    complete = all(value["verified"] for value in results.values())
    if args.activate and complete and not args.check_only:
        settings = ROOT / "local-settings.json"
        # This file contains a local path only. It is excluded from source archives.
        settings.write_text(json.dumps({"embedding_model_dir": str(directory)}, indent=2), encoding="utf-8")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "model_id": MODEL_ID,
              "revision": REVISION, "verified": complete, "files": results,
              "download_requested": not args.check_only,
              "settings_saved": bool(args.activate and complete and not args.check_only)}
    print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Do not leak signed CDN URLs or networking/credential details.
        print(json.dumps({"status": "model_preparation_failed", "retry_automatic": False}))
        raise SystemExit(2)
