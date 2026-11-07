"""Create a clean source archive; never include uploads, cookies or venv."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    roots = [ROOT / p for p in ("knowtrace", "web", "demo", "scripts", "tests", "docs")]
    files = [ROOT / p for p in ("README.md", "LICENSE", "pyproject.toml", "requirements-lock.txt", "requirements-semantic-lock.txt", ".gitignore", ".env.example", "API_CONTRACT.md")]
    for directory in roots:
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in p.parts) and p.suffix != ".pyc")
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.is_file():
                archive.write(path, Path("knowtrace") / path.relative_to(ROOT))
    print(f"Archive: {target}\nBytes: {target.stat().st_size}\nSHA256: {hashlib.sha256(target.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
