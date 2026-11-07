"""Explicit local embedding setup, separate from API credentials and unit tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def configured_embedder(*, project_root=ROOT, environ=None):
    """Return (encoder or None, configured). Never download or discover secrets."""
    source = os.environ if environ is None else environ
    directory = source.get("KNOWTRACE_EMBEDDING_DIR", "")
    configured = bool(directory)
    try:
        if not directory:
            settings = Path(project_root) / "local-settings.json"
            if not settings.exists():
                return None, False
            configured = True
            if settings.is_symlink() or settings.stat().st_size > 4096:
                raise ValueError()
            value = json.loads(settings.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or set(value) != {"embedding_model_dir"}:
                raise ValueError()
            directory = value["embedding_model_dir"]
        if (not isinstance(directory, str) or len(directory) > 1024
                or not directory or "://" in directory or not Path(directory).is_absolute()):
            raise ValueError()
        from .embeddings import LocalEmbedder
        return LocalEmbedder(Path(directory)), True
    except Exception:
        # An optional embedding failure must not prevent lexical retrieval.
        # The public API returns a fixed message, never a local path/exception.
        return None, configured
