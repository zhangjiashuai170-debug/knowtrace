"""Permission-scoped ingestion, evidence retrieval and optional local vectors."""

from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time
from urllib.parse import urlparse
import uuid

from .parsing import chunks, parse
from .retrieval import (LEXICAL_MODES, SEMANTIC_MODES, RETRIEVAL_MODES, RankIndex,
                        embedding_identity, normalized_vector, semantic_failure)

MAX_FILE_BYTES = 32 * 1024 * 1024
EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
DOCUMENT_FIELDS = (
    "id", "kb_id", "name", "chunk_count", "page_count", "parser",
    "source_url", "license_name", "created_at",
)
CHUNK_SELECT = """SELECT c.id AS chunk_id,c.document_id,c.kb_id,d.name AS document_name,
    c.text,c.page,c.section,c.line_start,c.line_end,d.source_url
    FROM chunks c JOIN documents d ON d.id=c.document_id"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _acl(allowed):
    if allowed is None:
        return None
    if isinstance(allowed, (str, bytes)):
        raise ValueError("权限范围应为知识库编号列表")
    try:
        values = list(allowed)
    except TypeError as error:
        raise ValueError("权限范围应为知识库编号列表") from error
    if any(not isinstance(value, str) for value in values):
        raise ValueError("知识库编号无效")
    return tuple(sorted(set(values)))


def _scope(column: str, allowed: tuple | None) -> tuple[str, list]:
    if allowed is None:
        return "1=1", []
    if not allowed:
        return "1=0", []
    return column + " IN (" + ",".join("?" for _ in allowed) + ")", list(allowed)


class KnowledgeCore:
    """The data directory is the only writable document store used by this core."""

    def __init__(self, data_dir: Path, *, embedder=None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.data_dir / "documents"
        self.staging_dir = self.data_dir / ".staging"
        for directory in (self.files_dir, self.staging_dir):
            if directory.is_symlink():
                raise ValueError("受管理目录不能是符号链接")
            directory.mkdir(exist_ok=True)
        self.database = self.data_dir / "knowledge.sqlite3"
        if self.database.is_symlink():
            raise ValueError("资料库不能是符号链接")
        self._lock = threading.RLock()
        self._cache: OrderedDict = OrderedDict()
        self._revision = -1
        self.embedder = embedder
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value INTEGER NOT NULL);
                INSERT OR IGNORE INTO meta VALUES('revision',0);
                CREATE TABLE IF NOT EXISTS kbs(
                    id TEXT PRIMARY KEY,name TEXT NOT NULL,description TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents(
                    id TEXT PRIMARY KEY,kb_id TEXT NOT NULL REFERENCES kbs(id),
                    name TEXT NOT NULL,sha256 TEXT NOT NULL,storage_name TEXT NOT NULL UNIQUE,
                    parser TEXT NOT NULL,page_count INTEGER,chunk_count INTEGER NOT NULL,
                    source_url TEXT,license_name TEXT,created_at TEXT NOT NULL,
                    UNIQUE(kb_id,sha256));
                CREATE TABLE IF NOT EXISTS chunks(
                    id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    kb_id TEXT NOT NULL REFERENCES kbs(id),ordinal INTEGER NOT NULL,text TEXT NOT NULL,
                    page INTEGER,section TEXT NOT NULL,line_start INTEGER NOT NULL,line_end INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_kb ON chunks(kb_id);
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id,ordinal);
                CREATE TABLE IF NOT EXISTS chunk_embeddings(
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_fingerprint TEXT NOT NULL,text_sha256 TEXT NOT NULL,
                    dimension INTEGER NOT NULL,vector_json TEXT NOT NULL,
                    PRIMARY KEY(chunk_id,model_fingerprint));
                CREATE TABLE IF NOT EXISTS pending_deletes(storage_name TEXT PRIMARY KEY);
            """)
        self._cleanup_pending()

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _safe_file(self, storage_name: str) -> Path | None:
        if not re.fullmatch(r"[a-f0-9]{32}\.(?:pdf|md|txt|docx)", storage_name):
            return None
        candidate = self.files_dir / storage_name
        if self.files_dir.is_symlink() or candidate.is_symlink():
            return None
        if candidate.resolve().parent != self.files_dir.resolve():
            return None
        return candidate

    def _cleanup_pending(self):
        with self._lock, closing(self._connect()) as db:
            names = [row[0] for row in db.execute("SELECT storage_name FROM pending_deletes")]
            for name in names:
                target = self._safe_file(name)
                if target is None:
                    continue
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    continue
                with db:
                    db.execute("DELETE FROM pending_deletes WHERE storage_name=?", (name,))

    def _changed(self, db):
        db.execute("UPDATE meta SET value=value+1 WHERE key='revision'")
        self._cache.clear()
        self._revision = -1

    @staticmethod
    def _document(row, duplicate=None):
        result = {field: row[field] for field in DOCUMENT_FIELDS}
        if duplicate is not None:
            result["duplicate"] = duplicate
        return result

    def list_kbs(self, allowed_kb_ids=None):
        condition, params = _scope("k.id", _acl(allowed_kb_ids))
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(f"""SELECT k.id,k.name,k.description,
                (SELECT COUNT(*) FROM documents d WHERE d.kb_id=k.id) AS document_count,
                (SELECT COUNT(*) FROM chunks c WHERE c.kb_id=k.id) AS chunk_count
                FROM kbs k WHERE {condition} ORDER BY k.created_at,k.id""", params)
            return [dict(row) for row in rows]

    def create_kb(self, name, description="", kb_id=None):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 128:
            raise ValueError("知识库名称需为 1 至 128 个字符")
        if not isinstance(description, str) or len(description) > 2000:
            raise ValueError("知识库说明过长")
        identifier = kb_id if kb_id is not None else "kb-" + uuid.uuid4().hex
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier):
            raise ValueError("知识库编号无效")
        with self._lock, closing(self._connect()) as db, db:
            existing = db.execute("SELECT id FROM kbs WHERE id=?", (identifier,)).fetchone()
            if existing is None:
                db.execute("INSERT INTO kbs VALUES(?,?,?,?)",
                           (identifier, name.strip(), description.strip(), _now()))
                self._changed(db)
        return next(item for item in self.list_kbs() if item["id"] == identifier)

    def ingest_file(self, path, filename, kb_id, source_url=None, license_name=None):
        if not isinstance(filename, str):
            raise ValueError("文件名无效")
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or name in (".", "..") or len(name) > 255 or any(ord(c) < 32 for c in name):
            raise ValueError("文件名无效")
        extension = Path(name).suffix.lower()
        if extension not in EXTENSIONS:
            raise ValueError("仅支持 PDF、Markdown、TXT 和 DOCX 文件")
        if source_url is not None:
            if not isinstance(source_url, str) or len(source_url) > 2000:
                raise ValueError("出处地址无效")
            parsed_url = urlparse(source_url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc or parsed_url.username:
                raise ValueError("出处地址需要为 HTTP 或 HTTPS 地址")
        if license_name is not None and (not isinstance(license_name, str) or len(license_name) > 200):
            raise ValueError("授权说明无效")
        source = Path(path)
        if not source.is_file():
            raise ValueError("上传文件不存在")
        with self._lock, closing(self._connect()) as db:
            if not db.execute("SELECT id FROM kbs WHERE id=?", (kb_id,)).fetchone():
                raise ValueError("知识库不存在")
        temporary = None
        installed = None
        try:
            digest = hashlib.sha256()
            size = 0
            with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
                    dir=self.staging_dir, suffix=extension, delete=False) as staged:
                temporary = Path(staged.name)
                while block := input_stream.read(65536):
                    size += len(block)
                    if size > MAX_FILE_BYTES:
                        raise ValueError("单个文件超过 32 MB 上限")
                    digest.update(block)
                    staged.write(block)
                staged.flush()
                os.fsync(staged.fileno())
            content_hash = digest.hexdigest()
            with self._lock, closing(self._connect()) as db:
                duplicate = db.execute("SELECT * FROM documents WHERE kb_id=? AND sha256=?",
                                       (kb_id, content_hash)).fetchone()
                if duplicate:
                    return self._document(duplicate, True)
            parsed = parse(temporary, extension)
            pieces = chunks(parsed)
            if not pieces or len(pieces) > 20000:
                raise ValueError("可检索分块为空或超过处理上限")
            with self._lock, closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                try:
                    duplicate = db.execute("SELECT * FROM documents WHERE kb_id=? AND sha256=?",
                                           (kb_id, content_hash)).fetchone()
                    if duplicate:
                        db.rollback()
                        return self._document(duplicate, True)
                    identifier = uuid.uuid4().hex
                    storage_name = identifier + extension
                    target = self._safe_file(storage_name)
                    if target is None:
                        raise ValueError("受管理文件位置无效")
                    # Reserve only a fresh generated path before replacing our own placeholder.
                    with target.open("xb"):
                        pass
                    installed = target
                    os.replace(temporary, target)
                    temporary = None
                    db.execute("""INSERT INTO documents(
                        id,kb_id,name,sha256,storage_name,parser,page_count,chunk_count,
                        source_url,license_name,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (identifier, kb_id, name, content_hash, storage_name, parsed.parser,
                         parsed.page_count, len(pieces), source_url, license_name, _now()))
                    db.executemany("""INSERT INTO chunks(
                        id,document_id,kb_id,ordinal,text,page,section,line_start,line_end)
                        VALUES(?,?,?,?,?,?,?,?,?)""", [
                            (uuid.uuid5(uuid.NAMESPACE_URL, identifier + ":" + str(i)).hex,
                             identifier, kb_id, i, piece["text"], piece["page"], piece["section"],
                             piece["line_start"], piece["line_end"]) for i, piece in enumerate(pieces)])
                    self._changed(db)
                    result = db.execute("SELECT * FROM documents WHERE id=?", (identifier,)).fetchone()
                    db.commit()
                    installed = None
                    return self._document(result, False)
                except Exception:
                    db.rollback()
                    raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if installed is not None:
                installed.unlink(missing_ok=True)

    def list_documents(self, allowed_kb_ids, kb_id=None):
        condition, params = _scope("kb_id", _acl(allowed_kb_ids))
        if kb_id is not None:
            condition += " AND kb_id=?"
            params.append(kb_id)
        with self._lock, closing(self._connect()) as db:
            return [self._document(row) for row in db.execute(
                "SELECT * FROM documents WHERE " + condition + " ORDER BY created_at DESC,id", params)]

    def _authorized_document(self, db, document_id, allowed):
        condition, params = _scope("kb_id", allowed)
        return db.execute("SELECT * FROM documents WHERE id=? AND " + condition,
                          [document_id, *params]).fetchone()

    def get_document(self, document_id, allowed_kb_ids):
        with self._lock, closing(self._connect()) as db:
            row = self._authorized_document(db, document_id, _acl(allowed_kb_ids))
            return self._document(row) if row is not None else None

    def get_document_chunks(self, document_id, allowed_kb_ids, page=None):
        if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
            raise ValueError("页码需为正整数")
        with self._lock, closing(self._connect()) as db:
            if self._authorized_document(db, document_id, _acl(allowed_kb_ids)) is None:
                return []
            query = CHUNK_SELECT + " WHERE c.document_id=?"
            params = [document_id]
            if page is not None:
                query += " AND c.page=?"
                params.append(page)
            return [dict(row) for row in db.execute(query + " ORDER BY c.ordinal", params)]

    def source_path(self, document_id, allowed_kb_ids):
        with self._lock, closing(self._connect()) as db:
            row = self._authorized_document(db, document_id, _acl(allowed_kb_ids))
            if row is None:
                return None
            path = self._safe_file(row["storage_name"])
            return path if path is not None and path.is_file() else None

    def delete_document(self, document_id, allowed_kb_ids):
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = self._authorized_document(db, document_id, _acl(allowed_kb_ids))
                if row is None or self._safe_file(row["storage_name"]) is None:
                    db.rollback()
                    return False
                db.execute("INSERT OR IGNORE INTO pending_deletes VALUES(?)", (row["storage_name"],))
                db.execute("DELETE FROM documents WHERE id=?", (document_id,))
                self._changed(db)
                db.commit()
            except Exception:
                db.rollback()
                raise
            self._cleanup_pending()
            return True

    def retrieval_status(self):
        """Public metadata only; no encoder invocation, paths or query history."""
        result = {"available": False, "configured": self.embedder is not None,
                  "default_mode": "hybrid", "modes": list(LEXICAL_MODES),
                  "model": None, "fingerprint": None, "dimension": None}
        if self.embedder is None:
            return result
        try:
            fingerprint, dimension = embedding_identity(self.embedder)
            metadata = self.embedder.metadata
            if not isinstance(metadata, dict):
                raise ValueError()
            permitted = {"model", "model_id", "revision", "provider", "dimension", "fingerprint",
                         "pooling", "normalize", "query_prefix", "document_prefix", "max_length",
                         "normalization", "max_tokens", "runtime_version", "tokenizers_version", "numpy_version",
                         "batch_size", "batch_policy",
                         "quantization", "license", "runtime", "local_only"}
            public = {}
            for key, value in metadata.items():
                if key not in permitted or not isinstance(value, (str, bool, int, float, type(None))):
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if isinstance(value, str) and (len(value) > 256 or any(ord(c) < 32 for c in value)
                        or value.startswith(("/", "\\", "file:")) or re.match(r"^[A-Za-z]:[\\/]", value)):
                    continue
                public[key] = value
            result.update(available=True, modes=list(RETRIEVAL_MODES), model=public,
                          fingerprint=fingerprint, dimension=dimension)
        except Exception:
            result["error_class"] = "InvalidEmbeddingConfiguration"
        return result

    def _document_vectors(self, rows, identity):
        """Read/store vectors for supplied authorized rows, never global rows."""
        fingerprint, dimension = identity
        if not rows:
            return []
        with self._lock, closing(self._connect()) as db:
            try:
                existing = {}
                identifiers = [row["chunk_id"] for row in rows]
                for start in range(0, len(identifiers), 256):
                    selected = identifiers[start:start + 256]
                    placeholders = ",".join("?" for _ in selected)
                    query = ("SELECT chunk_id,text_sha256,dimension,vector_json FROM chunk_embeddings "
                             "WHERE model_fingerprint=? AND chunk_id IN (" + placeholders + ")")
                    existing.update({row["chunk_id"]: dict(row) for row in db.execute(
                        query, [fingerprint, *selected])})
                result = [None] * len(rows)
                missing = []
                for position, row in enumerate(rows):
                    digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                    stored = existing.get(row["chunk_id"])
                    if stored is not None:
                        stored_dimension = stored["dimension"]
                        if (isinstance(stored_dimension, bool) or not isinstance(stored_dimension, int)
                                or not 1 <= stored_dimension <= 4096):
                            raise semantic_failure("本地语义向量缓存的维度无效。")
                        if stored["text_sha256"] == digest and stored_dimension == dimension:
                            try:
                                if (not isinstance(stored["vector_json"], str)
                                        or len(stored["vector_json"]) > dimension * 40 + 1000):
                                    raise ValueError()
                                vector = json.loads(stored["vector_json"])
                            except (ValueError, TypeError):
                                raise semantic_failure("本地语义向量缓存已损坏。") from None
                            result[position] = normalized_vector(vector, dimension, cached=True)
                            continue
                    missing.append((position, row, digest))
                inserts = []
                for start in range(0, len(missing), 32):
                    selected = missing[start:start + 32]
                    vectors = self.embedder.embed_documents([item[1]["text"] for item in selected])
                    if not isinstance(vectors, list) or len(vectors) != len(selected):
                        raise semantic_failure("本地语义模型返回的文档向量数量无效。")
                    for (position, row, digest), vector in zip(selected, vectors):
                        normalized = normalized_vector(vector, dimension)
                        result[position] = normalized
                        inserts.append((row["chunk_id"], fingerprint, digest, dimension,
                                        json.dumps(normalized, separators=(",", ":"), allow_nan=False)))
                if inserts:
                    with db:
                        db.executemany("INSERT OR REPLACE INTO chunk_embeddings VALUES(?,?,?,?,?)", inserts)
                return result
            except Exception as error:
                from .embeddings import SemanticError
                if isinstance(error, SemanticError):
                    raise
                raise SemanticError("本地语义索引未完成，请检查模型或重新检索。") from None

    def _index(self, allowed, *, semantic=False):
        identity = embedding_identity(self.embedder) if semantic else None
        cache_key = (allowed, identity)
        with self._lock, closing(self._connect()) as db:
            db.execute("BEGIN")
            revision = db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            if revision != self._revision:
                self._cache.clear()
                self._revision = revision
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]
            condition, params = _scope("c.kb_id", allowed)
            rows = [dict(row) for row in db.execute(
                CHUNK_SELECT + " WHERE " + condition + " ORDER BY c.document_id,c.ordinal", params)]
            index = RankIndex(rows, embedder=self.embedder if semantic else None,
                              vector_loader=(lambda values: self._document_vectors(values, identity)) if semantic else None)
            self._cache[cache_key] = index
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
            return index

    def search(self, question, allowed_kb_ids, mode="hybrid", top_k=5):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("问题需为 1 至 2000 个字符")
        if mode not in RETRIEVAL_MODES:
            raise ValueError("检索模式无效")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k 需为 1 至 20")
        return self._index(_acl(allowed_kb_ids), semantic=mode in SEMANTIC_MODES).search(question.strip(), mode, top_k)

    def ask(self, question, allowed_kb_ids, mode="hybrid", top_k=5):
        from .retrieval import MINIMUM_COVERAGE

        started = time.perf_counter()
        hits = self.search(question, allowed_kb_ids, mode, top_k)
        # Distinct content-term IDF coverage, including unknown query terms.
        supported = [hit for hit in hits if hit["coverage"] >= MINIMUM_COVERAGE
                     and not hit["evidence_rejections"]
                     and (hit["bm25_score"] > 0 or hit["tfidf_score"] >= .12)]
        candidates = [dict(hit, is_supporting_evidence=False) for hit in hits]
        decision = {
            "status": "accepted" if supported else "refused",
            "message": ("找到词法条件合格的原文片段；以下仅摘录证据，未核验完整语义。" if supported
                        else "候选片段不足以支持回答；可展开检索候选自行核对。" if hits
                        else "当前权限内未找到相关候选。"),
            "policy_version": "lexical-v2", "minimum_coverage": MINIMUM_COVERAGE,
            "coverage_definition": "权限内语料的 IDF 加权内容词覆盖率，分母保留未知内容词。",
            "query_terms": hits[0]["query_terms"] if hits else [],
            "candidate_count": len(hits),
            "supporting_chunk_ids": [hit["chunk_id"] for hit in supported[:3]],
            "candidate_decisions": [
                {"chunk_id": hit["chunk_id"], "coverage": hit["coverage"],
                 "matched_terms": hit["matched_terms"], "missing_terms": hit["missing_terms"],
                 "reasons": hit["evidence_rejections"] + (["insufficient_coverage"] if hit["coverage"] < MINIMUM_COVERAGE else [])}
                for hit in hits],
        }
        if not supported:
            return {"answer": "当前可访问资料中没有足够依据回答这个问题。可以换一种问法，或补充相关文档。",
                    "answer_mode": "extractive", "refused": True, "sources": [],
                    "retrieval_candidates": candidates, "evidence_decision": decision,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
        sources = [dict(hit, citation_id="S" + str(i)) for i, hit in enumerate(supported[:3], 1)]
        quotations = [f'[{hit["citation_id"]}] 《{hit["document_name"]}》\n{hit["text"]}'
                      for hit in sources]
        return {"answer": "以下内容为离线检索得到的原文证据摘录。\n\n" + "\n\n".join(quotations),
                "answer_mode": "extractive", "refused": False, "sources": sources,
                "retrieval_candidates": candidates, "evidence_decision": decision,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
