"""Loopback-only demo HTTP interface. Model inference is explicitly opt-in."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .core import KnowledgeCore
from .evaluation import dataset_summary, load_dataset, run_evaluation, save_run
from .integrity import read_manifest
from .config import model_config_from_env
from .generation import EvidenceGenerator, GenerationError, ModelConfig
from .embeddings import SemanticError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COOKIE = "knowtrace_session"
USERS = {
    "admin": {"id": "admin", "display_name": "管理员（演示）", "role": "admin", "kb_ids": None},
    "engineer": {"id": "engineer", "display_name": "工程成员（演示）", "role": "member", "kb_ids": ["kb-engineering"]},
    "support": {"id": "support", "display_name": "支持成员（演示）", "role": "member", "kb_ids": ["kb-support"]},
}


class Login(BaseModel):
    username: str = Field(max_length=80)
    password: str = Field(max_length=200)


class KBInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    kb_ids: list[str] | None = Field(default=None, max_length=50)
    mode: Literal["bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)
    answer_mode: Literal["extractive", "generated"] = "extractive"


class EvalInput(BaseModel):
    dataset_id: str
    mode: Literal["bm25", "tfidf", "hybrid", "semantic", "semantic_hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=10)
    split: Literal["dev", "test"] | None = None


def create_app(data_dir: Path | None = None, demo_dir: Path | None = None, *, model_config: ModelConfig | None = None, model_transport=None, embedder=None, retrieval_configured: bool | None = None) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("KNOWTRACE_DATA_DIR", PROJECT_ROOT / "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = Path(demo_dir or PROJECT_ROOT / "demo")
    core = KnowledgeCore(data_dir, embedder=embedder)
    secret_path = data_dir / ".session-key"
    if not secret_path.exists():
        try:
            with secret_path.open("xb") as handle:
                handle.write(secrets.token_bytes(32))
        except FileExistsError:
            pass
    session_key = secret_path.read_bytes()
    config = model_config or model_config_from_env()
    generator = EvidenceGenerator(config, transport=model_transport) if config.provider != "disabled" else None
    app = FastAPI(title="KnowTrace", version="0.3.0", docs_url="/api/docs", redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.state.core = core
    app.state.data_dir = data_dir
    app.state.last_model_success_at = None
    app.state.last_model_success_scope = None

    @app.exception_handler(SemanticError)
    async def semantic_failure(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"detail": "本地语义检索未完成，请使用词法模式或检查本机模型配置。"})

    def check_retrieval(mode):
        if mode in {"semantic", "semantic_hybrid"} and embedder is None:
            raise HTTPException(409, "本地语义模型尚未就绪，请选择词法检索模式。")

    @app.middleware("http")
    async def local_safety(request: Request, call_next):
        # A malicious site cannot mutate a local server through a browser form.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return Response("Cross-origin request blocked", status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    def current_user(request: Request) -> dict:
        try:
            payload, signature = request.cookies.get(COOKIE, "").split(".")
            expected = hmac.new(session_key, payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise ValueError()
            value = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            if value["expires"] < time.time():
                raise ValueError()
            return USERS[value["uid"]].copy()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise HTTPException(401, "请先选择演示账号登录") from None

    def allowed_scope(user: dict, requested: list[str] | None = None):
        allowed = user["kb_ids"]
        if requested is None:
            return allowed
        all_ids = {kb["id"] for kb in core.list_kbs(allowed)}
        if not set(requested).issubset(all_ids):
            raise HTTPException(403, "所选知识库不在当前账号权限内")
        return list(dict.fromkeys(requested))

    def get_dataset(dataset_id: str):
        path = demo_dir / "evaluation.json"
        if not path.exists():
            raise HTTPException(404, "尚未准备演示评测集")
        dataset = load_dataset(path)
        if dataset["id"] != dataset_id:
            raise HTTPException(404, "评测集不存在")
        return dataset

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.3.0", "mode": "offline", "model_connected": False,
                "model_configured": generator is not None, "default_answer_mode": "extractive"}

    @app.get("/api/model")
    def model_status(user=Depends(current_user)):
        verified = app.state.last_model_success_at is not None
        return {"provider": config.provider, "model": config.model, "configured": generator is not None,
                "live_verified": verified, "last_success_at": app.state.last_model_success_at,
                "last_success_scope": app.state.last_model_success_scope,
                "default_answer_mode": "extractive",
                "message": ("模型接口已配置，尚未实际请求；默认使用离线摘录。" if not verified else "曾完成模型请求；当前可用性以本次请求结果为准。") if generator else "尚未配置模型服务，当前使用离线证据摘录。"}

    @app.post("/api/auth/login")
    def login(body: Login, response: Response):
        # Public, intentionally low-privilege demo credentials. Not production auth.
        expected = body.username + "-demo"
        if body.username not in USERS or not hmac.compare_digest(body.password.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(401, "演示账号或密码不正确")
        payload = base64.urlsafe_b64encode(json.dumps({"uid": body.username, "expires": int(time.time()) + 8 * 3600, "nonce": secrets.token_hex(8)}).encode()).decode().rstrip("=")
        signature = hmac.new(session_key, payload.encode(), hashlib.sha256).hexdigest()
        response.set_cookie(COOKIE, f"{payload}.{signature}", httponly=True, samesite="strict", max_age=8 * 3600)
        return {"user": USERS[body.username]}

    @app.get("/api/retrieval")
    def retrieval_status(user=Depends(current_user)):
        status = core.retrieval_status()
        if retrieval_configured is not None:
            status["configured"] = retrieval_configured
        status["message"] = ("本地向量可用于候选排序；相似度不代表事实支持，回答仍通过原有证据准入。"
                             if status["available"] else "本地语义模型未就绪，词法检索仍可使用。")
        return status

    @app.post("/api/auth/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user=Depends(current_user)):
        return {"user": user}

    @app.get("/api/kbs")
    def kbs(user=Depends(current_user)):
        return {"items": core.list_kbs(user["kb_ids"])}

    @app.post("/api/kbs", status_code=201)
    def add_kb(body: KBInput, user=Depends(current_user)):
        if user["role"] != "admin":
            raise HTTPException(403, "只有管理员可创建知识库")
        if not body.name.strip():
            raise HTTPException(400, "名称不能为空")
        return core.create_kb(body.name.strip(), body.description.strip())

    @app.get("/api/documents")
    def documents(kb_id: str | None = None, user=Depends(current_user)):
        if kb_id:
            allowed_scope(user, [kb_id])
        return {"items": core.list_documents(user["kb_ids"], kb_id)}

    @app.post("/api/documents", status_code=201)
    def upload(file: UploadFile = File(...), kb_id: str = Form(...), source_url: str | None = Form(None), user=Depends(current_user)):
        allowed_scope(user, [kb_id])
        name = Path((file.filename or "").replace("\\", "/")).name
        if not name or len(name) > 180 or Path(name).suffix.lower() not in {".pdf", ".md", ".txt", ".docx"}:
            raise HTTPException(400, "支持 PDF、Markdown、TXT、DOCX，文件名不超过180字符")
        if source_url and (len(source_url) > 2000 or not source_url.startswith(("https://", "http://"))):
            raise HTTPException(400, "来源地址应为 http(s) URL，仅作为引用保存，不会抓取")
        temp_dir = data_dir / "tmp"
        temp_dir.mkdir(exist_ok=True)
        path = temp_dir / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
        try:
            size = 0
            with path.open("wb") as handle:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise HTTPException(413, "单个文件最多20MB")
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, "文件为空")
            result = core.ingest_file(path, name, kb_id, source_url=source_url)
            return {"document": result}
        except HTTPException:
            raise
        except Exception as exc:
            # Parser messages can contain source paths; keep public errors generic.
            raise HTTPException(400, "无法解析此文档。请检查格式、是否含可提取文本或是否已加密。") from exc
        finally:
            file.file.close()
            path.unlink(missing_ok=True)

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str, user=Depends(current_user)):
        if not core.delete_document(document_id, user["kb_ids"]):
            raise HTTPException(404, "文档不存在或无权访问")
        return {"deleted": True}

    @app.get("/api/documents/{document_id}/source")
    def source(document_id: str, user=Depends(current_user)):
        document = core.get_document(document_id, user["kb_ids"])
        path = core.source_path(document_id, user["kb_ids"])
        if document is None or path is None or not path.is_file():
            raise HTTPException(404, "文档不存在或无权访问")
        # Download only, so an HTML payload renamed .txt never runs in app origin.
        return FileResponse(path, filename=document["name"], media_type="application/octet-stream")

    @app.get("/api/documents/{document_id}/chunks")
    def chunks(document_id: str, page: int | None = None, user=Depends(current_user)):
        document = core.get_document(document_id, user["kb_ids"])
        if document is None:
            raise HTTPException(404, "文档不存在或无权访问")
        return {"document": document, "items": core.get_document_chunks(document_id, user["kb_ids"], page)}

    @app.post("/api/ask")
    def ask(body: Query, user=Depends(current_user)):
        start = time.perf_counter()
        if not body.question.strip():
            raise HTTPException(400, "问题不能为空")
        scope = allowed_scope(user, body.kb_ids)
        check_retrieval(body.mode)
        if body.answer_mode == "generated" and generator is None:
            raise HTTPException(409, "尚未配置模型服务，请使用离线摘录；模型配置在服务启动时读取。")
        result = core.ask(body.question.strip(), scope, body.mode, body.top_k)
        if body.answer_mode == "extractive":
            return result
        if result["refused"]:
            return {**result, "model_invoked": False, "generation_status": "skipped_insufficient_evidence"}
        try:
            generated = generator.generate(body.question.strip(), result["sources"])
        except GenerationError as error:
            headers = {"X-KnowTrace-Model-Error": getattr(error, "code", "invalid_model_output")}
            provider_status = getattr(error, "provider_http_status", None)
            if provider_status is not None:
                headers["X-KnowTrace-Provider-Status"] = str(provider_status)
            raise HTTPException(502, "模型请求或引用核验未通过，未采用未经核验的输出。可切换到离线摘录查看原文。", headers=headers) from None
        app.state.last_model_success_at = datetime.now(timezone.utc).isoformat()
        app.state.last_model_success_scope = "current_app_request"
        return {**generated, "model_invoked": True,
                "evidence_decision": result.get("evidence_decision"),
                "retrieval_candidates": result.get("retrieval_candidates", []),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    @app.post("/api/search")
    def search(body: Query, user=Depends(current_user)):
        start = time.perf_counter()
        if not body.question.strip():
            raise HTTPException(400, "问题不能为空")
        check_retrieval(body.mode)
        result = core.search(body.question.strip(), allowed_scope(user, body.kb_ids), body.mode, body.top_k)
        return {"items": result, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}

    @app.get("/api/evaluations/datasets")
    def datasets(user=Depends(current_user)):
        path = demo_dir / "evaluation.json"
        return {"items": [dataset_summary(load_dataset(path))] if path.exists() else []}

    @app.post("/api/evaluations/run")
    def evaluate(body: EvalInput, user=Depends(current_user)):
        check_retrieval(body.mode)
        try:
            result = run_evaluation(core, get_dataset(body.dataset_id), user["kb_ids"], body.mode, body.top_k, body.split, manifest=read_manifest(demo_dir))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        save_run(result, data_dir / "evaluations", user["id"])
        return result

    def visible_runs(user):
        directory = data_dir / "evaluations"
        runs = []
        if directory.exists():
            for path in directory.glob("*.json"):
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if user["role"] == "admin" or value.get("owner_user_id") == user["id"]:
                        runs.append(value)
                except (OSError, ValueError):
                    continue
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    @app.get("/api/evaluations/runs")
    def evaluations(user=Depends(current_user)):
        return {"items": [{k: v for k, v in r.items() if k not in {"cases", "owner_user_id"}} for r in visible_runs(user)]}

    def find_run(run_id, user):
        for item in visible_runs(user):
            if item["id"] == run_id:
                item.pop("owner_user_id", None)
                return item
        raise HTTPException(404, "报告不存在或无权访问")

    @app.get("/api/evaluations/runs/{run_id}")
    def report(run_id: str, user=Depends(current_user)):
        return find_run(run_id, user)

    @app.get("/api/evaluations/runs/{run_id}/download")
    def download_report(run_id: str, user=Depends(current_user)):
        value = find_run(run_id, user)
        return Response(json.dumps(value, ensure_ascii=False, indent=2), media_type="application/json", headers={"Content-Disposition": 'attachment; filename="knowtrace-evaluation.json"'})

    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
    return app
