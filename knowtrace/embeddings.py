"""Verified, local-only multilingual E5 ONNX inference.

Importing this module does not import an inference runtime, load weights,
discover credentials, or contact a model registry. The explicit constructor
accepts only already-downloaded files matching this pinned public checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILENAME = "model_qint8_avx512_vnni.onnx"
MODEL_FILES = {
    MODEL_FILENAME: {
        "size": 118_346_824,
        "sha256": "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
    },
    "tokenizer.json": {
        "size": 17_082_730,
        "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    },
}
DIMENSION = 384
MAX_TOKENS = 512
# Dynamic INT8 activation scales depend on the other samples in a batch.
# One passage per inference makes cached vectors independent of ACL grouping.
BATCH_SIZE = 1
MAX_DOCUMENTS = 2_000
MAX_DOCUMENT_CHARS = 20_000
MAX_QUERY_CHARS = 2_000
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "


class SemanticError(Exception):
    """Public-safe semantic failure, never carrying a path or input text."""


def _verify_files(directory: Path) -> None:
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError()
        for name, expected in MODEL_FILES.items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
                raise ValueError()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while piece := handle.read(1024 * 1024):
                    digest.update(piece)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError()
    except Exception:
        raise SemanticError("本地语义模型缺失或校验未通过，请运行模型准备程序。") from None


def _fingerprint() -> tuple[str, dict]:
    specification = {
        "implementation": "e5-onnx-maskedmean-v2",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": deepcopy(MODEL_FILES),
        "dimension": DIMENSION,
        "max_tokens": MAX_TOKENS,
        "query_prefix": QUERY_PREFIX,
        "document_prefix": DOCUMENT_PREFIX,
        "padding": "right; tokenizer <pad> ID; attention mask",
        "pooling": "attention_mask_mean",
        "normalization": "l2_float32",
        "batch_size": BATCH_SIZE,
        "batch_policy": "one_text_per_inference_for_corpus_independence",
    }
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True, ensure_ascii=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest, specification


class LocalEmbedder:
    """One explicitly configured CPU runtime; it never downloads model files."""

    dimension = DIMENSION

    def __init__(self, model_dir: Path):
        try:
            directory = Path(model_dir)
        except Exception:
            raise SemanticError("本地语义模型目录无效。") from None
        _verify_files(directory)
        # Optional dependencies are behind explicit local model construction.
        try:
            self._np = importlib.import_module("numpy")
            runtime = importlib.import_module("onnxruntime")
            tokenizer_runtime = importlib.import_module("tokenizers")
        except Exception:
            raise SemanticError("本地语义推理依赖不可用，请先安装语义检索依赖。") from None
        self._lock = threading.RLock()
        try:
            runtime.disable_telemetry_events()
            self._tokenizer = tokenizer_runtime.Tokenizer.from_file(str(directory / "tokenizer.json"))
            pad_id = self._tokenizer.token_to_id("<pad>")
            if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
                raise ValueError()
            self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
            self._tokenizer.enable_padding(direction="right", pad_id=pad_id,
                                           pad_type_id=0, pad_token="<pad>")
            options = runtime.SessionOptions()
            options.intra_op_num_threads = min(4, max(1, os.cpu_count() or 1))
            options.inter_op_num_threads = 1
            options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = runtime.InferenceSession(str(directory / MODEL_FILENAME),
                                                    sess_options=options,
                                                    providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            self._input_types = {item.name: item.type for item in inputs}
            if (len(inputs) != len(self._input_types)
                    or not {"input_ids", "attention_mask"}.issubset(self._input_types)
                    or set(self._input_types).difference({"input_ids", "attention_mask", "token_type_ids"})
                    or any(kind not in {"tensor(int64)", "tensor(int32)"}
                           for kind in self._input_types.values())):
                raise ValueError()
            outputs = self._session.get_outputs()
            names = [item.name for item in outputs]
            if "last_hidden_state" in names:
                self._output_name = "last_hidden_state"
            elif len(names) == 1 and isinstance(names[0], str) and names[0]:
                self._output_name = names[0]
            else:
                raise ValueError()
        except Exception:
            raise SemanticError("本地语义模型无法初始化，请检查依赖及模型兼容性。") from None
        self.fingerprint, specification = _fingerprint()
        self._metadata = {
            **specification,
            "fingerprint": self.fingerprint,
            "local_only": True,
            "runtime": "onnxruntime",
            "runtime_version": str(getattr(runtime, "__version__", "unknown")),
            "tokenizers_version": str(getattr(tokenizer_runtime, "__version__", "unknown")),
            "numpy_version": str(getattr(self._np, "__version__", "unknown")),
            "provider": "CPUExecutionProvider",
            "batch_size": BATCH_SIZE,
            "intra_op_threads": options.intra_op_num_threads,
        }

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    @staticmethod
    def _validated_texts(texts, *, maximum_chars: int) -> list[str]:
        if not isinstance(texts, list) or len(texts) > MAX_DOCUMENTS:
            raise SemanticError("语义编码输入数量无效或超过本机处理上限。")
        checked = []
        for text in texts:
            if (not isinstance(text, str) or not text.strip()
                    or len(text) > maximum_chars or "\x00" in text):
                raise SemanticError("语义编码文本为空、格式无效或超过长度限制。")
            try:
                text.encode("utf-8")
            except UnicodeError:
                raise SemanticError("语义编码文本包含无效字符。") from None
            checked.append(text.strip())
        return checked

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = []
        # Serialize tokenizer configuration-sensitive operations and bound CPU
        # concurrency. No query strings or embeddings are logged or persisted.
        with self._lock:
            for offset in range(0, len(texts), BATCH_SIZE):
                batch = texts[offset:offset + BATCH_SIZE]
                try:
                    encoded = self._tokenizer.encode_batch([prefix + text for text in batch])
                    if len(encoded) != len(batch):
                        raise ValueError()
                    ids = [item.ids for item in encoded]
                    masks = [item.attention_mask for item in encoded]
                    if not ids or not ids[0] or len(ids[0]) > MAX_TOKENS:
                        raise ValueError()
                    width = len(ids[0])
                    if any(len(row) != width for row in ids + masks):
                        raise ValueError()
                    if any(type(value) is not int or value < 0 for row in ids for value in row):
                        raise ValueError()
                    if any(type(value) is not int or value not in (0, 1) for row in masks for value in row):
                        raise ValueError()
                    if any(not any(row) for row in masks):
                        raise ValueError()
                    np = self._np
                    feeds = {}
                    values = {"input_ids": ids, "attention_mask": masks,
                              "token_type_ids": [[0] * width for _ in batch]}
                    for name, kind in self._input_types.items():
                        feeds[name] = np.asarray(values[name], dtype=np.int64 if kind == "tensor(int64)" else np.int32)
                    results = self._session.run([self._output_name], feeds)
                    if not isinstance(results, (list, tuple)) or len(results) != 1:
                        raise ValueError()
                    hidden = np.asarray(results[0])
                    if (hidden.ndim != 3 or hidden.shape != (len(batch), width, self.dimension)
                            or hidden.dtype.kind != "f" or not np.isfinite(hidden).all()):
                        raise ValueError()
                    mask = np.asarray(masks, dtype=np.float32)
                    pooled = ((hidden.astype(np.float32, copy=False) * mask[:, :, None]).sum(axis=1)
                              / mask.sum(axis=1, keepdims=True))
                    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                    if not np.isfinite(pooled).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                        raise ValueError()
                    normalized = pooled / norms
                    if not np.isfinite(normalized).all():
                        raise ValueError()
                    vectors.extend(normalized.tolist())
                except Exception:
                    raise SemanticError("本地语义编码未完成，请检查模型状态；未回退为其他检索结果。") from None
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        checked = self._validated_texts(texts, maximum_chars=MAX_DOCUMENT_CHARS)
        return self._encode(checked, DOCUMENT_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        checked = self._validated_texts([text], maximum_chars=MAX_QUERY_CHARS)
        return self._encode(checked, QUERY_PREFIX)[0]
