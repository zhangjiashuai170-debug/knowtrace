"""Pinned-file and numerical checks with a fake tokenizer/ORT, never weights."""
from copy import deepcopy
import builtins
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowtrace import embeddings as module


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    files = {module.MODEL_FILENAME: b"tiny fake model fixture", "tokenizer.json": b"tiny fake tokenizer fixture"}
    specs = {}
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
        specs[name] = {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    monkeypatch.setattr(module, "MODEL_FILES", specs)
    return tmp_path


@pytest.fixture
def runtime(model_files, monkeypatch):
    # NumPy is the local numerical dependency; tokenizer and inference remain
    # stubs. No model, tokenizer JSON parsing, registry or inference is loaded.
    np = pytest.importorskip("numpy")
    state = SimpleNamespace(encoded=[], feeds=[], telemetry_disabled=False,
                            input_names=["input_ids", "attention_mask", "token_type_ids"],
                            input_type="tensor(int64)", output_names=["last_hidden_state"],
                            output=None, encoded_override=None, init_error=None)

    class Tokenizer:
        @staticmethod
        def from_file(path):
            assert path == str(model_files / "tokenizer.json")
            if state.init_error:
                raise RuntimeError(state.init_error)
            result = Tokenizer()
            state.tokenizer = result
            return result

        def token_to_id(self, value):
            assert value == "<pad>"
            return getattr(state, "pad_id", 7)

        def enable_truncation(self, **kwargs):
            state.truncation = kwargs

        def enable_padding(self, **kwargs):
            state.padding = kwargs

        def encode_batch(self, texts):
            state.encoded.append(texts)
            if state.encoded_override is not None:
                return state.encoded_override(texts)
            return [SimpleNamespace(ids=[11, 12, 7], attention_mask=[1, 1, 0]) for _ in texts]

    class Session:
        def __init__(self, path, *, sess_options, providers):
            assert state.telemetry_disabled
            assert path == str(model_files / module.MODEL_FILENAME)
            assert providers == ["CPUExecutionProvider"]
            state.options = sess_options

        def get_inputs(self):
            return [SimpleNamespace(name=name, type=state.input_type) for name in state.input_names]

        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in state.output_names]

        def run(self, names, feeds):
            state.feeds.append(feeds)
            assert names == [state.output_names[0]]
            if state.output is not None:
                return state.output(feeds)
            batch, width = feeds["input_ids"].shape
            hidden = np.zeros((batch, width, module.DIMENSION), dtype=np.float32)
            hidden[:, 0, 0] = 3
            hidden[:, 1, 1] = 4
            # Deliberately enormous padding output must be excluded by mask.
            hidden[:, 2, :] = 999
            return [hidden]

    def disable_telemetry():
        state.telemetry_disabled = True

    ort = SimpleNamespace(disable_telemetry_events=disable_telemetry,
                          SessionOptions=lambda: SimpleNamespace(),
                          GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
                          InferenceSession=Session, __version__="fake-ort")
    tokenizer_runtime = SimpleNamespace(Tokenizer=Tokenizer, __version__="fake-tokenizers")
    imports = {"numpy": np, "onnxruntime": ort, "tokenizers": tokenizer_runtime}
    monkeypatch.setattr(module.importlib, "import_module", lambda name: imports[name])
    return state, np


def test_import_never_requires_optional_runtime_or_model(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "onnxruntime", "tokenizers", "transformers", "huggingface_hub"}:
            pytest.fail("Module import loaded an optional model runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("standalone_local_embeddings", module.__file__)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert isolated.DIMENSION == 384


@pytest.mark.parametrize("mutation", ["missing", "wrong_size", "wrong_hash"])
def test_files_are_verified_before_optional_imports(model_files, monkeypatch, mutation):
    target = model_files / module.MODEL_FILENAME
    if mutation == "missing":
        target.unlink()
    elif mutation == "wrong_size":
        target.write_bytes(b"short")
    else:
        target.write_bytes(b"x" * target.stat().st_size)
    monkeypatch.setattr(module.importlib, "import_module", lambda *_: pytest.fail("Unverified files reached runtime"))
    with pytest.raises(module.SemanticError) as caught:
        module.LocalEmbedder(model_files)
    assert str(model_files) not in str(caught.value)


def test_missing_dependencies_are_explicit_and_sanitized(model_files, monkeypatch):
    def unavailable(_):
        raise ImportError("sensitive-path /private/local/file")
    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    with pytest.raises(module.SemanticError) as caught:
        module.LocalEmbedder(model_files)
    assert "依赖" in str(caught.value) and "sensitive-path" not in str(caught.value)


def test_initialization_is_cpu_only_uses_tokenizer_pad_id_and_has_no_local_path(model_files, runtime):
    state, _ = runtime
    embedder = module.LocalEmbedder(model_files)
    assert state.telemetry_disabled
    assert state.truncation == {"max_length": 512}
    assert state.padding == {"direction": "right", "pad_id": 7, "pad_type_id": 0, "pad_token": "<pad>"}
    assert state.options.inter_op_num_threads == 1
    assert 1 <= state.options.intra_op_num_threads <= 4
    metadata = embedder.metadata
    assert embedder.dimension == 384 and len(embedder.fingerprint) == 64
    assert metadata["local_only"] is True and metadata["provider"] == "CPUExecutionProvider"
    assert str(model_files) not in str(metadata)
    metadata["files"].clear()
    assert embedder.metadata["files"]


def test_masked_mean_pooling_and_l2_normalization_exclude_padding(model_files, runtime):
    state, np = runtime
    embedder = module.LocalEmbedder(model_files)
    vector = embedder.embed_query("  文档保存多久？  ")
    assert len(vector) == 384
    assert vector[:2] == pytest.approx([.6, .8])
    assert vector[2:] == [0.0] * 382
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert state.encoded == [["query: 文档保存多久？"]]
    assert (state.feeds[0]["token_type_ids"] == 0).all()


def test_prefixes_and_bounded_batches_are_applied_to_every_text(model_files, runtime):
    state, _ = runtime
    embedder = module.LocalEmbedder(model_files)
    result = embedder.embed_documents([f"资料 {i}" for i in range(17)])
    assert len(result) == 17
    assert [len(values) for values in state.encoded] == [1] * 17
    assert all(text.startswith("passage: ") for values in state.encoded for text in values)
    assert embedder.embed_documents([]) == []
    assert len(state.encoded) == 17


def test_each_text_is_independent_of_batch_companions(model_files, runtime):
    state, np = runtime
    def batch_sensitive(feeds):
        count, width = feeds["input_ids"].shape
        hidden = np.zeros((count, width, module.DIMENSION), dtype=np.float32)
        hidden[:, :, 0] = 1
        hidden[:, :, 1] = count - 1
        return [hidden]
    state.output = batch_sensitive
    embedder = module.LocalEmbedder(model_files)
    alone = embedder.embed_documents(["same passage"])[0]
    together = embedder.embed_documents(["unrelated private passage", "same passage"])[1]
    assert together == alone
    assert embedder.metadata["batch_size"] == 1
    assert embedder.metadata["batch_policy"] == "one_text_per_inference_for_corpus_independence"


@pytest.mark.parametrize("names,kind", [
    (["input_ids"], "tensor(int64)"),
    (["input_ids", "attention_mask", "input_ids"], "tensor(int64)"),
    (["input_ids", "attention_mask", "arbitrary_remote_input"], "tensor(int64)"),
    (["input_ids", "attention_mask"], "tensor(float)"),
])
def test_unexpected_graph_inputs_fail_without_guessing(model_files, runtime, names, kind):
    state, _ = runtime
    state.input_names, state.input_type = names, kind
    with pytest.raises(module.SemanticError):
        module.LocalEmbedder(model_files)


def test_optional_token_type_ids_and_int32_inputs(model_files, runtime):
    state, np = runtime
    state.input_names, state.input_type = ["input_ids", "attention_mask"], "tensor(int32)"
    embedder = module.LocalEmbedder(model_files)
    embedder.embed_query("query")
    assert set(state.feeds[0]) == {"input_ids", "attention_mask"}
    assert all(value.dtype == np.int32 for value in state.feeds[0].values())


@pytest.mark.parametrize("pad", [None, -1, "0", True])
def test_invalid_pad_token_rejected(model_files, runtime, pad):
    state, _ = runtime
    state.pad_id = pad
    with pytest.raises(module.SemanticError):
        module.LocalEmbedder(model_files)


def test_runtime_initialization_exception_does_not_echo_source(model_files, runtime):
    state, _ = runtime
    state.init_error = "secret-source /private/weights"
    with pytest.raises(module.SemanticError) as caught:
        module.LocalEmbedder(model_files)
    assert "secret-source" not in str(caught.value) and "weights" not in str(caught.value)


@pytest.mark.parametrize("text", [None, "", "   ", "bad\x00text", "bad\ud800text", "a" * 2001])
def test_bad_queries_do_not_enter_tokenizer(model_files, runtime, text):
    state, _ = runtime
    embedder = module.LocalEmbedder(model_files)
    with pytest.raises(module.SemanticError):
        embedder.embed_query(text)
    assert state.encoded == []


@pytest.mark.parametrize("documents", ["not-list", ("tuple",), ["a" * 20001], ["x"] * 2001])
def test_document_size_and_count_bounds_precede_tokenization(model_files, runtime, documents):
    state, _ = runtime
    embedder = module.LocalEmbedder(model_files)
    with pytest.raises(module.SemanticError):
        embedder.embed_documents(documents)
    assert state.encoded == []


@pytest.mark.parametrize("ids,mask", [
    ([], []), ([11, 12], [1]), ([11, 12], [0, 0]),
    ([11, 12], [1, 2]), ([11, -1], [1, 1]),
    ([True, 12], [1, 1]), ([11] * 513, [1] * 513),
])
def test_malformed_tokenizer_output_never_reaches_inference(model_files, runtime, ids, mask):
    state, _ = runtime
    state.encoded_override = lambda texts: [SimpleNamespace(ids=ids, attention_mask=mask)]
    embedder = module.LocalEmbedder(model_files)
    with pytest.raises(module.SemanticError):
        embedder.embed_query("source-text-must-not-be-echoed")
    assert state.feeds == []


@pytest.mark.parametrize("kind", ["nan", "infinite", "zero", "dimension", "shape", "integer", "envelope"])
def test_invalid_tensors_are_explicit_failures_not_vectors(model_files, runtime, kind):
    state, np = runtime
    def output(feeds):
        batch, width = feeds["input_ids"].shape
        values = np.ones((batch, width, 384), dtype=np.float32)
        if kind == "nan": values[0, 0, 0] = np.nan
        elif kind == "infinite": values[0, 0, 0] = np.inf
        elif kind == "zero": values[:] = 0
        elif kind == "dimension": values = values[:, :, :383]
        elif kind == "shape": values = values[:, 0, :]
        elif kind == "integer": values = values.astype(np.int64)
        elif kind == "envelope": return []
        return [values]
    state.output = output
    embedder = module.LocalEmbedder(model_files)
    with pytest.raises(module.SemanticError) as caught:
        embedder.embed_query("private-source-text")
    assert "private-source-text" not in str(caught.value)
    assert len(state.feeds) == 1  # no inference retry


def test_fingerprint_tracks_artifacts_and_preprocessing(model_files, runtime, monkeypatch):
    original = module.LocalEmbedder(model_files).fingerprint
    specs = deepcopy(module.MODEL_FILES)
    specs["tokenizer.json"]["sha256"] = "a" * 64
    monkeypatch.setattr(module, "MODEL_FILES", specs)
    changed_artifact = module._fingerprint()[0]
    assert changed_artifact != original
    monkeypatch.setattr(module, "QUERY_PREFIX", "new-prefix: ")
    assert module._fingerprint()[0] != changed_artifact
