"use strict";

const $ = (id) => document.getElementById(id);
const modeNames = { hybrid: "词法混合", bm25: "BM25", tfidf: "字符 TF-IDF", semantic: "本地语义向量", semantic_hybrid: "向量＋词法" };
const semanticModes = new Set(["semantic", "semantic_hybrid"]);
const state = {
  user: null, kbs: [], documents: [], datasets: [], runs: [], selectedRuns: [],
  answer: "", model: null, retrieval: null, epoch: 0, sourceRequest: 0, modelRequest: 0, retrievalRequest: 0,
};
let toastTimer;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

function empty(container, title, description) {
  const box = node("div", null, "empty-state");
  box.append(node("h3", title));
  if (description) box.append(node("p", description));
  container.replaceChildren(box);
}

function visible(id, value) { $(id).classList.toggle("hidden", !value); }
function toast(text) {
  $("toast").textContent = text;
  visible("toast", true);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => visible("toast", false), 3500);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...options, headers });
  } catch {
    throw new Error("无法连接本地服务，请确认服务正在运行后重试。");
  }
  const text = await response.text();
  let payload;
  try { payload = text ? JSON.parse(text) : {}; } catch { payload = {}; }
  if (!response.ok) {
    let detail = payload.detail || `请求失败（HTTP ${response.status}）`;
    if (Array.isArray(detail)) detail = detail.map((item) => item.msg || String(item)).join("；");
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function clearSession() {
  state.epoch += 1;
  state.sourceRequest += 1;
  state.modelRequest += 1;
  state.retrievalRequest += 1;
  Object.assign(state, { user: null, kbs: [], documents: [], datasets: [], runs: [], selectedRuns: [], answer: "", model: null, retrieval: null });
  $("user-name").textContent = "访客";
  $("scope-label").textContent = "尚未登录";
  $("question").value = "";
  $("upload-form").reset();
  clearAnswer();
  $("answer-mode").value = "extractive";
  $("generated-option").disabled = true;
  $("generated-option").textContent = "模型回答 · 未配置";
  $("model-status").textContent = "登录后读取模型配置状态。";
  setSemanticModes([]);
  setRetrievalNote("登录后读取本地语义检索状态。");
  $("refresh-model").disabled = true;
  visible("case-panel", false);
  visible("new-kb", false);
  visible("global-alert", false);
  $("case-list").replaceChildren();
  $("comparison").replaceChildren();
  $("document-list").replaceChildren();
  $("run-history").replaceChildren();
  $("source-chunks").replaceChildren();
  $("source-title").textContent = "证据来源";
  $("source-excerpt").textContent = "";
  $("source-kind").textContent = "";
  visible("source-kind", false);
  $("kb-options").replaceChildren();
  for (const id of ["dataset", "upload-kb", "document-filter"]) $(id).replaceChildren();
  for (const id of ["stat-kbs", "stat-docs", "stat-chunks"]) $(id).textContent = "—";
  $("document-count").textContent = "0";
  for (const id of ["ask-button", "search-button", "create-kb-button", "refresh-documents", "refresh-runs"]) $(id).disabled = false;
  $("upload-button").disabled = true;
  $("run-evaluation").disabled = true;
  $("ask-note").textContent = "问答仅使用当前账号可访问、且在右侧勾选的知识库。";
  $("upload-note").textContent = "文件在本机解析，收录完成后即可检索。";
  $("evaluation-note").textContent = "数据集标签为 AI 辅助、来源核对的草稿，待独立人工复核。";
  for (const dialog of ["source-drawer", "kb-dialog"]) if ($(dialog).open) $(dialog).close();
}

function requireLogin(message) {
  clearSession();
  if (message) {
    $("login-error").textContent = message;
    visible("login-error", true);
  }
  if (!$("login-dialog").open) $("login-dialog").showModal();
}

function report(error, area = "global-alert", epoch = state.epoch) {
  if (epoch !== state.epoch) return;
  if (error.status === 401) {
    requireLogin("登录已失效，请重新进入工作空间。");
    return;
  }
  $(area).textContent = error.message || "操作失败，请重试。";
  visible(area, true);
}

function dateText(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value || "时间未提供") : date.toLocaleString("zh-CN", { hour12: false });
}

function locationText(item) {
  const parts = [];
  if (item.page !== null && item.page !== undefined) parts.push(`第 ${item.page} 页`);
  if (item.section) parts.push(item.section);
  if (item.line_start !== null && item.line_start !== undefined) {
    parts.push(item.line_end && item.line_end !== item.line_start ? `第 ${item.line_start}–${item.line_end} 行` : `第 ${item.line_start} 行`);
  }
  return parts.join(" · ") || "文档片段";
}

function kbName(id) { return state.kbs.find((kb) => kb.id === id)?.name || id || "知识库"; }

function updateStats() {
  $("stat-kbs").textContent = state.kbs.length;
  $("stat-docs").textContent = state.documents.length;
  $("stat-chunks").textContent = state.kbs.reduce((sum, kb) => sum + (Number(kb.chunk_count) || 0), 0).toLocaleString();
}

function renderKbs(previousSelection = null) {
  const options = [];
  for (const kb of state.kbs) {
    const label = node("label", null, "kb-choice");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = kb.id;
    input.checked = previousSelection ? previousSelection.includes(kb.id) : true;
    const text = node("span");
    text.append(node("strong", kb.name), node("small", `${kb.document_count ?? 0} 份文档 · ${kb.chunk_count ?? 0} 个片段`));
    label.append(input, text);
    options.push(label);
  }
  $("kb-options").replaceChildren(...options);
  if (!options.length) empty($("kb-options"), "暂无可访问知识库", "管理员可以先创建一个知识库。");
  const previousUpload = $("upload-kb").value;
  const previousFilter = $("document-filter").value;
  $("upload-kb").replaceChildren(...state.kbs.map((kb) => new Option(kb.name, kb.id)));
  $("document-filter").replaceChildren(new Option("全部知识库", ""), ...state.kbs.map((kb) => new Option(kb.name, kb.id)));
  if (state.kbs.some((kb) => kb.id === previousUpload)) $("upload-kb").value = previousUpload;
  if (state.kbs.some((kb) => kb.id === previousFilter)) $("document-filter").value = previousFilter;
  $("upload-button").disabled = state.kbs.length === 0;
  $("scope-label").textContent = state.user?.role === "admin" ? "管理员 · 全部知识库" : `${state.user?.display_name || "成员"} · ${state.kbs.length} 个知识库`;
  updateStats();
}

async function loadLibrary(epoch = state.epoch) {
  const selected = [...$("kb-options").querySelectorAll("input:checked")].map((input) => input.value);
  const hadKbs = state.kbs.length > 0;
  try {
    const [libraries, documents] = await Promise.all([api("/api/kbs"), api("/api/documents")]);
    if (epoch !== state.epoch) return;
    state.kbs = libraries.items || [];
    state.documents = documents.items || [];
    renderKbs(hadKbs ? selected : null);
    renderDocuments();
  } catch (error) {
    if (epoch !== state.epoch) return;
    empty($("document-list"), "文档加载失败", error.message);
    report(error, "global-alert", epoch);
  }
}

function renderDocuments() {
  const filter = $("document-filter").value;
  const documents = state.documents.filter((item) => !filter || item.kb_id === filter);
  $("document-count").textContent = documents.length;
  const rows = documents.map((document) => {
    const row = node("div", null, "document-row");
    const extension = document.name?.split(".").pop()?.toUpperCase().slice(0, 5) || "DOC";
    row.append(node("span", extension, "document-icon"));
    const info = node("div", null, "document-info");
    info.append(node("div", document.name, "document-name"));
    const page = document.page_count ? ` · ${document.page_count} 页` : "";
    info.append(node("p", `${kbName(document.kb_id)} · ${document.chunk_count ?? 0} 个片段${page} · ${dateText(document.created_at)}`));
    const actions = node("div", null, "document-actions");
    const inspect = node("button", "查看片段", "text-button");
    inspect.addEventListener("click", () => showSource({ document_id: document.id, document_name: document.name, kb_id: document.kb_id }));
    const download = node("a", "原文件 ↗", "text-button");
    download.href = `/api/documents/${encodeURIComponent(document.id)}/source`;
    download.target = "_blank";
    download.rel = "noopener";
    const remove = node("button", "删除", "text-button danger-button");
    remove.addEventListener("click", () => removeDocument(document, remove));
    actions.append(inspect, download, remove);
    row.append(info, actions);
    return row;
  });
  $("document-list").replaceChildren(...rows);
  if (!rows.length) empty($("document-list"), "这里还没有文档", "选择上方的文件，上传到一个可访问知识库。");
}

async function removeDocument(document, button) {
  if (!window.confirm(`删除「${document.name}」及其检索片段？`)) return;
  const epoch = state.epoch;
  button.disabled = true;
  try {
    await api(`/api/documents/${encodeURIComponent(document.id)}`, { method: "DELETE" });
    if (epoch !== state.epoch) return;
    clearAnswer();
    if ($("source-drawer").open) $("source-drawer").close();
    await loadLibrary(epoch);
    toast("文档已删除");
  } catch (error) { report(error, "global-alert", epoch); }
  finally { button.disabled = false; }
}

function clearAnswer() {
  state.answer = "";
  $("answer-content").textContent = "";
  $("sources").replaceChildren();
  $("candidates").replaceChildren();
  $("answer-provenance").textContent = "";
  $("evidence-decision").textContent = "";
  $("result-meta").textContent = "";
  visible("copy-answer", false);
  visible("answer-content", false);
  visible("answer-empty", true);
  visible("answer-provenance", false);
  visible("evidence-decision", false);
  visible("sources-heading", false);
  visible("candidate-section", false);
}

function renderSources(hits) {
  const cards = hits.map((hit, index) => {
    const card = node("button", null, "source-card");
    card.type = "button";
    const top = node("div", null, "source-card-top");
    top.append(node("span", hit.citation_id || "原文", "citation-id"), node("strong", hit.document_name));
    top.append(node("span", `来源 ${index + 1} ↗`, "small-note"));
    card.append(top, node("p", hit.text), node("div", `${kbName(hit.kb_id)} · ${locationText(hit)}`, "small-note"));
    card.addEventListener("click", () => showSource(hit));
    return card;
  });
  $("sources").replaceChildren(...cards);
  visible("sources-heading", cards.length > 0);
}

function renderCandidates(hits, searchOnly = false) {
  const cards = hits.map((hit, index) => {
    const card = node("button", null, "candidate-card");
    card.type = "button";
    const top = node("div", null, "source-card-top");
    top.append(node("span", "未验证", "candidate-tag"), node("strong", hit.document_name));
    top.append(node("span", `候选 ${index + 1} ↗`, "small-note"));
    card.append(top, node("p", hit.text), node("div", `${kbName(hit.kb_id)} · ${locationText(hit)}`, "small-note"));
    card.addEventListener("click", () => showSource(hit, true));
    return card;
  });
  $("candidate-title").textContent = searchOnly ? "检索候选 · 未验证" : "未验证候选";
  $("candidate-explanation").textContent = searchOnly
    ? "以下片段仅按相关度返回，尚未作为任何回答的引用依据。"
    : "以下片段仅是检索候选，尚未作为当前回答的已采用引用；需展开原文另行核对。";
  $("candidates").replaceChildren(...cards);
  visible("candidate-section", cards.length > 0);
}

async function loadModel(epoch = state.epoch) {
  if (!state.user) return;
  const requestId = ++state.modelRequest;
  $("refresh-model").disabled = true;
  $("model-status").textContent = "正在读取服务端模型配置…";
  try {
    const model = await api("/api/model");
    if (epoch !== state.epoch || requestId !== state.modelRequest) return;
    state.model = model;
    $("generated-option").disabled = !model.configured;
    $("generated-option").textContent = model.configured ? "模型回答" : "模型回答 · 未配置";
    if (!model.configured) {
      $("answer-mode").value = "extractive";
      $("model-status").textContent = "模型未配置，当前可使用离线证据摘录。";
    } else {
      const provider = model.provider === "openai_compatible" && String(model.model || "").startsWith("deepseek-")
        ? "DeepSeek"
        : { openai_compatible: "OpenAI 兼容", ollama: "Ollama" }[model.provider] || model.provider || "模型服务";
      const verification = model.live_verified
        ? model.last_success_scope === "isolated_synthetic_preflight"
          ? "合成资料联调已通过（历史记录）"
          : "服务端已记录成功调用"
        : "已配置，尚未实测连接";
      $("model-status").textContent = `${provider}${model.model ? ` · ${model.model}` : ""} · ${verification}。配置状态不代表服务当前可用。`;
    }
  } catch (error) {
    if (epoch !== state.epoch || requestId !== state.modelRequest) return;
    state.model = null;
    $("answer-mode").value = "extractive";
    $("generated-option").disabled = true;
    $("generated-option").textContent = "模型回答 · 状态不可用";
    $("model-status").textContent = `模型状态读取失败：${error.message}。离线证据摘录仍可选择。`;
    if (error.status === 401) report(error, "global-alert", epoch);
  } finally {
    if (epoch === state.epoch && requestId === state.modelRequest) $("refresh-model").disabled = false;
  }
}

function setSemanticModes(enabledModes) {
  for (const id of ["ask-mode", "evaluation-mode"]) {
    const select = $(id);
    for (const option of select.options) {
      if (semanticModes.has(option.value)) option.disabled = !enabledModes.includes(option.value);
    }
    if (semanticModes.has(select.value) && select.selectedOptions[0]?.disabled) select.value = "hybrid";
  }
}

function setRetrievalNote(text) {
  $("retrieval-status").textContent = text;
  $("evaluation-retrieval-status").textContent = text;
}

function retrievalModeReady(mode) {
  return !semanticModes.has(mode)
    || (state.retrieval?.available === true && state.retrieval.modes?.includes(mode));
}

async function loadRetrieval(epoch = state.epoch) {
  if (!state.user) return;
  const requestId = ++state.retrievalRequest;
  setRetrievalNote("正在读取本地语义检索状态…");
  try {
    const retrieval = await api("/api/retrieval");
    if (epoch !== state.epoch || requestId !== state.retrievalRequest) return;
    const modes = retrieval.available === true && Array.isArray(retrieval.modes)
      ? retrieval.modes.filter((mode) => semanticModes.has(mode)) : [];
    state.retrieval = { ...retrieval, available: modes.length > 0, modes };
    setSemanticModes(modes);
    if (modes.length) {
      const name = typeof retrieval.model?.model_id === "string" ? retrieval.model.model_id : "本地向量模型";
      setRetrievalNote(`${name} 已就绪。向量在本机计算；相似候选仍需核对原文。`);
    } else {
      setRetrievalNote("语义检索尚未启用，当前可用关键词与字符检索。");
    }
  } catch (error) {
    if (epoch !== state.epoch || requestId !== state.retrievalRequest) return;
    state.retrieval = null;
    setSemanticModes([]);
    setRetrievalNote(error.status === 404
      ? "当前服务尚未启用语义检索，关键词与字符检索仍可使用。"
      : "语义检索状态暂不可用，当前可用关键词与字符检索。");
    if (error.status === 401) report(error, "global-alert", epoch);
  }
}

async function ask(searchOnly = false) {
  if (!$("question-form").reportValidity()) return;
  const ids = [...$("kb-options").querySelectorAll("input:checked")].map((input) => input.value);
  if (!ids.length) { $("ask-note").textContent = "请先勾选至少一个知识库。"; return; }
  const question = $("question").value.trim();
  if (!question) return;
  const answerMode = $("answer-mode").value;
  if (!retrievalModeReady($("ask-mode").value)) {
    $("ask-note").textContent = "本地语义检索尚未就绪，请选择关键词、字符或词法混合检索。";
    return;
  }
  if (!searchOnly && answerMode === "generated" && !state.model?.configured) {
    $("ask-note").textContent = "模型尚未配置或状态未加载，请选择离线证据摘录。";
    return;
  }
  const epoch = state.epoch;
  $("ask-button").disabled = true;
  $("search-button").disabled = true;
  visible("global-alert", false);
  clearAnswer();
  $("ask-note").textContent = searchOnly ? "正在检索文档片段…" : answerMode === "generated" ? "正在检索证据并请求已配置模型…" : "正在检索并整理证据摘录…";
  try {
    const body = { question, kb_ids: ids, mode: $("ask-mode").value, top_k: Number($("ask-topk").value) };
    if (!searchOnly) body.answer_mode = answerMode;
    const result = await api(searchOnly ? "/api/search" : "/api/ask", {
      method: "POST", body: JSON.stringify(body),
    });
    if (epoch !== state.epoch) return;
    visible("answer-empty", false);
    const hits = searchOnly ? [] : result.sources || [];
    if (searchOnly) {
      const candidates = result.items || [];
      $("answer-content").textContent = candidates.length ? `检索到 ${candidates.length} 个候选片段。点击候选卡片查看原文位置。仅检索不会生成回答或验证这些片段能否支持结论。` : "没有检索到匹配证据，可以换一个更具体的问题。";
      $("answer-content").classList.remove("refused");
      renderCandidates(candidates, true);
    } else {
      state.answer = result.answer || "服务未返回回答文本。";
      $("answer-content").textContent = state.answer;
      $("answer-content").classList.toggle("refused", Boolean(result.refused));
      visible("copy-answer", Boolean(result.answer));
      if (result.refused) renderCandidates(result.retrieval_candidates || []);
      const provenance = result.answer_mode === "generated"
        ? "模型回答 · 引用格式已核对（未验证语义支持）"
        : "离线证据摘录 · 未调用生成模型";
      $("answer-provenance").textContent = provenance;
      visible("answer-provenance", true);
      const decision = result.evidence_decision?.message;
      if (decision) {
        $("evidence-decision").textContent = `证据判定 · ${decision}`;
        visible("evidence-decision", true);
      }
    }
    visible("answer-content", true);
    renderSources(hits);
    const elapsed = Number.isFinite(Number(result.elapsed_ms)) ? `${Number(result.elapsed_ms).toFixed(1)} ms` : "耗时未提供";
    $("result-meta").textContent = `${searchOnly ? "仅检索" : result.answer_mode === "generated" ? "模型回答" : "离线摘录"} · ${elapsed}`;
    $("ask-note").textContent = result.refused ? "本次已拒答。若有检索候选，它们会在独立区域中标记为未验证。" : "检索完成。引用位置由文档解析结果提供，可点击来源核对。";
  } catch (error) {
    if (epoch !== state.epoch) return;
    $("ask-note").textContent = !searchOnly && answerMode === "generated" ? `模型回答未生成：${error.message}` : error.message;
    report(error, "global-alert", epoch);
  } finally {
    if (epoch === state.epoch) {
      $("ask-button").disabled = false;
      $("search-button").disabled = false;
    }
  }
}

async function showSource(hit, unverified = false) {
  const epoch = state.epoch;
  const requestId = ++state.sourceRequest;
  $("source-title").textContent = hit.document_name || "证据来源";
  $("source-kind").textContent = unverified ? "未验证检索候选 · 此片段尚未作为回答的引用依据。" : "";
  visible("source-kind", unverified);
  $("source-location").textContent = `${kbName(hit.kb_id)} · ${locationText(hit)}`;
  $("source-excerpt").textContent = hit.text || "下方列出此文档的解析片段。";
  $("source-download").href = `/api/documents/${encodeURIComponent(hit.document_id)}/source`;
  visible("source-original", false);
  if (hit.source_url) {
    try {
      const url = new URL(hit.source_url);
      if (["http:", "https:"].includes(url.protocol)) {
        $("source-original").href = url.href;
        visible("source-original", true);
      }
    } catch { /* A source URL is optional; malformed values stay unlinked. */ }
  }
  empty($("source-chunks"), "正在读取原文片段");
  if (!$("source-drawer").open) $("source-drawer").showModal();
  try {
    const page = hit.page !== null && hit.page !== undefined ? `?page=${encodeURIComponent(hit.page)}` : "";
    const result = await api(`/api/documents/${encodeURIComponent(hit.document_id)}/chunks${page}`);
    if (epoch !== state.epoch || requestId !== state.sourceRequest) return;
    const chunks = (result.items || []).map((chunk) => {
      const card = node("article", null, "chunk-card");
      card.append(node("small", locationText(chunk)), node("p", chunk.text));
      return card;
    });
    $("source-chunks").replaceChildren(...chunks);
    if (!chunks.length) empty($("source-chunks"), "没有可显示的片段", "文档可能已删除，或当前账号无权访问。");
  } catch (error) {
    if (epoch !== state.epoch || requestId !== state.sourceRequest) return;
    empty($("source-chunks"), "来源加载失败", error.message);
    if (error.status === 401) report(error, "global-alert", epoch);
  }
}

function renderDatasets() {
  $("dataset").replaceChildren(...state.datasets.map((dataset) => new Option(`${dataset.name} · ${dataset.size} 题`, dataset.id)));
  if (!state.datasets.length) $("dataset").append(new Option("暂无可用数据集", ""));
  $("run-evaluation").disabled = state.datasets.length === 0;
  updateDatasetDescription();
}

function updateDatasetDescription() {
  const dataset = state.datasets.find((item) => item.id === $("dataset").value);
  $("dataset-description").textContent = dataset ? `${dataset.answerable} 可回答 · ${dataset.unanswerable} 不可回答${dataset.description ? ` · ${dataset.description}` : ""}` : "";
}

async function loadEvaluations(epoch = state.epoch) {
  try {
    const [datasets, runs] = await Promise.all([api("/api/evaluations/datasets"), api("/api/evaluations/runs")]);
    if (epoch !== state.epoch) return;
    state.datasets = datasets.items || [];
    state.runs = runs.items || [];
    state.selectedRuns = state.selectedRuns.filter((id) => state.runs.some((run) => run.id === id));
    renderDatasets();
    renderRuns();
    if (!state.selectedRuns.length && state.runs.length) await selectRun(state.runs[0].id, true);
    else await renderComparison();
  } catch (error) {
    if (epoch !== state.epoch) return;
    empty($("run-history"), "评测记录加载失败", error.message);
    report(error, "global-alert", epoch);
  }
}

function renderRuns() {
  const rows = state.runs.map((run) => {
    const row = node("div", null, `run-row${state.selectedRuns.includes(run.id) ? " selected" : ""}`);
    const label = node("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedRuns.includes(run.id);
    checkbox.addEventListener("change", () => selectRun(run.id, checkbox.checked));
    label.append(checkbox, node("span", modeNames[run.mode] || run.mode || "评测运行"));
    row.append(label, node("p", dateText(run.created_at)));
    row.append(node("p", `Top ${run.top_k ?? "—"} · ${{ dev: "开发集", test: "测试集", all: "全部题目" }[run.split] || "分组未提供"} · ${run.scope?.case_count ?? run.cases?.length ?? "—"} 题`));
    const detail = node("button", "查看逐题结果 ↗", "text-button");
    detail.addEventListener("click", () => showCases(run.id));
    row.append(detail);
    return row;
  });
  $("run-history").replaceChildren(...rows);
  if (!rows.length) empty($("run-history"), "暂无运行记录", "选择数据集与模式，开始第一次评测。");
}

async function selectRun(id, selected) {
  if (selected && !state.selectedRuns.includes(id)) {
    if (state.selectedRuns.length >= 2) { toast("最多同时比较两次运行，请先取消一项。"); renderRuns(); return; }
    state.selectedRuns.push(id);
  } else if (!selected) state.selectedRuns = state.selectedRuns.filter((value) => value !== id);
  renderRuns();
  await renderComparison();
}

async function fullRun(id) {
  const existing = state.runs.find((run) => run.id === id);
  if (existing?.cases && existing?.metrics) return existing;
  const epoch = state.epoch;
  const run = await api(`/api/evaluations/runs/${encodeURIComponent(id)}`);
  if (epoch === state.epoch) {
    const index = state.runs.findIndex((value) => value.id === id);
    if (index >= 0) state.runs[index] = run;
  }
  return run;
}

const metrics = [
  ["recall_at_1", "Recall @ 1", "percent", false], ["recall_at_3", "Recall @ 3", "percent", false],
  ["recall_at_5", "Recall @ 5", "percent", false], ["mrr", "MRR", "decimal", false],
  ["refusal_accuracy", "拒答准确率", "percent", false], ["false_refusal_rate", "误拒答率", "percent", true],
  ["expected_source_use_rate", "预期来源实际采用率", "percent", false],
  ["unsupported_acceptance_rate", "无答案接受率", "percent", true],
  ["p50_ms", "P50 延迟", "ms", true], ["p95_ms", "P95 延迟", "ms", true],
];

function metricText(value, format) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  if (format === "percent") return `${(Number(value) * 100).toFixed(1)}%`;
  if (format === "ms") return `${Number(value).toFixed(1)} ms`;
  return Number(value).toFixed(3);
}

async function renderComparison() {
  const epoch = state.epoch;
  const ids = [...state.selectedRuns];
  if (!ids.length) { empty($("comparison"), "用数据看见差异", "运行一次评测，或在左侧选择已有记录。"); return; }
  try {
    const runs = await Promise.all(ids.map(fullRun));
    if (epoch !== state.epoch || ids.join() !== state.selectedRuns.join()) return;
    const table = node("table", null, "metric-table");
    const head = node("thead");
    const titleRow = node("tr");
    titleRow.append(node("th", "指标"));
    runs.forEach((run) => {
      const label = node("th", modeNames[run.mode] || run.mode);
      label.append(node("div", `Top ${run.top_k ?? "—"} · ${run.split || "all"}`, "small-note"));
      titleRow.append(label);
    });
    head.append(titleRow);
    const body = node("tbody");
    for (const [key, title, format, lower] of metrics) {
      const row = node("tr");
      row.append(node("td", title));
      const values = runs.map((run) => run.metrics?.[key]);
      runs.forEach((run, index) => {
        const cell = node("td", metricText(values[index], format));
        if (values.length === 2 && values.every((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))) {
          const own = Number(values[index]);
          const other = Number(values[1 - index]);
          if (lower ? own < other : own > other) cell.classList.add("better");
        }
        row.append(cell);
      });
      body.append(row);
    }
    table.append(head, body);
    $("comparison").replaceChildren(table, node("p", "仅比较检索、拒答与延迟。不同数据集、Top K 或权限范围的结果不宜直接比较。— 表示服务未提供该指标。", "comparison-footnote"));
  } catch (error) {
    if (epoch !== state.epoch || ids.join() !== state.selectedRuns.join()) return;
    empty($("comparison"), "指标加载失败", error.message);
    if (error.status === 401) report(error, "global-alert", epoch);
  }
}

async function showCases(id) {
  const epoch = state.epoch;
  try {
    const run = await fullRun(id);
    if (epoch !== state.epoch) return;
    $("case-title").textContent = `${modeNames[run.mode] || run.mode} · 逐题结果`;
    $("download-evaluation").href = `/api/evaluations/runs/${encodeURIComponent(id)}/download`;
    const rows = (run.cases || []).map((item) => {
      const row = node("tr");
      row.append(node("td", item.question), node("td", item.expected_document || "应拒答"), node("td", item.hit_rank ?? "—"), node("td", item.refused ? "是" : "否"), node("td", metricText(item.latency_ms, "ms")));
      const result = node("td");
      result.append(node("span", item.passed ? "通过" : "未通过", `pass-badge${item.passed ? "" : " fail"}`));
      row.append(result);
      return row;
    });
    $("case-list").replaceChildren(...rows);
    visible("case-panel", true);
    $("case-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { report(error, "global-alert", epoch); }
}

async function loadWorkspace() {
  const epoch = state.epoch;
  $("user-name").textContent = state.user.display_name || state.user.id;
  visible("new-kb", state.user.role === "admin");
  await Promise.all([loadLibrary(epoch), loadEvaluations(epoch), loadModel(epoch), loadRetrieval(epoch)]);
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  const view = button.dataset.view;
  document.querySelectorAll(".view").forEach((section) => section.classList.toggle("hidden", section.id !== `view-${view}`));
  document.querySelectorAll("[data-view]").forEach((item) => {
    item.classList.toggle("selected", item === button);
    if (item === button) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
  });
  $("page-label").textContent = { ask: "证据问答", library: "知识库", evaluation: "检索评测" }[view];
}));

document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  document.querySelector('[data-view="ask"]').click();
});
$("login-dialog").addEventListener("cancel", (event) => event.preventDefault());
$("demo-account").addEventListener("change", () => {
  $("login-username").value = $("demo-account").value;
  $("login-password").value = `${$("demo-account").value}-demo`;
});
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-button").disabled = true;
  visible("login-error", false);
  const epoch = state.epoch;
  try {
    const result = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: $("login-username").value, password: $("login-password").value }) });
    if (epoch !== state.epoch) return;
    state.user = result.user;
    $("login-dialog").close();
    await loadWorkspace();
  } catch (error) { report(error, "login-error", epoch); }
  finally { $("login-button").disabled = false; }
});
$("switch-user").addEventListener("click", async () => {
  $("switch-user").disabled = true;
  const epoch = state.epoch;
  try {
    await api("/api/auth/logout", { method: "POST" });
    if (epoch !== state.epoch) return;
    visible("login-error", false);
    requireLogin();
  } catch (error) { report(error, "global-alert", epoch); }
  finally { $("switch-user").disabled = false; }
});
$("question-form").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
$("search-button").addEventListener("click", () => ask(true));
$("refresh-model").addEventListener("click", () => Promise.all([loadModel(), loadRetrieval()]));
$("select-all-kbs").addEventListener("click", () => {
  $("kb-options").querySelectorAll("input").forEach((input) => { input.checked = true; });
});
$("copy-answer").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(state.answer); toast("已复制证据回答"); }
  catch { toast("复制失败，可以选中文字后手动复制。"); }
});
$("close-source").addEventListener("click", () => { state.sourceRequest += 1; $("source-drawer").close(); });
$("source-drawer").addEventListener("cancel", () => { state.sourceRequest += 1; });
$("document-filter").addEventListener("change", renderDocuments);
$("refresh-documents").addEventListener("click", async () => {
  visible("global-alert", false);
  $("refresh-documents").disabled = true;
  await loadLibrary();
  $("refresh-documents").disabled = false;
});
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) return;
  const epoch = state.epoch;
  const body = new FormData();
  body.append("file", file);
  body.append("kb_id", $("upload-kb").value);
  if ($("source-url").value.trim()) body.append("source_url", $("source-url").value.trim());
  $("upload-button").disabled = true;
  visible("global-alert", false);
  $("upload-note").textContent = "正在上传并解析文档，请稍候…";
  try {
    const result = await api("/api/documents", { method: "POST", body });
    if (epoch !== state.epoch) return;
    $("upload-note").textContent = result.document?.duplicate ? "此知识库已收录相同内容，已复用原文档。" : `已收录 ${result.document?.name || file.name}，共 ${result.document?.chunk_count ?? 0} 个片段。`;
    $("upload-file").value = "";
    await loadLibrary(epoch);
  } catch (error) {
    if (epoch === state.epoch) $("upload-note").textContent = error.message;
    if (error.status === 401) report(error, "global-alert", epoch);
  } finally { if (epoch === state.epoch) $("upload-button").disabled = state.kbs.length === 0; }
});
$("new-kb").addEventListener("click", () => { $("kb-form").reset(); visible("kb-error", false); $("kb-dialog").showModal(); });
$("close-kb").addEventListener("click", () => $("kb-dialog").close());
$("kb-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const epoch = state.epoch;
  $("create-kb-button").disabled = true;
  try {
    await api("/api/kbs", { method: "POST", body: JSON.stringify({ name: $("kb-name").value.trim(), description: $("kb-description").value.trim() }) });
    if (epoch !== state.epoch) return;
    $("kb-dialog").close();
    await loadLibrary(epoch);
    toast("知识库已创建");
  } catch (error) { report(error, "kb-error", epoch); }
  finally { $("create-kb-button").disabled = false; }
});
$("dataset").addEventListener("change", updateDatasetDescription);
$("refresh-runs").addEventListener("click", async () => {
  visible("global-alert", false);
  $("refresh-runs").disabled = true;
  await loadEvaluations();
  $("refresh-runs").disabled = false;
});
$("evaluation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!retrievalModeReady($("evaluation-mode").value)) {
    $("evaluation-note").textContent = "本地语义检索尚未就绪，请选择关键词、字符或词法混合检索。";
    return;
  }
  const epoch = state.epoch;
  $("run-evaluation").disabled = true;
  visible("global-alert", false);
  $("evaluation-note").textContent = "正在本机运行评测，完成后将显示真实指标…";
  try {
    const run = await api("/api/evaluations/run", { method: "POST", body: JSON.stringify({ dataset_id: $("dataset").value, mode: $("evaluation-mode").value, top_k: Number($("evaluation-topk").value), split: $("evaluation-split").value || null }) });
    if (epoch !== state.epoch) return;
    state.selectedRuns = [run.id];
    await loadEvaluations(epoch);
    if (epoch !== state.epoch) return;
    $("evaluation-note").textContent = `运行完成 · ${run.cases?.length ?? "—"} 题 · ${metricText(run.duration_ms, "ms")}。标签为 AI 辅助、来源核对的草稿，待独立人工复核。`;
    await showCases(run.id);
  } catch (error) {
    if (epoch === state.epoch) $("evaluation-note").textContent = error.message;
    if (error.status === 401) report(error, "global-alert", epoch);
  } finally { if (epoch === state.epoch) $("run-evaluation").disabled = state.datasets.length === 0; }
});

async function boot() {
  api("/api/health").then((health) => {
    $("health-status").textContent = health.model_connected ? "已连接模型" : "离线模式 · 本地检索";
    $("health-status").classList.remove("error");
  }).catch((error) => {
    $("health-status").textContent = "服务连接失败";
    $("health-status").classList.add("error");
    $("health-status").title = error.message;
  });
  const epoch = state.epoch;
  try {
    const result = await api("/api/auth/me");
    if (epoch !== state.epoch) return;
    state.user = result.user;
    if (!state.user) requireLogin(); else await loadWorkspace();
  } catch (error) {
    if (epoch !== state.epoch) return;
    requireLogin(error.status === 401 ? "" : error.message);
  }
}

boot();
