/* Real offline v0.2 UI checks in a new, isolated headless Edge profile. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const server = new URL(process.env.KNOWTRACE_TEST_URL || 'http://127.0.0.1:8767');
assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(server.hostname), 'Tests require a loopback server');
assert.notEqual(server.port, '8766', 'Do not run mutation tests against the existing 8766 workspace');
assert.equal(server.pathname, '/', 'Use the test server origin without a path');
const base = server.origin;
const output = path.resolve(__dirname, '../artifacts/browser-v2');
fs.mkdirSync(output, { recursive: true });

async function localOnly(context) {
  await context.route('**/*', route => {
    const request = new URL(route.request().url());
    return request.origin === base ? route.continue() : route.abort();
  });
}

async function login(page, username) {
  await page.locator('#login-dialog').waitFor({ state: 'visible' });
  await page.locator('#demo-account').selectOption(username);
  await page.locator('#login-button').click();
  await page.locator('#login-dialog').waitFor({ state: 'hidden' });
  await page.waitForFunction(() => document.querySelector('#stat-docs').textContent !== '—');
  await page.waitForFunction(() => !document.querySelector('#refresh-model').disabled);
}

async function question(page, text, button = '#ask-button') {
  await page.locator('#question').fill(text);
  const [response] = await Promise.all([
    page.waitForResponse(value => value.url() === base + (button === '#search-button' ? '/api/search' : '/api/ask') && value.request().method() === 'POST'),
    page.locator(button).click(),
  ]);
  assert.ok(response.ok(), `Question returned HTTP ${response.status()}`);
  const result = await response.json();
  await page.waitForFunction(() => !document.querySelector('#ask-button').disabled);
  return { result, request: response.request().postDataJSON() };
}

function trackPageErrors(page, errors) {
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 500) errors.push(`HTTP ${response.status()} ${new URL(response.url()).pathname}`);
  });
}

async function checkMobile(browser, checks, errors) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  try {
    await localOnly(context);
    const page = await context.newPage();
    trackPageErrors(page, errors);
    await page.goto(base);
    await login(page, 'engineer');
    const bounds = async id => page.locator(id).boundingBox();
    const mode = await bounds('#ask-mode');
    const answer = await bounds('#answer-mode');
    const topk = await bounds('#ask-topk');
    const search = await bounds('#search-button');
    const ask = await bounds('#ask-button');
    assert.ok(mode.width >= 200 && answer.width >= 240, 'Mobile selectors must show readable labels');
    assert.ok(Math.abs(mode.y - topk.y) < 2, 'Retrieval settings share a row');
    assert.ok(search.y >= mode.y + mode.height, 'Action buttons have a separate row');
    assert.ok(Math.abs(search.y - ask.y) < 2 && Math.abs(search.width - ask.width) < 2);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await page.screenshot({ path: path.join(output, 'mobile-ask.png'), fullPage: true });
    await page.locator('[data-view="evaluation"]').click();
    const evaluationMode = await bounds('#evaluation-mode');
    const split = await bounds('#evaluation-split');
    assert.ok(evaluationMode.width >= 140 && split.width >= 140, 'Mobile evaluation options remain readable');
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await page.screenshot({ path: path.join(output, 'mobile-evaluation.png'), fullPage: true });
    checks.push('390px answer/retrieval selectors, two-row actions, evaluation settings and no page overflow');
  } finally {
    await context.close();
  }
}

async function checkDelayedAccountSwitch(page, checks) {
  await page.locator('[data-view="ask"]').click();
  let ready;
  let release;
  let finished;
  const fetched = new Promise(resolve => { ready = resolve; });
  const delivered = new Promise(resolve => { finished = resolve; });
  await page.route('**/api/ask', async route => {
    const response = await route.fetch();
    ready();
    await new Promise(resolve => { release = resolve; });
    await route.fulfill({ response });
    finished();
  });
  try {
    await page.locator('#question').fill('路径参数');
    await page.locator('#ask-button').click();
    await fetched;
    await page.locator('#switch-user').click();
    await login(page, 'support');
    release();
    await delivered;
    await page.waitForFunction(() => document.querySelector('#answer-content').classList.contains('hidden'));
    assert.equal(await page.locator('#sources .source-card').count(), 0);
    assert.equal(await page.locator('#candidates .candidate-card').count(), 0);
    assert.equal(await page.locator('#source-excerpt').textContent(), '');
    assert.equal(await page.locator('#answer-mode').inputValue(), 'extractive');
    const documents = await page.request.get(base + '/api/documents');
    assert.ok((await documents.json()).items.every(document => document.kb_id === 'kb-support'));
    checks.push('Actual earlier-scope response is delayed, then ignored after account switch; model mode resets');
  } finally {
    if (release) release();
    await page.unroute('**/api/ask');
  }
}

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const checks = [];
  const errors = [];
  let context;
  try {
    context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    context.setDefaultTimeout(20000);
    await localOnly(context);
    const page = await context.newPage();
    trackPageErrors(page, errors);
    await page.goto(base);
    await login(page, 'engineer');

    const model = await page.request.get(base + '/api/model');
    assert.ok(model.ok());
    const configuration = await model.json();
    assert.equal(configuration.configured, false, 'This script requires the isolated, unconfigured offline test service');
    assert.equal(configuration.live_verified, false);
    assert.equal(await page.locator('#answer-mode').inputValue(), 'extractive');
    assert.equal(await page.locator('#generated-option').isDisabled(), true);
    assert.match(await page.locator('#model-status').textContent(), /未配置/);
    checks.push('Actual authenticated model status is unconfigured; extractive default and generated option disabled');

    const accepted = await question(page, '路径参数');
    assert.equal(accepted.request.answer_mode, 'extractive');
    assert.equal(accepted.result.refused, false);
    assert.equal(accepted.result.answer_mode, 'extractive');
    assert.ok(accepted.result.sources.length > 0 && accepted.result.sources.length <= 3);
    assert.equal(await page.locator('#sources .source-card').count(), accepted.result.sources.length);
    assert.equal(await page.locator('#candidates .candidate-card').count(), 0);
    assert.deepEqual(await page.locator('#sources .citation-id').allTextContents(), accepted.result.sources.map(source => source.citation_id));
    assert.match(await page.locator('#answer-provenance').textContent(), /未调用生成模型/);
    assert.equal(await page.locator('#evidence-decision').textContent(), `证据判定 · ${accepted.result.evidence_decision.message}`);
    await page.screenshot({ path: path.join(output, 'desktop-accepted.png'), fullPage: true });
    await page.locator('#sources .source-card').first().click();
    await page.locator('.chunk-card').first().waitFor();
    assert.match(await page.locator('#source-location').textContent(), /行/);
    assert.equal(await page.locator('#source-kind').isVisible(), false);
    await page.screenshot({ path: path.join(output, 'desktop-source.png'), fullPage: true });
    await page.locator('#close-source').click();
    checks.push('Accepted evidence uses actual <=3 quoted sources with API-matching citation IDs and source-line drawer');

    const refused = await question(page, '请从 FastAPI 文档中计算明天海王星首都的失业率及独角兽税率');
    assert.equal(refused.result.refused, true, 'Out-of-scope facts must not be answered');
    assert.equal(refused.result.sources.length, 0);
    assert.ok(refused.result.retrieval_candidates.length > 0, 'The query must retain related but insufficient candidates');
    assert.equal(await page.locator('#sources .source-card').count(), 0);
    assert.equal(await page.locator('#candidates .candidate-card').count(), refused.result.retrieval_candidates.length);
    assert.equal(await page.locator('#candidates .citation-id').count(), 0);
    assert.match(await page.locator('#candidate-title').textContent(), /未验证候选/);
    await page.screenshot({ path: path.join(output, 'desktop-refusal.png'), fullPage: true });
    await page.locator('#candidates .candidate-card').first().click();
    assert.match(await page.locator('#source-kind').textContent(), /未验证检索候选/);
    await page.locator('#close-source').click();
    checks.push('Real refusal has no supporting sources; retained candidates have no S citation and stay marked unverified in drawer');

    const searched = await question(page, '路径参数', '#search-button');
    assert.equal(Object.hasOwn(searched.request, 'answer_mode'), false);
    assert.ok(searched.result.items.length > 0);
    assert.equal(await page.locator('#sources .source-card').count(), 0);
    assert.equal(await page.locator('#candidates .candidate-card').count(), searched.result.items.length);
    assert.equal(await page.locator('#candidates .citation-id').count(), 0);
    assert.match(await page.locator('#candidate-title').textContent(), /检索候选/);
    checks.push('Search-only mode returns real candidates without invented answer sources or citation labels');

    await page.locator('[data-view="library"]').click();
    const beforeUpload = Number(await page.locator('#stat-docs').textContent());
    const filename = `ui-v2-${Date.now()}.txt`;
    await page.locator('#upload-file').setInputFiles({
      name: filename, mimeType: 'text/plain',
      buffer: Buffer.from(`# 浏览器回归样本 ${filename}\n紫色航船编号为 R42。\n<script>window.KNOWTRACE_XSS=1</script>`),
    });
    await page.locator('#upload-button').click();
    await page.waitForFunction(() => document.querySelector('#upload-note').textContent.startsWith('已收录'));
    await page.waitForFunction(count => Number(document.querySelector('#stat-docs').textContent) === count, beforeUpload + 1);
    const uploaded = page.locator('.document-row').filter({ hasText: filename });
    await uploaded.getByRole('button', { name: '查看片段', exact: true }).click();
    await page.locator('.chunk-card').first().waitFor();
    assert.match(await page.locator('#source-chunks').textContent(), /<script>/);
    assert.equal(await page.evaluate(() => window.KNOWTRACE_XSS), undefined);
    await page.locator('#close-source').click();
    page.once('dialog', dialog => dialog.accept());
    await uploaded.getByRole('button', { name: '删除', exact: true }).click();
    await page.waitForFunction(count => Number(document.querySelector('#stat-docs').textContent) === count, beforeUpload);
    checks.push('Actual upload, literal unsafe-looking document content, deletion and restored counts');

    await page.locator('[data-view="evaluation"]').click();
    await page.locator('#evaluation-split').selectOption('test');
    const [evaluationResponse] = await Promise.all([
      page.waitForResponse(response => response.url() === base + '/api/evaluations/run' && response.request().method() === 'POST'),
      page.locator('#run-evaluation').click(),
    ]);
    assert.ok(evaluationResponse.ok());
    const evaluation = await evaluationResponse.json();
    await page.waitForFunction(() => document.querySelector('#evaluation-note').textContent.startsWith('运行完成'));
    assert.equal(evaluation.split, 'test');
    assert.equal(await page.locator('#case-list tr').count(), evaluation.cases.length);
    const [download] = await Promise.all([
      page.waitForEvent('download'), page.locator('#download-evaluation').click(),
    ]);
    await download.saveAs(path.join(output, 'evaluation.json'));
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(output, 'evaluation.json'), 'utf8')), evaluation);
    await page.screenshot({ path: path.join(output, 'desktop-evaluation.png'), fullPage: true });
    checks.push('Actual offline test-split evaluation, per-case UI rows and matching downloaded report');

    await checkDelayedAccountSwitch(page, checks);
    await context.close();
    context = null;
    await checkMobile(browser, checks, errors);
    assert.deepEqual(errors, []);
    const result = {
      status: 'passed', version: '0.2', checks, errors,
      browser: 'isolated headless Microsoft Edge', runtime_network: 'local test origin only',
      generated_provider: 'not exercised; provider request behavior is covered by mock API tests',
    };
    fs.writeFileSync(path.join(output, 'checks.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ ...result, output }, null, 2));
  } catch (error) {
    fs.writeFileSync(path.join(output, 'checks.json'), JSON.stringify({ status: 'failed', checks, errors, error: error.message }, null, 2));
    throw error;
  } finally {
    if (context) await context.close();
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
