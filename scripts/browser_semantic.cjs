/* Local-only semantic UI checks against isolated public demo data. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const url = new URL(process.env.KNOWTRACE_TEST_URL || 'http://127.0.0.1:8767');
assert.equal(url.hostname, '127.0.0.1');
assert.notEqual(url.port, '8766', 'Keep UI tests off the main workspace');
const output = path.resolve(__dirname, '../artifacts/browser-semantic');
fs.mkdirSync(output, { recursive: true });

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const checks = [], errors = [];
  let generatedRequests = 0;
  async function context(options = {}) {
    const ctx = await browser.newContext(options);
    await ctx.route('**/*', route => {
      const target = new URL(route.request().url());
      if (target.origin !== url.origin) return route.abort();
      if (target.pathname === '/api/ask' && route.request().method() === 'POST'
          && route.request().postDataJSON()?.answer_mode === 'generated') {
        generatedRequests++;
        return route.abort();
      }
      return route.continue();
    });
    return ctx;
  }
  async function login(page) {
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url.origin);
    await page.locator('#demo-account').selectOption('engineer');
    await page.locator('#login-button').click();
    await page.locator('#login-dialog').waitFor({ state: 'hidden' });
    await page.waitForFunction(() => !document.querySelector('#refresh-model').disabled);
  }
  try {
    const ctx = await context({ viewport: { width: 1440, height: 1000 } });
    try {
      const page = await ctx.newPage();
      await login(page);
      await page.waitForFunction(() => !document.querySelector('#ask-mode option[value="semantic"]').disabled);
      assert.match(await page.locator('#retrieval-status').textContent(), /本地|向量/);
      await page.locator('#ask-mode').selectOption('semantic');
      await page.locator('#question').fill('How to reject missing resources with a deliberate HTTP error?');
      const [response] = await Promise.all([
        page.waitForResponse(r => r.url() === url.origin + '/api/search' && r.request().method() === 'POST'),
        page.locator('#search-button').click(),
      ]);
      assert.ok(response.ok());
      assert.equal(response.request().postDataJSON().mode, 'semantic');
      const hits = (await response.json()).items;
      assert.equal(hits[0].document_name, 'handling-errors.md');
      assert.ok(hits.every(hit => hit.kb_id === 'kb-engineering'));
      await page.waitForFunction(() => !document.querySelector('#search-button').disabled);
      await page.screenshot({ path: path.join(output, 'semantic-search.png'), fullPage: true });
      checks.push('Real local E5 search from UI returns expected source first for a cross-language question');

      await page.locator('#ask-mode').selectOption('semantic_hybrid');
      await page.locator('#question').fill('zebras galaxy pineapple');
      const [answerResponse] = await Promise.all([
        page.waitForResponse(r => r.url() === url.origin + '/api/ask' && r.request().method() === 'POST'),
        page.locator('#ask-button').click(),
      ]);
      assert.ok(answerResponse.ok());
      const answer = await answerResponse.json();
      assert.equal(answer.refused, true);
      assert.deepEqual(answer.sources, []);
      assert.ok(answer.retrieval_candidates.length);
      await page.waitForFunction(() => !document.querySelector('#ask-button').disabled);
      await page.screenshot({ path: path.join(output, 'semantic-refusal.png'), fullPage: true });
      checks.push('Dense candidates remain unverified and cannot bypass local evidence admission');
      await page.locator('[data-view="evaluation"]').click();
      assert.equal(await page.locator('#evaluation-mode option[value="semantic_hybrid"]').isDisabled(), false);
      checks.push('Evaluation exposes the same available local semantic modes');
    } finally { await ctx.close(); }

    const mobile = await context({ viewport: { width: 390, height: 844 } });
    try {
      const page = await mobile.newPage();
      await login(page);
      await page.waitForFunction(() => !document.querySelector('#ask-mode option[value="semantic"]').disabled);
      await page.locator('#ask-mode').selectOption('semantic_hybrid');
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
      await page.screenshot({ path: path.join(output, 'mobile-semantic.png'), fullPage: true });
      checks.push('390px semantic selectors and status do not overflow');
    } finally { await mobile.close(); }

    const legacy = await context();
    try {
      await legacy.route('**/api/retrieval', route => route.fulfill({ status: 404, body: '{}' }));
      const page = await legacy.newPage();
      await login(page);
      await page.waitForFunction(() => document.querySelector('#retrieval-status').textContent.includes('尚未启用'));
      assert.equal(await page.locator('#ask-mode option[value="semantic"]').isDisabled(), true);
      assert.equal(await page.locator('#ask-mode').inputValue(), 'hybrid');
      checks.push('An older server without retrieval metadata keeps lexical modes usable and semantic options disabled');
    } finally { await legacy.close(); }
    assert.equal(generatedRequests, 0);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'checks.json'), JSON.stringify({ status: 'passed', checks, errors, generated_requests: generatedRequests, runtime_network: 'isolated loopback origin only' }, null, 2));
    console.log(JSON.stringify({ status: 'passed', checks, generatedRequests }, null, 2));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
