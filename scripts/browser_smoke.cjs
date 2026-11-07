/* Isolated headless UI test; never attaches to the user's browser profile. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.KNOWTRACE_TEST_URL || 'http://127.0.0.1:8767';
const output = path.resolve(__dirname, '../artifacts/browser');
fs.mkdirSync(output, { recursive: true });

async function login(page, user) {
  await page.locator('#login-dialog').waitFor({ state: 'visible' });
  await page.locator('#demo-account').selectOption(user);
  await page.locator('#login-button').click();
  await page.locator('#login-dialog').waitFor({ state: 'hidden' });
  await page.waitForFunction(() => document.querySelector('#stat-docs').textContent !== '—');
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const checks = [];
  const errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await context.route('**/*', route => route.request().url().startsWith(base + '/') ? route.continue() : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 500) errors.push(`${response.status()} ${response.url()}`); });
    await page.goto(base);
    await login(page, 'engineer');
    assert.equal(await page.locator('#stat-docs').textContent(), '20');
    await page.locator('#question').fill('路径参数');
    await page.locator('#ask-button').click();
    await page.locator('.source-card').first().waitFor();
    assert.match(await page.locator('#result-meta').textContent(), /离线摘录/);
    await page.screenshot({ path: path.join(output, 'desktop-ask.png'), fullPage: true });
    await page.locator('.source-card').first().click();
    await page.locator('.chunk-card').first().waitFor();
    assert.match(await page.locator('#source-location').textContent(), /行/);
    assert.ok((await page.locator('#source-download').getAttribute('href')).endsWith('/source'));
    await page.screenshot({ path: path.join(output, 'desktop-source.png'), fullPage: true });
    await page.locator('#close-source').click();
    checks.push('Demo login, scoped corpus, actual evidence answer and source-line drawer');

    await page.locator('[data-view="library"]').click();
    await page.locator('#upload-file').setInputFiles({ name: 'ui-smoke.txt', mimeType: 'text/plain', buffer: Buffer.from('# UI测试\n紫色航船编号为R42。\n<script>window.BAD=1</script>') });
    await page.locator('#upload-button').click();
    await page.waitForFunction(() => document.querySelector('#upload-note').textContent.startsWith('已收录'));
    await page.waitForFunction(() => document.querySelector('#stat-docs').textContent === '21');
    assert.equal(await page.evaluate(() => window.BAD), undefined);
    await page.screenshot({ path: path.join(output, 'desktop-library.png'), fullPage: true });
    const ownRow = page.locator('.document-row').filter({ hasText: 'ui-smoke.txt' });
    page.once('dialog', dialog => dialog.accept());
    await ownRow.getByRole('button', { name: '删除', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('#stat-docs').textContent === '20');
    checks.push('Actual upload/delete, refreshed counts, literal untrusted document text');

    await page.locator('[data-view="evaluation"]').click();
    await page.locator('#evaluation-split').selectOption('test');
    await page.locator('#run-evaluation').click();
    await page.waitForFunction(() => document.querySelector('#evaluation-note').textContent.startsWith('运行完成'), { timeout: 30000 });
    assert.equal(await page.locator('#case-list tr').count(), 50);
    await page.locator('#download-evaluation').waitFor();
    const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-evaluation').click()]);
    await download.saveAs(path.join(output, 'ui-evaluation.json'));
    const report = JSON.parse(fs.readFileSync(path.join(output, 'ui-evaluation.json'), 'utf8'));
    assert.equal(report.split, 'test');
    assert.equal(report.cases.length, 50);
    await page.screenshot({ path: path.join(output, 'desktop-evaluation.png'), fullPage: true });
    checks.push('Actual test-split evaluation, 50 case rows, downloadable matching JSON');

    await page.locator('[data-view="ask"]').click();
    let signalReady;
    let release;
    let signalFinished;
    const ready = new Promise(resolve => { signalReady = resolve; });
    const finished = new Promise(resolve => { signalFinished = resolve; });
    await page.route('**/api/ask', async route => {
      const response = await route.fetch();
      signalReady();
      await new Promise(resolve => { release = resolve; });
      await route.fulfill({ response });
      signalFinished();
    });
    await page.locator('#question').fill('路径参数');
    await page.locator('#ask-button').click();
    await ready;
    await page.locator('#switch-user').click();
    await login(page, 'support');
    release();
    await finished;
    await page.unroute('**/api/ask');
    assert.equal(await page.locator('#stat-docs').textContent(), '1');
    await page.waitForFunction(() => document.querySelector('#answer-content').classList.contains('hidden'));
    assert.equal(await page.locator('.source-card').count(), 0);
    const docs = await page.request.get(base + '/api/documents');
    assert.ok((await docs.json()).items.every(doc => doc.kb_id === 'kb-support'));
    checks.push('Account switch suppresses delayed prior-scope answer and sources');
    await context.close();

    const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
    await mobile.route('**/*', route => route.request().url().startsWith(base + '/') ? route.continue() : route.abort());
    const phone = await mobile.newPage();
    phone.on('pageerror', error => errors.push(error.message));
    await phone.goto(base);
    await login(phone, 'engineer');
    assert.ok(await phone.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await phone.screenshot({ path: path.join(output, 'mobile-ask.png'), fullPage: true });
    checks.push('390px mobile layout has no horizontal page overflow');
    await mobile.close();
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'checks.json'), JSON.stringify({ status: 'passed', checks, errors, browser: 'isolated headless Microsoft Edge', runtime_network: 'only local test server allowed' }, null, 2));
    console.log(JSON.stringify({ status: 'passed', checks, output }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
