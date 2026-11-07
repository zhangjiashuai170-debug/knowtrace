/* Record a real offline workflow using only seeded demo data in isolated Edge. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const server = new URL(process.env.KNOWTRACE_TEST_URL || 'http://127.0.0.1:8767');
assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(server.hostname), 'Recording requires a loopback test server');
assert.notEqual(server.port, '8766', 'Use the isolated demo data server, not the existing 8766 workspace');
assert.equal(server.pathname, '/', 'Use the test server origin without a path');
const base = server.origin;
const output = path.resolve(__dirname, '../artifacts/demo');
fs.mkdirSync(output, { recursive: true });
const stamp = new Date().toISOString().replace(/[:.]/g, '-');

async function pause(page, milliseconds) {
  // Intentional viewing time in the demo, not a readiness assertion.
  await page.waitForTimeout(milliseconds);
}

async function main() {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  let context;
  let video;
  let completed = false;
  const errors = [];
  const started = Date.now();
  try {
    context = await browser.newContext({
      viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1,
      recordVideo: { dir: output, size: { width: 1440, height: 1000 } },
    });
    context.setDefaultTimeout(25000);
    await context.route('**/*', route => new URL(route.request().url()).origin === base ? route.continue() : route.abort());
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    video = page.video();
    await page.goto(base);
    await page.locator('#login-dialog').waitFor({ state: 'visible' });
    await page.locator('#demo-account').selectOption('engineer');
    await pause(page, 1500);
    await page.locator('#login-button').click();
    await page.locator('#login-dialog').waitFor({ state: 'hidden' });
    await page.waitForFunction(() => !document.querySelector('#refresh-model').disabled);
    const modelResponse = await page.request.get(base + '/api/model');
    assert.ok(modelResponse.ok());
    assert.equal((await modelResponse.json()).configured, false, 'Record the actual unconfigured offline mode');
    assert.equal(await page.locator('#answer-mode').inputValue(), 'extractive');
    await pause(page, 1800);

    await page.locator('#question').fill('FastAPI 中如何定义路径参数？');
    await pause(page, 1300);
    const [answerResponse] = await Promise.all([
      page.waitForResponse(response => response.url() === base + '/api/ask' && response.request().method() === 'POST'),
      page.locator('#ask-button').click(),
    ]);
    assert.ok(answerResponse.ok());
    const answer = await answerResponse.json();
    assert.equal(answer.answer_mode, 'extractive');
    assert.equal(answer.refused, false);
    await page.locator('#sources .source-card').first().waitFor();
    await page.locator('.result-panel').scrollIntoViewIfNeeded();
    await pause(page, 2700);

    await page.locator('#sources .source-card').first().click();
    await page.locator('.chunk-card').first().waitFor();
    await pause(page, 3200);
    await page.locator('#close-source').click();
    await page.locator('[data-view="evaluation"]').click();
    await page.locator('#evaluation-split').selectOption('test');
    await page.locator('#evaluation-form').scrollIntoViewIfNeeded();
    await pause(page, 1600);
    await page.locator('#run-evaluation').click();
    await page.waitForFunction(() => document.querySelector('#evaluation-note').textContent.startsWith('运行完成'));
    await page.locator('.evaluation-layout').scrollIntoViewIfNeeded();
    await pause(page, 3200);
    await page.locator('#case-panel').scrollIntoViewIfNeeded();
    await pause(page, 2300);
    assert.deepEqual(errors, []);
    completed = true;
  } finally {
    if (context) await context.close();
    await browser.close();
    if (video) {
      const recorded = await video.path();
      const destination = path.join(output, `knowtrace-offline-${stamp}${completed ? '' : '-incomplete'}.webm`);
      fs.renameSync(recorded, destination);
      const metadata = {
        status: completed ? 'completed' : 'incomplete', video: path.basename(destination),
        elapsed_seconds: Number(((Date.now() - started) / 1000).toFixed(1)),
        browser: 'isolated headless Microsoft Edge', data: 'seeded demo engineering account and corpus',
        workflow: ['demo login', 'offline evidence question', 'source lines', 'actual test-split retrieval evaluation', 'case results'],
        model_calls: 'none; answer mode stays extractive and generated provider is not configured',
        network: 'local test origin only', overlay_or_mocked_success: false,
      };
      fs.writeFileSync(path.join(output, `knowtrace-offline-${stamp}.json`), JSON.stringify(metadata, null, 2));
      console.log(JSON.stringify({ ...metadata, path: destination }, null, 2));
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
