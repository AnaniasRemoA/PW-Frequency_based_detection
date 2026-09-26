const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function testE2E() {
  console.log('--- Starting Playwright End-to-End Analysis Test ---');
  const screenshotDir = path.join(__dirname, 'test-results');
  
  const sampleVideo = path.resolve(__dirname, '../examples/Sample/id1_id9_0000_fake.mp4');
  console.log('Using sample test video:', sampleVideo);
  if (!fs.existsSync(sampleVideo)) {
    throw new Error('Sample video not found at: ' + sampleVideo);
  }

  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome'
  });

  const page = await browser.newPage({ viewport: { width: 1366, height: 768 } });
  
  // Track errors and console
  page.on('console', msg => console.log('[Browser Console]', msg.type(), msg.text()));
  page.on('pageerror', err => console.error('[Browser PageError]', err));

  console.log('Navigating to http://localhost:5173/analyze...');
  await page.goto('http://localhost:5173/analyze', { waitUntil: 'networkidle' });

  // 1. Set file input
  console.log('Setting video file on input[type="file"]...');
  const fileInput = await page.locator('input[type="file"]');
  await fileInput.setInputFiles(sampleVideo);

  // Take screenshot of uploaded state
  await page.waitForTimeout(500);
  const uploadedScreenshot = path.join(screenshotDir, '04_video_uploaded.png');
  await page.screenshot({ path: uploadedScreenshot });
  console.log('Saved uploaded state screenshot to:', uploadedScreenshot);

  // Adjust stride to 8 and maxFrames to 60 for fast automated testing
  console.log('Adjusting sliders for rapid test (maxFrames=60, stride=8)...');
  await page.locator('.setting-row input[type="range"]').nth(0).evaluate(el => {
    el.value = 60;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.locator('.setting-row input[type="range"]').nth(1).evaluate(el => {
    el.value = 8;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });

  // 2. Click Run Deepfake Analysis
  console.log('Clicking "Run Deepfake Analysis"...');
  await page.click('button:has-text("Run Deepfake Analysis")');

  // Verify loading state
  await page.waitForTimeout(500);
  const loadingScreenshot = path.join(screenshotDir, '05_analysis_loading.png');
  await page.screenshot({ path: loadingScreenshot });
  console.log('Saved loading state screenshot to:', loadingScreenshot);

  // 3. Wait for navigation to /results (allow up to 90s for model inference)
  console.log('Waiting for inference and results page navigation...');
  await page.waitForURL('**/results', { timeout: 90000 });
  console.log('Successfully reached Results page:', page.url());

  // Wait for results layout to render
  await page.waitForSelector('.verdict-card', { timeout: 10000 });
  
  const verdict = await page.locator('.verdict-title').innerText();
  console.log('Verdict Title:', verdict);

  const stats = await page.locator('.stat-item').allInnerTexts();
  console.log('Detection Stats:\n', stats.join(' | '));

  // Take screenshot of results page
  const resultsScreenshot = path.join(screenshotDir, '06_results_page.png');
  await page.screenshot({ path: resultsScreenshot, fullPage: true });
  console.log('Saved Results page screenshot to:', resultsScreenshot);

  await browser.close();
  console.log('\n--- End-to-End Analysis Test PASSED! ---');
}

testE2E().catch(err => {
  console.error('End-to-End Test Failed:', err);
  process.exit(1);
});
