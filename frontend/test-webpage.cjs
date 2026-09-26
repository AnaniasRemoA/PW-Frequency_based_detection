const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runTest() {
  console.log('--- Starting Playwright Webpage Test ---');
  const screenshotDir = path.join(__dirname, 'test-results');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome' // Use installed Chrome
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 }
  });

  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  // 1. Test Home Page
  console.log('1. Testing Home Page (http://localhost:5173/)...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  
  const title = await page.title();
  console.log('Page Title:', title);

  const heroHeading = await page.locator('h1').innerText();
  console.log('Hero Heading:', heroHeading.replace('\n', ' '));

  const homeScreenshot = path.join(screenshotDir, '01_home_page.png');
  await page.screenshot({ path: homeScreenshot, fullPage: true });
  console.log('Saved Home screenshot to:', homeScreenshot);

  // 2. Test Navigation to Analyze Page
  console.log('\n2. Testing Navigation to Analyze Page...');
  await page.click('text=Start Detection');
  await page.waitForURL('**/analyze');
  console.log('Navigated to URL:', page.url());

  const analyzeHeading = await page.locator('.section-title').first().innerText();
  console.log('Section Title:', analyzeHeading);

  const analyzeScreenshot = path.join(screenshotDir, '02_analyze_page.png');
  await page.screenshot({ path: analyzeScreenshot, fullPage: true });
  console.log('Saved Analyze screenshot to:', analyzeScreenshot);

  // 3. Test Form Inputs & Drop Zone
  console.log('\n3. Verifying Form Elements on Analyze Page...');
  const uploadAreaVisible = await page.locator('.upload-area').isVisible();
  console.log('Upload Area visible:', uploadAreaVisible);

  const strideVal = await page.locator('input[type="number"]').nth(0).inputValue().catch(() => 'N/A');
  console.log('Stride input default value:', strideVal);

  // 4. Test Mobile Responsive Viewport
  console.log('\n4. Testing Mobile Viewport (375x667)...');
  await page.setViewportSize({ width: 375, height: 667 });
  const mobileScreenshot = path.join(screenshotDir, '03_analyze_mobile.png');
  await page.screenshot({ path: mobileScreenshot, fullPage: true });
  console.log('Saved Mobile screenshot to:', mobileScreenshot);

  // 5. Console Error Audit
  console.log('\n5. Console Errors Audit:');
  if (consoleErrors.length === 0) {
    console.log('✓ Zero console errors detected!');
  } else {
    console.log('! Console errors found:', consoleErrors);
  }

  await browser.close();
  console.log('\n--- Playwright Automated Test Finished Successfully ---');
}

runTest().catch(err => {
  console.error('Playwright Test Failed:', err);
  process.exit(1);
});
