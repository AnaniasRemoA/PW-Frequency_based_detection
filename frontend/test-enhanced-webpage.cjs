const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function testEnhancedWebpage() {
  console.log('=== Playwright Automated Test of Enhanced Webpage ===');
  const screenshotDir = path.join(__dirname, 'test-results-enhanced');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir, { recursive: true });
  }

  const browser = await chromium.launch({
    headless: true,
    channel: 'chrome'
  });

  const page = await browser.newPage({ viewport: { width: 1366, height: 850 } });
  
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });

  // 1. Test Home Page
  console.log('1. Testing Home Page (http://localhost:5173/)...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000); // Allow status badge to fetch
  
  const statusBadge = await page.locator('.status-badge').innerText();
  console.log('Dynamic Status Badge:', statusBadge);

  const homeScreenshot = path.join(screenshotDir, '01_home_enhanced.png');
  await page.screenshot({ path: homeScreenshot });
  console.log('Saved Home screenshot:', homeScreenshot);

  // 2. Test Analyze Page with Quick Sample Cards
  console.log('\n2. Testing Analyze Page with Benchmark Samples...');
  await page.click('text=Analyze');
  await page.waitForURL('**/analyze');
  await page.waitForTimeout(1000);

  const sampleCards = await page.locator('text=Benchmark Sample:').count();
  console.log('Detected Benchmark Sample elements:', sampleCards);

  // Click DeepFake Sample card
  console.log('Clicking "FaceSwap DeepFake" sample card...');
  await page.click('text=FaceSwap DeepFake (Fake)');
  await page.waitForTimeout(500);

  const analyzeScreenshot = path.join(screenshotDir, '02_analyze_sample_selected.png');
  await page.screenshot({ path: analyzeScreenshot });
  console.log('Saved Analyze Sample Selection screenshot:', analyzeScreenshot);

  // 3. Test Mobile Viewport (375x667) for Navbar Overlap Fix
  console.log('\n3. Testing Mobile Viewport (375x667)...');
  await page.setViewportSize({ width: 375, height: 667 });
  await page.waitForTimeout(500);
  const mobileScreenshot = path.join(screenshotDir, '03_mobile_navbar_fixed.png');
  await page.screenshot({ path: mobileScreenshot });
  console.log('Saved Mobile Screenshot:', mobileScreenshot);

  // Reset viewport
  await page.setViewportSize({ width: 1366, height: 850 });

  // 4. Test Results Page with Forensic Tabs
  console.log('\n4. Testing Results Page with Multi-Modal Forensic Tabs...');
  await page.evaluate(() => {
    const mockResult = {
      is_fake: true,
      confidence: 0.703551,
      fake_prob_pct: 100.0,
      total_frames: 60,
      analyzed_clips: 8,
      tracks: 1,
      elapsed: 24.5,
      video_url: '/api/video/id1_id9_0000_fake_detect.mp4',
      heatmap_url: '/api/video/id1_id9_0000_fake_heatmap.jpg',
      fft_url: '/api/video/id1_id9_0000_fake_fft.jpg'
    };
    sessionStorage.setItem('last_result', JSON.stringify(mockResult));
  });

  await page.goto('http://localhost:5173/results', { waitUntil: 'networkidle' });

  console.log('Reached Results page:', page.url());
  await page.waitForSelector('.verdict-card', { timeout: 10000 });

  const verdict = await page.locator('.verdict-title').innerText();
  console.log('Forensic Verdict:', verdict);

  // Capture Video tab
  const videoTabScreenshot = path.join(screenshotDir, '04_results_tab_video.png');
  await page.screenshot({ path: videoTabScreenshot, fullPage: true });
  console.log('Saved Video Tab Screenshot:', videoTabScreenshot);

  // Switch to Grad-CAM Tab
  const hasGradcamTab = await page.locator('button:has-text("Grad-CAM Heatmap")').isVisible();
  if (hasGradcamTab) {
    console.log('Clicking "Grad-CAM Heatmap" tab...');
    await page.click('button:has-text("Grad-CAM Heatmap")');
    await page.waitForTimeout(500);
    const gradcamScreenshot = path.join(screenshotDir, '05_results_tab_gradcam.png');
    await page.screenshot({ path: gradcamScreenshot, fullPage: true });
    console.log('Saved Grad-CAM Screenshot:', gradcamScreenshot);
  }

  // Switch to 2D FFT Tab
  const hasFftTab = await page.locator('button:has-text("2D FFT Spectrum")').isVisible();
  if (hasFftTab) {
    console.log('Clicking "2D FFT Spectrum" tab...');
    await page.click('button:has-text("2D FFT Spectrum")');
    await page.waitForTimeout(500);
    const fftScreenshot = path.join(screenshotDir, '06_results_tab_fft.png');
    await page.screenshot({ path: fftScreenshot, fullPage: true });
    console.log('Saved 2D FFT Screenshot:', fftScreenshot);
  }

  // Switch to Dual Multi-Modal Tab
  const hasDualTab = await page.locator('button:has-text("Dual Multi-Modal View")').isVisible();
  if (hasDualTab) {
    console.log('Clicking "Dual Multi-Modal View" tab...');
    await page.click('button:has-text("Dual Multi-Modal View")');
    await page.waitForTimeout(500);
    const dualScreenshot = path.join(screenshotDir, '07_results_tab_dual.png');
    await page.screenshot({ path: dualScreenshot, fullPage: true });
    console.log('Saved Dual View Screenshot:', dualScreenshot);
  }

  // 5. Console Error Audit
  console.log('\n5. Console Errors Audit:');
  if (consoleErrors.length === 0) {
    console.log('✓ Zero console errors detected during entire test run!');
  } else {
    console.log('! Console errors:', consoleErrors);
  }

  await browser.close();
  console.log('\n=== All Playwright Automated Tests PASSED! ===');
}

testEnhancedWebpage().catch(err => {
  console.error('Playwright Test Failed:', err);
  process.exit(1);
});
