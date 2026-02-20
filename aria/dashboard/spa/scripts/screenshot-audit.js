/**
 * ARIA Dashboard Screenshot Audit
 *
 * Captures full-page screenshots of every dashboard route, hits the
 * corresponding backend API endpoints, and generates a mismatch report
 * comparing what the frontend renders vs what the API returns.
 *
 * Usage:
 *   node aria/dashboard/spa/scripts/screenshot-audit.js [--base-url URL]
 *
 * Requirements:
 *   - ARIA hub running (aria serve / systemctl --user start aria-hub)
 *   - Puppeteer installed (npm install in spa/)
 */

import puppeteer from 'puppeteer';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../../../../');
const SCREENSHOT_DIR = resolve(PROJECT_ROOT, 'docs/audit/screenshots');
const REPORT_PATH = resolve(PROJECT_ROOT, 'docs/audit/audit-report.md');

const DEFAULT_BASE = 'http://127.0.0.1:8001';
const LOAD_TIMEOUT = 15_000; // ms to wait for page load
const LOADING_POLL_INTERVAL = 250; // ms between loading-state checks
const LOADING_MAX_WAIT = 10_000; // ms max to wait for loading to clear

// ---------------------------------------------------------------------------
// Route + API mapping
// ---------------------------------------------------------------------------

const ROUTES = [
  {
    path: '/',
    name: 'home',
    apis: [
      '/health',
      '/api/ml/anomalies',
      '/api/shadow/accuracy',
      '/api/pipeline',
      '/api/cache/intelligence',
      '/api/cache/activity_summary',
      '/api/cache/automation_suggestions',
      '/api/cache/entities',
    ],
  },
  {
    path: '/observe',
    name: 'observe',
    apis: [
      '/api/cache/intelligence',
      '/api/cache/activity_summary',
      '/api/cache/presence',
    ],
  },
  {
    path: '/understand',
    name: 'understand',
    apis: [
      '/api/ml/anomalies',
      '/api/shadow/accuracy',
      '/api/ml/drift',
      '/api/ml/shap',
      '/api/patterns',
    ],
  },
  {
    path: '/decide',
    name: 'decide',
    apis: [
      '/api/cache/automation_suggestions',
      '/api/automations/feedback',
    ],
  },
  {
    path: '/discovery',
    name: 'discovery',
    apis: ['/api/discovery/status', '/api/settings/discovery'],
  },
  {
    path: '/capabilities',
    name: 'capabilities',
    apis: ['/api/capabilities/registry', '/api/capabilities/candidates'],
  },
  {
    path: '/ml-engine',
    name: 'ml-engine',
    apis: [
      '/api/ml/models',
      '/api/ml/drift',
      '/api/ml/features',
      '/api/ml/hardware',
      '/api/ml/online',
    ],
  },
  {
    path: '/data-curation',
    name: 'data-curation',
    apis: ['/api/curation', '/api/curation/summary'],
  },
  {
    path: '/validation',
    name: 'validation',
    apis: ['/api/validation/latest'],
  },
  {
    path: '/settings',
    name: 'settings',
    apis: ['/api/config'],
  },
  {
    path: '/guide',
    name: 'guide',
    apis: [],
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  let baseUrl = DEFAULT_BASE;
  const idx = args.indexOf('--base-url');
  if (idx !== -1 && args[idx + 1]) {
    baseUrl = args[idx + 1].replace(/\/$/, '');
  }
  return { baseUrl };
}

async function fetchApi(baseUrl, endpoint) {
  const url = `${baseUrl}${endpoint}`;
  const start = Date.now();
  try {
    const res = await fetch(url);
    const elapsed = Date.now() - start;
    const body = await res.text();
    let json = null;
    try {
      json = JSON.parse(body);
    } catch {
      // non-JSON response
    }
    return { url, status: res.status, elapsed, json, error: null };
  } catch (err) {
    return { url, status: null, elapsed: Date.now() - start, json: null, error: err.message };
  }
}

async function waitForLoadingClear(page) {
  const deadline = Date.now() + LOADING_MAX_WAIT;
  while (Date.now() < deadline) {
    const hasLoading = await page.evaluate(() => {
      // Check for common loading indicators
      const loadingEls = document.querySelectorAll(
        '[class*="loading"], [class*="Loading"], [class*="spinner"], [class*="Spinner"]'
      );
      for (const el of loadingEls) {
        if (el.offsetParent !== null) return true; // visible
      }
      return false;
    });
    if (!hasLoading) return;
    await new Promise((r) => setTimeout(r, LOADING_POLL_INTERVAL));
  }
}

function summarizeJson(json) {
  if (json === null || json === undefined) return 'null';
  if (Array.isArray(json)) return `array[${json.length}]`;
  if (typeof json === 'object') {
    const keys = Object.keys(json);
    if (keys.length <= 6) return `{${keys.join(', ')}}`;
    return `{${keys.slice(0, 5).join(', ')}, ... +${keys.length - 5}}`;
  }
  return String(json).slice(0, 80);
}

// ---------------------------------------------------------------------------
// Page audit
// ---------------------------------------------------------------------------

async function auditPage(browser, baseUrl, route) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const dashUrl = `${baseUrl}/ui/#${route.path}`;
  const result = {
    route: route.name,
    path: route.path,
    dashUrl,
    screenshot: null,
    pageError: null,
    consoleErrors: [],
    apiResults: [],
  };

  // Capture console errors
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      result.consoleErrors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => {
    result.pageError = err.message;
  });

  try {
    await page.goto(dashUrl, { waitUntil: 'networkidle2', timeout: LOAD_TIMEOUT });
    await waitForLoadingClear(page);

    // Small extra settle time for renders
    await new Promise((r) => setTimeout(r, 500));

    const screenshotPath = resolve(SCREENSHOT_DIR, `${route.name}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    result.screenshot = screenshotPath;
  } catch (err) {
    result.pageError = err.message;
  }

  // Hit API endpoints
  for (const endpoint of route.apis) {
    const apiResult = await fetchApi(baseUrl, endpoint);
    result.apiResults.push(apiResult);
  }

  await page.close();
  return result;
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

function generateReport(results) {
  const now = new Date().toISOString();
  const lines = [
    '# ARIA Dashboard Screenshot Audit Report',
    '',
    `Generated: ${now}`,
    '',
    '## Summary',
    '',
    `| Route | Screenshot | Page Errors | Console Errors | API Endpoints | API Failures |`,
    `|-------|-----------|-------------|----------------|---------------|-------------|`,
  ];

  let totalApis = 0;
  let totalApiFails = 0;
  let totalPageErrors = 0;

  for (const r of results) {
    const apiFails = r.apiResults.filter(
      (a) => a.error || (a.status && a.status >= 400)
    ).length;
    totalApis += r.apiResults.length;
    totalApiFails += apiFails;
    if (r.pageError) totalPageErrors++;

    lines.push(
      `| ${r.route} | ${r.screenshot ? 'captured' : 'FAILED'} | ${r.pageError ? 'YES' : '-'} | ${r.consoleErrors.length || '-'} | ${r.apiResults.length} | ${apiFails || '-'} |`
    );
  }

  lines.push('');
  lines.push(
    `**Totals:** ${results.length} routes, ${totalApis} API calls, ${totalApiFails} failures, ${totalPageErrors} page errors`
  );

  // Detail sections per route
  for (const r of results) {
    lines.push('', `---`, '', `## ${r.route} (\`${r.path}\`)`);

    if (r.screenshot) {
      lines.push('', `Screenshot: \`screenshots/${r.route}.png\``);
    } else {
      lines.push('', 'Screenshot: **NOT CAPTURED**');
    }

    if (r.pageError) {
      lines.push('', `**Page error:** ${r.pageError}`);
    }

    if (r.consoleErrors.length) {
      lines.push('', '**Console errors:**');
      for (const e of r.consoleErrors) {
        lines.push(`- ${e}`);
      }
    }

    if (r.apiResults.length) {
      lines.push('', '### API Endpoints', '');
      lines.push('| Endpoint | Status | Time (ms) | Response | Error |');
      lines.push('|----------|--------|-----------|----------|-------|');
      for (const a of r.apiResults) {
        const status = a.error ? 'ERR' : String(a.status);
        const summary = a.error ? '-' : summarizeJson(a.json);
        const errCol = a.error || '-';
        lines.push(`| \`${a.url.replace(/^https?:\/\/[^/]+/, '')}\` | ${status} | ${a.elapsed} | ${summary} | ${errCol} |`);
      }
    } else {
      lines.push('', '_No API endpoints (static content)._');
    }
  }

  lines.push('');
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const { baseUrl } = parseArgs();

  console.log(`ARIA Dashboard Screenshot Audit`);
  console.log(`Base URL: ${baseUrl}`);
  console.log(`Screenshots: ${SCREENSHOT_DIR}`);
  console.log(`Report: ${REPORT_PATH}`);
  console.log('');

  await mkdir(SCREENSHOT_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const results = [];

  for (const route of ROUTES) {
    process.stdout.write(`Auditing ${route.name} ...`);
    try {
      const result = await auditPage(browser, baseUrl, route);
      results.push(result);
      const apiOk = result.apiResults.filter((a) => !a.error && a.status < 400).length;
      const apiTotal = result.apiResults.length;
      console.log(
        ` ${result.screenshot ? 'screenshot OK' : 'screenshot FAILED'}` +
          (apiTotal ? `, APIs: ${apiOk}/${apiTotal}` : '') +
          (result.pageError ? ` [PAGE ERROR]` : '')
      );
    } catch (err) {
      console.log(` FAILED: ${err.message}`);
      results.push({
        route: route.name,
        path: route.path,
        dashUrl: `${baseUrl}/ui/#${route.path}`,
        screenshot: null,
        pageError: err.message,
        consoleErrors: [],
        apiResults: [],
      });
    }
  }

  await browser.close();

  const report = generateReport(results);
  await writeFile(REPORT_PATH, report, 'utf-8');

  console.log('');
  console.log(`Report written to ${REPORT_PATH}`);

  // Exit with error code if any failures
  const hasFailures = results.some(
    (r) =>
      r.pageError ||
      r.apiResults.some((a) => a.error || (a.status && a.status >= 400))
  );
  if (hasFailures) {
    console.log('WARNING: Some routes or API endpoints had errors.');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(2);
});
