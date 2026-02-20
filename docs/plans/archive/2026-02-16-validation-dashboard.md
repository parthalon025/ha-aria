# Validation Dashboard Page — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Validation" page to the ARIA dashboard that runs the synthetic validation suite on demand, displays accuracy scores per scenario/metric, and shows metamorphic + backtest results.

**Architecture:** New `/api/validation/run` POST endpoint triggers a subprocess that runs the validation pytest suite, captures structured JSON output, and returns it. New `/api/validation/latest` GET endpoint returns cached last-run results. A new `Validation.jsx` page renders the results with the existing design language (PageBanner, HeroCard, CollapsibleSection, accuracy color coding).

**Tech Stack:** FastAPI (backend), Preact (frontend), pytest subprocess with JSON output, existing CSS design tokens

---

### Task 1: Backend — Validation Runner Module

**Files:**
- Create: `aria/hub/validation_runner.py`

Create a module that executes the validation test suite and parses results into structured JSON.

```python
"""Run ARIA validation suite and parse results into structured JSON."""

import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Cache last result in memory (survives across requests, reset on restart)
_last_result = None


def run_validation() -> dict:
    """Execute validation suite via pytest subprocess, return structured results."""
    global _last_result

    project_root = Path(__file__).resolve().parent.parent.parent
    venv_pytest = project_root / ".venv" / "bin" / "python"

    cmd = [
        str(venv_pytest), "-m", "pytest",
        "tests/integration/test_validation_scenarios.py",
        "tests/integration/test_validation_backtest.py",
        "-v", "--timeout=120", "--tb=short",
        "-q",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root),
        )

        result = _parse_pytest_output(proc.stdout, proc.stderr, proc.returncode)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = _extract_duration(proc.stdout)
        _last_result = result
        return result

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Validation suite exceeded 300s timeout",
                "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()}


def get_latest() -> dict | None:
    """Return the most recent validation result, or None."""
    return _last_result


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> dict:
    """Parse pytest verbose output into structured result."""
    lines = stdout.splitlines()
    tests = []
    passed = 0
    failed = 0
    skipped = 0

    # Parse PASSED/FAILED lines from -v output
    for line in lines:
        if " PASSED" in line:
            name = line.split(" PASSED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "passed"})
            passed += 1
        elif " FAILED" in line:
            name = line.split(" FAILED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "failed"})
            failed += 1
        elif " SKIPPED" in line:
            name = line.split(" SKIPPED")[0].strip().split("::")[-1]
            tests.append({"name": name, "status": "skipped"})
            skipped += 1

    # Extract the accuracy report from captured stdout
    report = _extract_accuracy_report(stdout)

    return {
        "status": "passed" if returncode == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "tests": tests,
        "report": report,
        "raw_output": stdout[-2000:] if len(stdout) > 2000 else stdout,
    }


def _extract_accuracy_report(stdout: str) -> dict:
    """Extract scenario accuracy scores from the validation report printed to stdout."""
    scenarios = {}
    overall = None
    lines = stdout.splitlines()
    in_report = False

    for line in lines:
        if "ARIA VALIDATION REPORT" in line:
            in_report = True
            continue
        if in_report and line.strip().startswith("OVERALL"):
            parts = line.split()
            # OVERALL    84%
            for p in parts:
                if p.endswith("%"):
                    overall = float(p.rstrip("%"))
                    break
            in_report = False
            continue
        if in_report and "%" in line and not line.startswith("-") and not line.startswith("="):
            parts = line.split()
            if len(parts) >= 2:
                scenario_name = parts[0]
                # Find percentage values
                pcts = [float(p.rstrip("%")) for p in parts[1:] if p.endswith("%")]
                if pcts:
                    scenarios[scenario_name] = {
                        "overall": pcts[0],
                        "metrics": pcts[1:] if len(pcts) > 1 else [],
                    }

    # Extract backtest report similarly
    backtest = {}
    in_backtest = False
    for line in lines:
        if "REAL-DATA BACKTEST" in line or "Real vs Synthetic" in line:
            in_backtest = True
            continue
        if in_backtest and ("Real data" in line.lower() or "real:" in line.lower()):
            pcts = []
            for p in line.split():
                if p.endswith("%"):
                    pcts.append(float(p.rstrip("%")))
            if pcts:
                backtest["overall"] = pcts[0]
                backtest["metrics"] = pcts[1:] if len(pcts) > 1 else []
            in_backtest = False

    return {
        "scenarios": scenarios,
        "overall": overall,
        "backtest": backtest,
    }


def _extract_duration(stdout: str) -> float | None:
    """Extract test duration from pytest summary line like '95 passed in 61.33s'."""
    for line in reversed(stdout.splitlines()):
        if "passed" in line and " in " in line:
            parts = line.split(" in ")
            if parts:
                time_str = parts[-1].strip().rstrip("s").strip()
                # Handle "0:01:01" format too
                if ":" in time_str:
                    segments = time_str.split(":")
                    return sum(float(s) * (60 ** i) for i, s in enumerate(reversed(segments)))
                try:
                    return float(time_str)
                except ValueError:
                    pass
    return None
```

### Task 2: Backend — API Endpoints

**Files:**
- Modify: `aria/hub/api.py`

Add two endpoints after the existing route definitions (find the `# --- Activity ---` section and add before it, or at the end of the route setup):

```python
# --- Validation ---

@router.post("/api/validation/run")
async def run_validation_suite():
    """Run the full validation test suite. Returns structured results."""
    from aria.hub.validation_runner import run_validation
    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_validation)
    return result

@router.get("/api/validation/latest")
async def get_validation_latest():
    """Return the most recent validation run result."""
    from aria.hub.validation_runner import get_latest
    result = get_latest()
    if result is None:
        return {"status": "no_runs", "message": "No validation runs yet. POST /api/validation/run to start one."}
    return result
```

### Task 3: Frontend — Validation Page Component

**Files:**
- Create: `aria/dashboard/spa/src/pages/Validation.jsx`

Create the Validation page following the existing pattern (PageBanner, HeroCard, CollapsibleSection, fetchJson, LoadingState, ErrorState). The page should:

1. On mount, fetch `/api/validation/latest` to show last results (or "No runs yet")
2. Have a "Run Validation" button that POSTs to `/api/validation/run`
3. Show a loading spinner during the run (~60s)
4. Display results in sections:
   - **Hero card**: Overall accuracy % with color coding (green/yellow/red)
   - **Scenario table**: Each scenario with overall + per-metric scores
   - **Metamorphic tests**: Pass/fail status for invariant assertions
   - **Backtest results**: Real-data accuracy vs synthetic comparison
   - **Test summary**: X passed, Y failed, Z skipped with duration

Use the existing `accuracyColor()` from `constants.js` for color coding. Use `t-frame`, `data-mono`, and other existing CSS classes. Follow the MLEngine.jsx and Shadow.jsx patterns for structure.

Page banner text: `VALIDATION`

```jsx
/**
 * Validation page — run and display ARIA validation suite results.
 * Fetches from /api/validation/latest, triggers /api/validation/run.
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson, postJson } from '../api.js';
import { accuracyColor } from '../constants.js';
import PageBanner from '../components/PageBanner.jsx';
import HeroCard from '../components/HeroCard.jsx';
import CollapsibleSection from '../components/CollapsibleSection.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

const METRIC_NAMES = {
  0: 'Power',
  1: 'Lights',
  2: 'Occupancy',
  3: 'Unavail',
  4: 'Events',
};

function StatusDot({ status }) {
  const color = status === 'passed'
    ? 'var(--status-healthy)'
    : status === 'failed'
    ? 'var(--status-error)'
    : 'var(--text-tertiary)';
  return <span style={`display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${color}; margin-right: 6px;`} />;
}

function ScenarioTable({ scenarios }) {
  if (!scenarios || !Object.keys(scenarios).length) return null;
  return (
    <div class="t-frame" data-label="scenarios" style="padding: 0.75rem; overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; font-size: var(--type-body);">
        <thead>
          <tr style="border-bottom: 1px solid var(--border-subtle);">
            <th style="text-align: left; padding: 4px 8px; color: var(--text-secondary);">Scenario</th>
            <th style="text-align: right; padding: 4px 8px; color: var(--text-secondary);">Overall</th>
            {Object.values(METRIC_NAMES).map(n => (
              <th key={n} style="text-align: right; padding: 4px 8px; color: var(--text-secondary);">{n}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(scenarios).map(([name, data]) => (
            <tr key={name} style="border-bottom: 1px solid var(--border-subtle);">
              <td class="data-mono" style="padding: 4px 8px;">{name}</td>
              <td class="data-mono" style={`padding: 4px 8px; text-align: right; ${accuracyColor(data.overall)}`}>
                {data.overall}%
              </td>
              {(data.metrics || []).map((m, i) => (
                <td key={i} class="data-mono" style={`padding: 4px 8px; text-align: right; ${accuracyColor(m)}`}>
                  {m}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TestList({ tests, filterStatus }) {
  const filtered = filterStatus ? tests.filter(t => t.status === filterStatus) : tests;
  if (!filtered.length) return <p class="text-sm" style="color: var(--text-tertiary)">None</p>;
  return (
    <div class="space-y-1">
      {filtered.map(t => (
        <div key={t.name} class="flex items-center text-sm" style="color: var(--text-secondary);">
          <StatusDot status={t.status} />
          <span class="data-mono">{t.name}</span>
        </div>
      ))}
    </div>
  );
}

export default function Validation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  function loadLatest() {
    setLoading(true);
    fetchJson('/api/validation/latest')
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }

  function runValidation() {
    setRunning(true);
    setError(null);
    postJson('/api/validation/run', {})
      .then((result) => {
        setData(result);
        setRunning(false);
      })
      .catch((err) => {
        setError(err);
        setRunning(false);
      });
  }

  useEffect(() => { loadLatest(); }, []);

  if (loading) return <LoadingState label="Loading validation results..." />;
  if (error) return <ErrorState error={error} onRetry={loadLatest} />;

  const report = data?.report || {};
  const hasRun = data?.status !== 'no_runs';
  const overall = report.overall;

  return (
    <div class="space-y-6">
      <PageBanner title="VALIDATION" />

      {/* Run button */}
      <div class="flex items-center gap-4">
        <button
          onClick={runValidation}
          disabled={running}
          class="t-btn-primary"
          style="padding: 8px 20px; font-size: var(--type-body); cursor: pointer; border: none; border-radius: var(--radius);"
        >
          {running ? 'Running...' : 'Run Validation Suite'}
        </button>
        {running && (
          <span class="text-sm" style="color: var(--text-tertiary)">
            This takes ~60 seconds...
          </span>
        )}
        {hasRun && data.timestamp && (
          <span class="text-sm" style="color: var(--text-tertiary)">
            Last run: {new Date(data.timestamp).toLocaleString()}
            {data.duration_seconds && ` (${data.duration_seconds.toFixed(0)}s)`}
          </span>
        )}
      </div>

      {!hasRun ? (
        <div class="t-frame" data-label="no data" style="padding: 2rem; text-align: center;">
          <p style="color: var(--text-secondary);">No validation runs yet. Click "Run Validation Suite" to start.</p>
        </div>
      ) : (
        <>
          {/* Hero card with overall accuracy */}
          <HeroCard
            label="Prediction Accuracy"
            value={overall != null ? `${overall.toFixed(0)}%` : '—'}
            sub={`${data.passed} passed · ${data.failed} failed · ${data.skipped} skipped`}
            accentColor={overall >= 70 ? 'var(--status-healthy)' : overall >= 40 ? 'var(--status-warning)' : 'var(--status-error)'}
          />

          {/* Scenario scores */}
          <CollapsibleSection title="Scenario Scores" defaultOpen>
            <ScenarioTable scenarios={report.scenarios} />
          </CollapsibleSection>

          {/* Backtest results */}
          {report.backtest && report.backtest.overall != null && (
            <CollapsibleSection title="Real-Data Backtest" defaultOpen>
              <div class="t-frame" data-label="backtest" style="padding: 0.75rem;">
                <div class="flex items-center gap-4">
                  <span class="text-sm" style="color: var(--text-secondary);">Real data accuracy:</span>
                  <span class="data-mono text-lg font-bold" style={accuracyColor(report.backtest.overall)}>
                    {report.backtest.overall}%
                  </span>
                  {overall != null && (
                    <span class="text-sm" style="color: var(--text-tertiary);">
                      vs synthetic {overall}%
                    </span>
                  )}
                </div>
              </div>
            </CollapsibleSection>
          )}

          {/* Test details */}
          <CollapsibleSection title={`All Tests (${data.total})`}>
            <TestList tests={data.tests || []} />
          </CollapsibleSection>

          {/* Failed tests if any */}
          {data.failed > 0 && (
            <CollapsibleSection title={`Failed Tests (${data.failed})`} defaultOpen>
              <TestList tests={data.tests || []} filterStatus="failed" />
            </CollapsibleSection>
          )}
        </>
      )}
    </div>
  );
}
```

### Task 4: Frontend — Wire Into Router and Sidebar

**Files:**
- Modify: `aria/dashboard/spa/src/app.jsx`
- Modify: `aria/dashboard/spa/src/components/Sidebar.jsx`

**app.jsx changes:**
1. Add import: `import Validation from './pages/Validation.jsx';`
2. Add route: `<Validation path="/validation" />` (after the Guide route)

**Sidebar.jsx changes:**
1. Add a `CheckIcon` SVG function (checkmark/clipboard icon):
```jsx
function CheckIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M9 11l3 3L22 4" /><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
    </svg>
  );
}
```
2. Add to `NAV_ITEMS` array in the "Actions" section (after Settings, before the closing `];`):
```js
{ path: '/validation', label: 'Validation', icon: CheckIcon },
```

### Task 5: Build, Test, and Commit

**Files:**
- Rebuild: `cd aria/dashboard/spa && npm run build`

1. Build the SPA bundle
2. Verify the new endpoint responds: `curl -s http://127.0.0.1:8001/api/validation/latest | python3 -m json.tool` (should return `{"status": "no_runs", ...}` after restart, or may 404 until restart)
3. Run existing tests to confirm no regressions: `.venv/bin/python -m pytest tests/integration/test_validation_scenarios.py tests/integration/test_validation_backtest.py -v --timeout=120`
4. Commit all changes:
```bash
git add aria/hub/validation_runner.py aria/hub/api.py aria/dashboard/spa/src/pages/Validation.jsx aria/dashboard/spa/src/app.jsx aria/dashboard/spa/src/components/Sidebar.jsx aria/dashboard/spa/dist/
git commit -m "feat: add validation dashboard page with run-on-demand suite"
```
5. Push to GitHub
