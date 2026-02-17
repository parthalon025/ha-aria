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
        if (result.status === 'already_running') {
          setError({ message: 'A validation run is already in progress. Try again shortly.' });
        } else {
          setData(result);
        }
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
      <PageBanner page="VALIDATION" />

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
