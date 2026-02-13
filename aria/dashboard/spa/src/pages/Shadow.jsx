import { useState, useEffect } from 'preact/hooks';
import { fetchJson, baseUrl } from '../api.js';
import { relativeTime } from './intelligence/utils.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

const STAGES = ['backtest', 'shadow', 'suggest', 'autonomous'];
const STAGE_LABELS = ['Backtest', 'Shadow', 'Suggest', 'Autonomous'];

const TYPE_COLORS = {
  next_domain_action: 'background: var(--accent-glow); color: var(--accent);',
  room_activation: 'background: rgba(168,85,247,0.15); color: #a855f7;',
  routine_trigger: 'background: rgba(245,158,11,0.15); color: var(--status-warning);',
};
const TYPE_LABELS = {
  next_domain_action: 'Domain',
  room_activation: 'Room',
  routine_trigger: 'Routine',
};
const OUTCOME_COLORS = {
  correct: 'background: rgba(34,197,94,0.15); color: var(--status-healthy);',
  disagreement: 'background: rgba(245,158,11,0.15); color: var(--status-warning);',
  nothing: 'background: var(--bg-surface-raised); color: var(--text-tertiary);',
};

// Gate thresholds — must stay in sync with PIPELINE_GATES in hub/api.py
const GATE_REQUIREMENTS = {
  backtest: { field: 'backtest_accuracy', threshold: 0.40, label: 'backtest accuracy', nextStage: 'shadow' },
  shadow: { field: 'shadow_accuracy_7d', threshold: 0.50, label: '7-day shadow accuracy', nextStage: 'suggest' },
  suggest: { field: 'suggest_approval_rate_14d', threshold: 0.70, label: '14-day approval rate', nextStage: 'autonomous' },
};

function PipelineStage({ pipeline, onAdvance, onRetreat, advanceError }) {
  const stage = pipeline?.current_stage || 'backtest';
  const idx = STAGES.indexOf(stage);
  const pct = Math.max(((idx + 1) / STAGES.length) * 100, 10);
  const enteredAt = pipeline?.stage_entered_at;

  const gate = GATE_REQUIREMENTS[stage];
  const gateValue = gate ? (pipeline?.[gate.field] ?? 0) : null;
  const gateMet = gate ? gateValue >= gate.threshold : false;

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Pipeline Stage</h2>
      <div class="t-card" style="padding: 1rem;">
        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-xs mb-1" style="color: var(--text-tertiary)">
              {STAGE_LABELS.map((label, i) => (
                <span key={label} style={i <= idx ? 'font-weight: 700; color: var(--accent)' : ''}>{label}</span>
              ))}
            </div>
            <div class="h-3 rounded-full" style="background: var(--bg-inset)">
              <div class="h-3 rounded-full transition-all" style={`background: var(--accent); width: ${pct}%`} />
            </div>
          </div>
          {enteredAt && (
            <p class="text-xs" style="color: var(--text-tertiary)">
              In {stage} since {new Date(enteredAt).toLocaleDateString()}
            </p>
          )}
          <div class="flex items-center gap-2 mt-3">
            <button
              onClick={onRetreat}
              disabled={idx === 0}
              class="t-btn t-btn-secondary text-xs px-3 py-1.5"
            >
              &larr; Retreat
            </button>
            <button
              onClick={onAdvance}
              disabled={idx === STAGES.length - 1}
              class="t-btn t-btn-primary text-xs px-3 py-1.5"
            >
              Advance &rarr;
            </button>
          </div>
          {gate && (
            <p class="text-xs" style={gateMet ? 'color: var(--status-healthy)' : 'color: var(--status-error)'}>
              Advance to {gate.nextStage} requires &gt;{Math.round(gate.threshold * 100)}% {gate.label}
              {' '}(current: {Math.round(gateValue * 100)}%)
            </p>
          )}
          {advanceError && (
            <p class="text-xs font-medium" style="color: var(--status-error)">{advanceError}</p>
          )}
        </div>
      </div>
    </section>
  );
}

function accuracyColor(pct) {
  if (pct >= 70) return 'color: var(--status-healthy)';
  if (pct >= 40) return 'color: var(--status-warning)';
  return 'color: var(--status-error)';
}

function AccuracySummary({ accuracy, pipeline }) {
  const overall = accuracy?.overall_accuracy ?? 0;
  const total = accuracy?.predictions_total ?? 0;
  const correct = accuracy?.predictions_correct ?? 0;
  const disagree = accuracy?.predictions_disagreement ?? 0;
  const nothing = accuracy?.predictions_nothing ?? 0;
  const stage = pipeline?.current_stage || 'backtest';

  const stats = [
    { label: 'Accuracy', value: `${Math.round(overall)}%`, colorStyle: accuracyColor(overall) },
    { label: 'Total Predictions', value: total },
    { label: 'Correct', value: correct },
    { label: 'Disagreements', value: disagree },
  ];

  return (
    <section class="space-y-3">
      <div class="flex items-center gap-3">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Accuracy</h2>
        <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{stage}</span>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((s, i) => (
          <div key={i} class="t-card" style="padding: 1rem;">
            <div class="text-2xl font-bold" style={s.colorStyle || 'color: var(--accent)'}>{s.value}</div>
            <div class="text-sm mt-1" style="color: var(--text-tertiary)">{s.label}</div>
          </div>
        ))}
      </div>
      {nothing > 0 && (
        <p class="text-xs px-1" style="color: var(--text-tertiary)">{nothing} prediction{nothing !== 1 ? 's' : ''} had no matching outcome (expired with no activity).</p>
      )}
    </section>
  );
}

function DailyTrend({ trend }) {
  if (!trend || trend.length === 0) {
    return (
      <section class="space-y-3">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Daily Trend</h2>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">Accuracy trends will appear after 24-48 hours of predictions.</span>
        </div>
      </section>
    );
  }

  const maxAcc = Math.max(...trend.map(d => d.accuracy ?? 0), 1);

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Daily Trend</h2>
      <div class="t-card" style="padding: 1rem;">
        <div class="flex items-end gap-1 h-20">
          {trend.map((d, i) => {
            const acc = d.accuracy ?? 0;
            const height = Math.max((acc / maxAcc) * 100, 4);
            const color = acc >= 70 ? '#22c55e' : acc >= 40 ? '#f59e0b' : '#ef4444';
            const label = d.date?.slice(5) || '';
            return (
              <div
                key={i}
                class="flex-1 rounded-t transition-all"
                style={{ height: `${height}%`, backgroundColor: color, minWidth: '6px' }}
                title={`${label}: ${Math.round(acc)}% (${d.count ?? 0} predictions)`}
              />
            );
          })}
        </div>
        <div class="flex justify-between text-[10px] mt-1" style="color: var(--text-tertiary)">
          <span>{trend[0]?.date?.slice(5) || ''}</span>
          <span>{trend[trend.length - 1]?.date?.slice(5) || ''}</span>
        </div>
      </div>
    </section>
  );
}

function PredictionFeed({ predictions }) {
  const items = predictions?.predictions || [];

  if (items.length === 0) {
    return (
      <section class="space-y-3">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Recent Predictions</h2>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No predictions yet. The shadow engine generates predictions when state changes occur.</span>
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Recent Predictions</h2>
      <div class="t-card" style="padding: 0;">
        {items.map((p, i) => {
          const typeStyle = TYPE_COLORS[p.prediction_type] || 'background: var(--bg-surface-raised); color: var(--text-secondary);';
          const typeLabel = TYPE_LABELS[p.prediction_type] || p.prediction_type;
          const outcomeStyle = p.outcome ? OUTCOME_COLORS[p.outcome] : null;
          const conf = p.confidence ?? 0;

          return (
            <div key={i} class="flex items-center gap-3 px-4 py-2.5" style={i > 0 ? 'border-top: 1px solid var(--border-subtle)' : ''}>
              <span class="text-xs w-14 flex-shrink-0" style="color: var(--text-tertiary)">{relativeTime(p.timestamp)}</span>
              <span class="text-xs font-medium flex-shrink-0 px-1.5 py-0.5" style={`border-radius: var(--radius); ${typeStyle}`}>{typeLabel}</span>
              <span class="text-sm flex-1 truncate" style="color: var(--text-secondary)">{p.predicted_value || '\u2014'}</span>
              <div class="w-16 flex-shrink-0">
                <div class="h-1.5 rounded-full" style="background: var(--bg-inset)">
                  <div class="h-1.5 rounded-full" style={`background: var(--accent-dim); width: ${Math.round(conf * 100)}%`} />
                </div>
              </div>
              {outcomeStyle && (
                <span class="text-xs font-medium flex-shrink-0 capitalize px-1.5 py-0.5" style={`border-radius: var(--radius); ${outcomeStyle}`}>{p.outcome}</span>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DisagreementsPanel({ disagreements }) {
  const items = disagreements?.disagreements || [];

  if (items.length === 0) {
    return (
      <section class="space-y-3">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Top Disagreements</h2>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--status-healthy)">No disagreements recorded yet.</span>
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Top Disagreements</h2>
      <p class="text-xs" style="color: var(--text-tertiary)">High-confidence wrong predictions — the most informative for learning.</p>
      <div class="space-y-2">
        {items.map((d, i) => {
          const typeStyle = TYPE_COLORS[d.prediction_type] || 'background: var(--bg-surface-raised); color: var(--text-secondary);';
          const typeLabel = TYPE_LABELS[d.prediction_type] || d.prediction_type;
          const conf = d.confidence ?? 0;

          return (
            <div key={i} class="t-card p-3 space-y-1" style="border-left: 4px solid var(--status-warning);">
              <div class="flex items-center gap-2">
                <span class="text-lg font-bold" style="color: var(--status-warning)">{Math.round(conf * 100)}%</span>
                <span class="text-xs font-medium px-1.5 py-0.5" style={`border-radius: var(--radius); ${typeStyle}`}>{typeLabel}</span>
                <span class="text-xs ml-auto" style="color: var(--text-tertiary)">{relativeTime(d.timestamp)}</span>
              </div>
              <div class="text-sm" style="color: var(--text-secondary)">
                <span style="color: var(--text-tertiary)">Predicted:</span> {d.predicted_value || '\u2014'}
              </div>
              {d.actual_value && (
                <div class="text-sm" style="color: var(--text-secondary)">
                  <span style="color: var(--text-tertiary)">Actual:</span> {d.actual_value}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function Shadow() {
  const [pipeline, setPipeline] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [disagreements, setDisagreements] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [advanceError, setAdvanceError] = useState(null);

  async function handleAdvance() {
    setAdvanceError(null);
    try {
      // Use raw fetch instead of postJson to parse structured 400 error bodies
      const res = await fetch(`${baseUrl}/api/pipeline/advance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (body.error === 'Gate not met') {
          setAdvanceError(
            `Gate not met: ${body.gate} requires ${Math.round(body.required * 100)}%, current ${Math.round(body.current * 100)}%`
          );
        } else {
          setAdvanceError(body.detail || body.error || `HTTP ${res.status}`);
        }
        return;
      }
      await fetchAll();
    } catch (err) {
      setAdvanceError(err.message || String(err));
    }
  }

  async function handleRetreat() {
    setAdvanceError(null);
    try {
      const res = await fetch(`${baseUrl}/api/pipeline/retreat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setAdvanceError(body.detail || body.error || `HTTP ${res.status}`);
        return;
      }
      await fetchAll();
    } catch (err) {
      setAdvanceError(err.message || String(err));
    }
  }

  async function fetchAll() {
    setLoading(true);
    setError(null);
    try {
      const [p, a, pr, d] = await Promise.all([
        fetchJson('/api/pipeline'),
        fetchJson('/api/shadow/accuracy'),
        fetchJson('/api/shadow/predictions?limit=20'),
        fetchJson('/api/shadow/disagreements?limit=10'),
      ]);
      setPipeline(p);
      setAccuracy(a);
      setPredictions(pr);
      setDisagreements(d);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchAll(); }, []);

  if (loading && !pipeline) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Shadow Mode</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Prediction accuracy, pipeline progress, and learning insights.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Shadow Mode</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">Prediction accuracy, pipeline progress, and learning insights.</p>
        </div>
        <ErrorState error={error} onRetry={fetchAll} />
      </div>
    );
  }

  return (
    <div class="space-y-8">
      <div class="animate-fade-in-up">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Shadow Mode</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">Prediction accuracy, pipeline progress, and learning insights.</p>
      </div>

      <div class="animate-fade-in-up delay-100">
        <PipelineStage pipeline={pipeline} onAdvance={handleAdvance} onRetreat={handleRetreat} advanceError={advanceError} />
      </div>
      <div class="animate-fade-in-up delay-200">
        <AccuracySummary accuracy={accuracy} pipeline={pipeline} />
      </div>
      <div class="animate-fade-in-up delay-300">
        <DailyTrend trend={accuracy?.daily_trend} />
      </div>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in-up delay-400">
        <PredictionFeed predictions={predictions} />
        <DisagreementsPanel disagreements={disagreements} />
      </div>
    </div>
  );
}
