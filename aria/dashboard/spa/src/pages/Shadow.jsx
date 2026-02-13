import { useState, useEffect } from 'preact/hooks';
import { fetchJson, baseUrl } from '../api.js';
import { relativeTime } from './intelligence/utils.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

const STAGES = ['backtest', 'shadow', 'suggest', 'autonomous'];
const STAGE_LABELS = ['Backtest', 'Shadow', 'Suggest', 'Autonomous'];

const TYPE_COLORS = {
  next_domain_action: 'bg-blue-100 text-blue-700',
  room_activation: 'bg-purple-100 text-purple-700',
  routine_trigger: 'bg-amber-100 text-amber-700',
};
const TYPE_LABELS = {
  next_domain_action: 'Domain',
  room_activation: 'Room',
  routine_trigger: 'Routine',
};
const OUTCOME_COLORS = {
  correct: 'bg-green-100 text-green-700',
  disagreement: 'bg-amber-100 text-amber-700',
  nothing: 'bg-gray-100 text-gray-500',
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
      <h2 class="text-lg font-bold text-gray-900">Pipeline Stage</h2>
      <div class="bg-white rounded-lg shadow-sm p-4 space-y-4">
        <div>
          <div class="flex justify-between text-xs text-gray-500 mb-1">
            {STAGE_LABELS.map((label, i) => (
              <span key={label} class={i <= idx ? 'font-bold text-blue-600' : ''}>{label}</span>
            ))}
          </div>
          <div class="h-3 rounded-full bg-gray-200">
            <div class="h-3 rounded-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
        {enteredAt && (
          <p class="text-xs text-gray-400">
            In {stage} since {new Date(enteredAt).toLocaleDateString()}
          </p>
        )}
        <div class="flex items-center gap-2 mt-3">
          <button
            onClick={onRetreat}
            disabled={idx === 0}
            class="text-xs px-3 py-1.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200 disabled:opacity-30"
          >
            &larr; Retreat
          </button>
          <button
            onClick={onAdvance}
            disabled={idx === STAGES.length - 1}
            class="text-xs px-3 py-1.5 rounded bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-30"
          >
            Advance &rarr;
          </button>
        </div>
        {gate && (
          <p class={`text-xs ${gateMet ? 'text-green-600' : 'text-red-500'}`}>
            Advance to {gate.nextStage} requires &gt;{Math.round(gate.threshold * 100)}% {gate.label}
            {' '}(current: {Math.round(gateValue * 100)}%)
          </p>
        )}
        {advanceError && (
          <p class="text-xs text-red-600 font-medium">{advanceError}</p>
        )}
      </div>
    </section>
  );
}

function accuracyColor(pct) {
  if (pct >= 70) return 'text-green-600';
  if (pct >= 40) return 'text-amber-500';
  return 'text-red-500';
}

function AccuracySummary({ accuracy, pipeline }) {
  const overall = accuracy?.overall_accuracy ?? 0;
  const total = accuracy?.predictions_total ?? 0;
  const correct = accuracy?.predictions_correct ?? 0;
  const disagree = accuracy?.predictions_disagreement ?? 0;
  const nothing = accuracy?.predictions_nothing ?? 0;
  const stage = pipeline?.current_stage || 'backtest';

  const stats = [
    { label: 'Accuracy', value: `${Math.round(overall)}%`, colorClass: accuracyColor(overall) },
    { label: 'Total Predictions', value: total },
    { label: 'Correct', value: correct },
    { label: 'Disagreements', value: disagree },
  ];

  return (
    <section class="space-y-3">
      <div class="flex items-center gap-3">
        <h2 class="text-lg font-bold text-gray-900">Accuracy</h2>
        <span class="text-xs font-medium bg-blue-100 text-blue-700 rounded-full px-2.5 py-0.5 capitalize">{stage}</span>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((s, i) => (
          <div key={i} class="bg-white rounded-lg shadow-sm p-4">
            <div class={`text-2xl font-bold ${s.colorClass || 'text-blue-500'}`}>{s.value}</div>
            <div class="text-sm text-gray-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>
      {nothing > 0 && (
        <p class="text-xs text-gray-400 px-1">{nothing} prediction{nothing !== 1 ? 's' : ''} had no matching outcome (expired with no activity).</p>
      )}
    </section>
  );
}

function DailyTrend({ trend }) {
  if (!trend || trend.length === 0) {
    return (
      <section class="space-y-3">
        <h2 class="text-lg font-bold text-gray-900">Daily Trend</h2>
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-600">
          Accuracy trends will appear after 24-48 hours of predictions.
        </div>
      </section>
    );
  }

  const maxAcc = Math.max(...trend.map(d => d.accuracy ?? 0), 1);

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold text-gray-900">Daily Trend</h2>
      <div class="bg-white rounded-lg shadow-sm p-4">
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
        <div class="flex justify-between text-[10px] text-gray-400 mt-1">
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
        <h2 class="text-lg font-bold text-gray-900">Recent Predictions</h2>
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm text-gray-600">
          No predictions yet. The shadow engine generates predictions when state changes occur.
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold text-gray-900">Recent Predictions</h2>
      <div class="bg-white rounded-lg shadow-sm divide-y divide-gray-50">
        {items.map((p, i) => {
          const typeCls = TYPE_COLORS[p.prediction_type] || 'bg-gray-100 text-gray-600';
          const typeLabel = TYPE_LABELS[p.prediction_type] || p.prediction_type;
          const outcomeCls = p.outcome ? OUTCOME_COLORS[p.outcome] : null;
          const conf = p.confidence ?? 0;

          return (
            <div key={i} class="flex items-center gap-3 px-4 py-2.5">
              <span class="text-xs text-gray-400 w-14 flex-shrink-0">{relativeTime(p.timestamp)}</span>
              <span class={`text-xs font-medium rounded px-1.5 py-0.5 flex-shrink-0 ${typeCls}`}>{typeLabel}</span>
              <span class="text-sm text-gray-700 flex-1 truncate">{p.predicted_value || '\u2014'}</span>
              <div class="w-16 flex-shrink-0">
                <div class="h-1.5 bg-gray-100 rounded-full">
                  <div class="h-1.5 rounded-full bg-blue-400" style={{ width: `${Math.round(conf * 100)}%` }} />
                </div>
              </div>
              {outcomeCls && (
                <span class={`text-xs font-medium rounded px-1.5 py-0.5 flex-shrink-0 capitalize ${outcomeCls}`}>{p.outcome}</span>
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
        <h2 class="text-lg font-bold text-gray-900">Top Disagreements</h2>
        <div class="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-700">
          No disagreements recorded yet.
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold text-gray-900">Top Disagreements</h2>
      <p class="text-xs text-gray-400">High-confidence wrong predictions — the most informative for learning.</p>
      <div class="space-y-2">
        {items.map((d, i) => {
          const typeCls = TYPE_COLORS[d.prediction_type] || 'bg-gray-100 text-gray-600';
          const typeLabel = TYPE_LABELS[d.prediction_type] || d.prediction_type;
          const conf = d.confidence ?? 0;

          return (
            <div key={i} class="bg-white rounded-lg shadow-sm p-3 border-l-4 border-amber-500 space-y-1">
              <div class="flex items-center gap-2">
                <span class="text-lg font-bold text-amber-500">{Math.round(conf * 100)}%</span>
                <span class={`text-xs font-medium rounded px-1.5 py-0.5 ${typeCls}`}>{typeLabel}</span>
                <span class="text-xs text-gray-400 ml-auto">{relativeTime(d.timestamp)}</span>
              </div>
              <div class="text-sm text-gray-700">
                <span class="text-gray-500">Predicted:</span> {d.predicted_value || '\u2014'}
              </div>
              {d.actual_value && (
                <div class="text-sm text-gray-700">
                  <span class="text-gray-500">Actual:</span> {d.actual_value}
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
          <h1 class="text-2xl font-bold text-gray-900">Shadow Mode</h1>
          <p class="text-sm text-gray-500">Prediction accuracy, pipeline progress, and learning insights.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Shadow Mode</h1>
          <p class="text-sm text-gray-500">Prediction accuracy, pipeline progress, and learning insights.</p>
        </div>
        <ErrorState error={error} onRetry={fetchAll} />
      </div>
    );
  }

  return (
    <div class="space-y-8">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Shadow Mode</h1>
        <p class="text-sm text-gray-500">Prediction accuracy, pipeline progress, and learning insights.</p>
      </div>

      <PipelineStage pipeline={pipeline} onAdvance={handleAdvance} onRetreat={handleRetreat} advanceError={advanceError} />
      <AccuracySummary accuracy={accuracy} pipeline={pipeline} />
      <DailyTrend trend={accuracy?.daily_trend} />
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PredictionFeed predictions={predictions} />
        <DisagreementsPanel disagreements={disagreements} />
      </div>
    </div>
  );
}
