import { useState, useEffect, useMemo } from 'preact/hooks';
import { fetchJson, baseUrl } from '../api.js';
import { relativeTime } from './intelligence/utils.jsx';
import { accuracyColor, SHADOW_PREDICTIONS_LIMIT, SHADOW_DISAGREEMENTS_LIMIT } from '../constants.js';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import TimeChart from '../components/TimeChart.jsx';

const STAGES = ['backtest', 'shadow', 'suggest', 'autonomous'];

const TYPE_COLORS = {
  next_domain_action: 'background: var(--accent-glow); color: var(--accent);',
  room_activation: 'background: var(--accent-purple-glow); color: var(--accent-purple);',
  routine_trigger: 'background: var(--status-warning-glow); color: var(--status-warning);',
};
const TYPE_LABELS = {
  next_domain_action: 'Domain',
  room_activation: 'Room',
  routine_trigger: 'Routine',
};
const OUTCOME_COLORS = {
  correct: 'background: var(--status-healthy-glow); color: var(--status-healthy);',
  disagreement: 'background: var(--status-warning-glow); color: var(--status-warning);',
  nothing: 'background: var(--bg-surface-raised); color: var(--text-tertiary);',
};

// Gate thresholds come from /api/pipeline response — fallback for initial render
const DEFAULT_GATES = {
  backtest: { field: 'backtest_accuracy', threshold: 0.40, label: 'backtest accuracy' },
  shadow: { field: 'shadow_accuracy_7d', threshold: 0.50, label: '7-day shadow accuracy' },
  suggest: { field: 'suggest_approval_rate_14d', threshold: 0.70, label: '14-day approval rate' },
};

function PipelineStage({ pipeline, onAdvance, onRetreat, advanceError }) {
  const stages = pipeline?.stages || STAGES;
  const gates = pipeline?.gates || DEFAULT_GATES;
  const stage = pipeline?.current_stage || 'backtest';
  const idx = stages.indexOf(stage);
  const pct = Math.max(((idx + 1) / stages.length) * 100, 10);
  const enteredAt = pipeline?.stage_entered_at;

  const gate = gates[stage];
  const gateValue = gate ? (pipeline?.[gate.field] ?? 0) : null;
  const gateMet = gate ? gateValue >= gate.threshold : false;
  const nextStage = idx < stages.length - 1 ? stages[idx + 1] : null;

  return (
    <section class="space-y-3">
      <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Pipeline Stage</h2></div>
      <div class="t-frame" data-label={`stage: ${stage}`} style="padding: 1rem;">
        <div class="space-y-4">
          <div>
            <div class="flex justify-between text-xs mb-1" style="color: var(--text-tertiary)">
              {stages.map((s, i) => (
                <span key={s} class="capitalize" style={i <= idx ? 'font-weight: 700; color: var(--accent)' : ''}>{s}</span>
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
              disabled={idx === stages.length - 1}
              class="t-btn t-btn-primary text-xs px-3 py-1.5"
            >
              Advance &rarr;
            </button>
          </div>
          {gate && (
            <p class="text-xs" style={gateMet ? 'color: var(--status-healthy)' : 'color: var(--status-error)'}>
              Advance{nextStage ? ` to ${nextStage}` : ''} requires &gt;{Math.round(gate.threshold * 100)}% {gate.label}
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
      <div class="flex items-center gap-3 t-section-header" style="padding-bottom: 6px;">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Accuracy</h2>
        <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{stage}</span>
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s, i) => (
          <div key={i} class="t-frame" data-label={s.label.toLowerCase()} style="padding: 1rem;">
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

function ThompsonExplorer({ accuracy }) {
  const thompson = accuracy?.thompson_sampling;

  if (!thompson || !thompson.buckets || Object.keys(thompson.buckets).length === 0) {
    return (
      <section class="space-y-3">
        <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Thompson Sampling</h2></div>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">Thompson sampling stats will appear after the shadow engine has made predictions.</span>
        </div>
      </section>
    );
  }

  const bucketKeys = Object.keys(thompson.buckets);

  return (
    <section class="space-y-3">
      <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Thompson Sampling</h2></div>
      <p class="text-xs" style="color: var(--text-tertiary)">Thompson Sampling decides which predictions to try. Each time-of-day slot builds its own confidence — ARIA explores more in uncertain slots and exploits what works in confident ones.</p>
      <div class="flex items-center gap-3 flex-wrap">
        <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{thompson.strategy}</span>
        <span class="text-xs" style="color: var(--text-tertiary)">discount: <span style="color: var(--text-secondary); font-weight: 600;">{thompson.discount_factor}</span></span>
        <span class="text-xs" style="color: var(--text-tertiary)">window: <span style="color: var(--text-secondary); font-weight: 600;">{thompson.window_size}</span></span>
      </div>
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {bucketKeys.map((key) => {
          const b = thompson.buckets[key];
          const exploreColor = b.explore_rate > 0.5 ? 'color: var(--status-warning)' : 'color: var(--status-healthy)';
          return (
            <div key={key} class="t-frame" data-label={key} style="padding: 0.75rem;">
              <div class="text-lg font-bold" style="color: var(--accent)">{b.alpha}/{b.beta}</div>
              <div class="text-xs mt-1" style="color: var(--text-tertiary)">alpha/beta</div>
              <div class="text-sm font-bold mt-2" style={exploreColor}>{Math.round(b.explore_rate * 100)}%</div>
              <div class="text-xs" style="color: var(--text-tertiary)">explore rate</div>
              <div class="text-xs mt-1" style="color: var(--text-secondary)">{b.samples} samples</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CorrectionPropagation({ propagation }) {
  if (!propagation || propagation.enabled === false) {
    return (
      <section class="space-y-3">
        <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Correction Propagation</h2></div>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">Correction propagation is not active. It will activate once the shadow engine has enough prediction data to propagate corrections.</span>
        </div>
      </section>
    );
  }

  const stats = propagation.stats || {};
  const bufferSize = stats.replay_buffer_size ?? 0;
  const bufferCapacity = stats.replay_buffer_capacity ?? 1;
  const fillPct = bufferCapacity > 0 ? Math.round((bufferSize / bufferCapacity) * 100) : 0;

  return (
    <section class="space-y-3">
      <div class="flex items-center gap-3 t-section-header" style="padding-bottom: 6px;">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">Correction Propagation</h2>
        <span class="text-xs font-medium rounded-full px-2.5 py-0.5" style="background: var(--status-healthy-glow); color: var(--status-healthy)">enabled</span>
      </div>
      <p class="text-xs" style="color: var(--text-tertiary)">When ARIA gets a prediction wrong, it propagates the correction to similar situations nearby in time and space. The replay buffer stores recent corrections for learning.</p>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="t-frame" data-label="replay buffer" style="padding: 1rem;">
          <div class="text-2xl font-bold" style="color: var(--accent)">{bufferSize} / {bufferCapacity}</div>
          <div class="text-sm mt-1" style="color: var(--text-tertiary)">replay buffer</div>
          <div class="h-1.5 rounded-full mt-2" style="background: var(--bg-inset)">
            <div class="h-1.5 rounded-full" style={`background: var(--accent); width: ${fillPct}%`} />
          </div>
        </div>
        <div class="t-frame" data-label="cell observations" style="padding: 1rem;">
          <div class="text-2xl font-bold" style="color: var(--accent)">{stats.cell_observations ?? 0}</div>
          <div class="text-sm mt-1" style="color: var(--text-tertiary)">cell observations</div>
        </div>
        <div class="t-frame" data-label="kernel bandwidth" style="padding: 1rem;">
          <div class="text-2xl font-bold" style="color: var(--accent)">{stats.bandwidth ?? 0}</div>
          <div class="text-sm mt-1" style="color: var(--text-tertiary)">kernel bandwidth</div>
        </div>
      </div>
    </section>
  );
}

function DailyTrend({ trend, pipeline }) {
  const gates = pipeline?.gates || DEFAULT_GATES;
  const stages = pipeline?.stages || STAGES;
  const stage = pipeline?.current_stage || 'backtest';
  const gate = gates[stage];
  const thresholdPct = gate ? Math.round(gate.threshold * 100) : null;
  const idx = stages.indexOf(stage);
  const nextStage = idx < stages.length - 1 ? stages[idx + 1] : null;

  const { chartData, chartSeries } = useMemo(() => {
    if (!trend || trend.length === 0) return { chartData: null, chartSeries: [] };

    // Build timestamps and raw values
    const timestamps = [];
    const rawAccuracy = [];
    const counts = [];

    for (const d of trend) {
      // Parse date string as noon UTC to avoid timezone shifting
      const ts = Math.floor(new Date(d.date + 'T12:00:00Z').getTime() / 1000);
      timestamps.push(ts);
      rawAccuracy.push(d.accuracy ?? null);
      counts.push(d.count ?? 0);
    }

    // Compute 7-day rolling average
    const rolling = [];
    for (let i = 0; i < rawAccuracy.length; i++) {
      const windowStart = Math.max(0, i - 6);
      let sum = 0;
      let validCount = 0;
      for (let j = windowStart; j <= i; j++) {
        if (rawAccuracy[j] != null) {
          sum += rawAccuracy[j];
          validCount++;
        }
      }
      rolling.push(validCount > 0 ? sum / validCount : null);
    }

    return {
      chartData: [timestamps, rolling, counts],
      chartSeries: [
        { label: '7-day Accuracy', color: 'var(--accent)', width: 2 },
        { label: 'Predictions', color: 'var(--text-tertiary)', width: 1 },
      ],
    };
  }, [trend]);

  if (!trend || trend.length === 0) {
    return (
      <section class="space-y-3">
        <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Daily Trend</h2></div>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">Accuracy trends will appear after 24-48 hours of predictions.</span>
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Daily Trend</h2>
      <p class="text-xs" style="color: var(--text-tertiary)">How accurate are ARIA's predictions over time? The line smooths out daily swings to show the real trend. Higher = better guesses.</p>
      <div class="t-frame" data-label="accuracy trend" style="padding: 1rem;">
        <TimeChart data={chartData} series={chartSeries} height={160} />
        {/* Legend */}
        <div class="flex flex-wrap items-center gap-3 mt-2 pt-2" style="border-top: 1px solid var(--border-subtle); font-size: var(--type-micro); color: var(--text-tertiary);">
          <div class="flex items-center gap-1">
            <span style="display: inline-block; width: 16px; height: 2px; background: var(--accent);" />
            <span>7-day rolling accuracy</span>
          </div>
          <div class="flex items-center gap-1">
            <span style="display: inline-block; width: 16px; height: 2px; background: var(--text-tertiary);" />
            <span>Predictions per day</span>
          </div>
        </div>
        {thresholdPct != null && (
          <p class="text-xs mt-2" style="color: var(--text-tertiary)">
            Gate threshold: <span style="color: var(--accent); font-weight: 600;">{thresholdPct}%</span> {gate.label} required{nextStage ? ` to advance to ${nextStage}` : ''}
          </p>
        )}
      </div>
    </section>
  );
}

function PredictionFeed({ predictions }) {
  const items = predictions?.predictions || [];

  if (items.length === 0) {
    return (
      <section class="space-y-3">
        <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Recent Predictions</h2></div>
        <div class="t-callout" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No predictions yet. The shadow engine generates predictions when state changes occur.</span>
        </div>
      </section>
    );
  }

  return (
    <section class="space-y-3">
      <h2 class="text-lg font-bold" style="color: var(--text-primary)">Recent Predictions</h2>
      <div class="t-frame" data-label="recent predictions" style="padding: 0;">
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
        <div class="t-section-header" style="padding-bottom: 6px;"><h2 class="text-lg font-bold" style="color: var(--text-primary)">Top Disagreements</h2></div>
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
            <div key={i} class="t-frame p-3 space-y-1" data-label="disagreement" style="border-left: 4px solid var(--status-warning);">
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
  const [propagation, setPropagation] = useState(null);
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
      const [p, a, pr, d, prop] = await Promise.all([
        fetchJson('/api/pipeline'),
        fetchJson('/api/shadow/accuracy'),
        fetchJson(`/api/shadow/predictions?limit=${SHADOW_PREDICTIONS_LIMIT}`),
        fetchJson(`/api/shadow/disagreements?limit=${SHADOW_DISAGREEMENTS_LIMIT}`),
        fetchJson('/api/shadow/propagation'),
      ]);
      setPipeline(p);
      setAccuracy(a);
      setPredictions(pr);
      setDisagreements(d);
      setPropagation(prop);
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
    <div class="space-y-8 animate-page-enter">
      <PageBanner page="SHADOW" subtitle="Predict-compare-score validation loop." />

      {/* Hero — learning by watching */}
      <HeroCard
        value={Math.round(accuracy?.overall_accuracy ?? 0)}
        label="shadow accuracy"
        unit="%"
        delta={`${pipeline?.current_stage || 'backtest'} stage \u2022 ${accuracy?.predictions_total ?? 0} predictions`}
        sparkData={useMemo(() => {
          const trend = accuracy?.daily_trend;
          if (!trend || trend.length < 2) return null;
          const ts = [];
          const vals = [];
          for (const d of trend) {
            ts.push(Math.floor(new Date(d.date + 'T12:00:00Z').getTime() / 1000));
            vals.push(d.accuracy ?? null);
          }
          return [ts, vals];
        }, [accuracy])}
      />

      <PipelineStage pipeline={pipeline} onAdvance={handleAdvance} onRetreat={handleRetreat} advanceError={advanceError} />
      <AccuracySummary accuracy={accuracy} pipeline={pipeline} />
      <ThompsonExplorer accuracy={accuracy} />
      <DailyTrend trend={accuracy?.daily_trend} pipeline={pipeline} />
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PredictionFeed predictions={predictions} />
        <DisagreementsPanel disagreements={disagreements} />
      </div>
      <CorrectionPropagation propagation={propagation} />
    </div>
  );
}
