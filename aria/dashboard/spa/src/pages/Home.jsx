import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import AriaLogo from '../components/AriaLogo.jsx';
import HeroCard from '../components/HeroCard.jsx';
import PageBanner from '../components/PageBanner.jsx';
import PipelineSankey from '../components/PipelineSankey.jsx';
import PipelineStatusBar from '../components/PipelineStatusBar.jsx';
import OodaSummaryCard from '../components/OodaSummaryCard.jsx';
import { relativeTime } from './intelligence/utils.jsx';

export default function Home() {
  const intelligence = useCache('intelligence');
  const activity = useCache('activity_summary');
  const entities = useCache('entities');
  const automations = useCache('automation_suggestions');

  const [health, setHealth] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [shadowAccuracy, setShadowAccuracy] = useState(null);
  const [pipeline, setPipeline] = useState(null);

  useEffect(() => {
    fetchJson('/health').then(setHealth).catch(() => {});
    fetchJson('/api/ml/anomalies').then(setAnomalies).catch(() => {});
    fetchJson('/api/shadow/accuracy').then(setShadowAccuracy).catch(() => {});
    fetchJson('/api/pipeline').then(setPipeline).catch(() => {});
  }, []);

  const loading = intelligence.loading;
  const error = intelligence.error;

  // ── Anomaly hero ──
  const anomalyItems = anomalies?.anomalies || [];
  const criticalCount = anomalyItems.filter((item) => item.severity === 'critical' || (item.score != null && item.score < -0.5)).length;
  const anomalyValue = anomalyItems.length > 0
    ? `${anomalyItems.length} detected${criticalCount > 0 ? ` (${criticalCount} critical)` : ''}`
    : 'Clear';
  const lastAnomaly = anomalyItems.length > 0 ? null : anomalies?.last_anomaly_at;
  const anomalyDelta = anomalyItems.length > 0 ? null : (lastAnomaly ? `last anomaly ${relativeTime(lastAnomaly)}` : null);

  // ── Recommendations hero ──
  const suggestions = useComputed(() => {
    if (!automations.data || !automations.data.data) return [];
    return automations.data.data.suggestions || [];
  }, [automations.data]);
  const pending = suggestions.filter((item) => (item.status || 'pending').toLowerCase() === 'pending');
  const approved = suggestions.filter((item) => (item.status || '').toLowerCase() === 'approved');
  const recValue = pending.length > 0
    ? `${pending.length} pending review`
    : 'None pending';
  const recDelta = pending.length > 0 ? null : (approved.length > 0 ? `${approved.length} approved this week` : null);

  // ── Accuracy hero (trailing 7-day average) ──
  const dailyTrend = shadowAccuracy?.daily_trend || [];
  const last7 = dailyTrend.slice(-7);
  const avg7d = last7.length > 0
    ? last7.reduce((sum, day) => sum + (day.accuracy ?? 0), 0) / last7.length
    : null;
  const accValue = avg7d != null ? `${Math.round(avg7d * 100)}%` : '\u2014';
  const accDelta = avg7d != null ? '7-day avg' : null;

  // ── Observe summary metrics ──
  const actInner = useComputed(() => {
    if (!activity.data || !activity.data.data) return null;
    return activity.data.data;
  }, [activity.data]);
  const occ = actInner ? (actInner.occupancy || null) : null;

  // ── Sankey cache data ──
  const cacheData = useComputed(() => ({
    capabilities: entities.data,
    pipeline: { data: pipeline },
    shadow_accuracy: { data: shadowAccuracy },
    activity_labels: activity.data,
  }), [entities.data, pipeline, shadowAccuracy, activity.data]);

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <div class="t-frame" data-label="aria">
          <AriaLogo className="w-24 mb-1" color="var(--text-primary)" />
          <p class="text-sm" style="color: var(--text-tertiary); font-family: var(--font-mono);">
            Your home at a glance — anomalies, automation suggestions, and prediction accuracy.
          </p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="HOME" subtitle="Your home at a glance — anomalies, automation suggestions, and prediction accuracy." />
        <ErrorState error={error} onRetry={intelligence.refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="HOME" subtitle="Your home at a glance — anomalies, automation suggestions, and prediction accuracy." />

      {/* Three hero cards */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <HeroCard
          value={anomalyValue}
          label="anomalies"
          delta={anomalyDelta}
          warning={anomalyItems.length > 0}
        />
        <HeroCard
          value={recValue}
          label="recommendations"
          delta={recDelta}
        />
        <HeroCard
          value={accValue}
          label="accuracy"
          delta={accDelta}
        />
      </div>

      {/* Pipeline status bar */}
      <PipelineStatusBar />

      {/* OODA summary cards */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <OodaSummaryCard
          title="Observe"
          subtitle="What's happening in your home right now."
          metric={occ && occ.anyone_home ? 'Home' : occ ? 'Away' : null}
          metricLabel="occupancy"
          href="#/observe"
        />
        <OodaSummaryCard
          title="Understand"
          subtitle="What's unusual, what's repeating, and why."
          metric={anomalyItems.length || 0}
          metricLabel={anomalyItems.length === 1 ? 'anomaly' : 'anomalies'}
          href="#/understand"
          accentColor={anomalyItems.length > 0 ? 'var(--status-warning)' : undefined}
        />
        <OodaSummaryCard
          title="Decide"
          subtitle="Automation recommendations to review."
          metric={pending.length}
          metricLabel="pending"
          href="#/decide"
        />
      </div>

      {/* Compact Pipeline Sankey */}
      <PipelineSankey
        moduleStatuses={health?.modules || {}}
        cacheData={cacheData}
      />
    </div>
  );
}
