import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import PageBanner from '../components/PageBanner.jsx';
import InlineSettings from '../components/InlineSettings.jsx';
import DataSourceConfig from '../components/DataSourceConfig.jsx';
import { Section, Callout } from './intelligence/utils.jsx';
import { AnomalyAlerts } from './intelligence/AnomalyAlerts.jsx';
import { PredictionsVsActuals } from './intelligence/PredictionsVsActuals.jsx';
import { DriftStatus } from './intelligence/DriftStatus.jsx';
import { ShapAttributions } from './intelligence/ShapAttributions.jsx';
import { Baselines } from './intelligence/Baselines.jsx';
import { TrendsOverTime } from './intelligence/TrendsOverTime.jsx';
import { Correlations } from './intelligence/Correlations.jsx';

function PatternsList({ patterns }) {
  if (!patterns || patterns.length === 0) {
    return (
      <Section title="Patterns" subtitle="Recurring event sequences from your logbook.">
        <Callout>No patterns detected yet. Needs several days of logbook data.</Callout>
      </Section>
    );
  }

  return (
    <Section title="Patterns" subtitle="Recurring event sequences from your logbook." summary={`${patterns.length} pattern${patterns.length === 1 ? '' : 's'}`}>
      <div class="space-y-2">
        {patterns.map((pat, idx) => (
          <div key={pat.name || idx} class="t-frame p-3" data-label={pat.name || 'pattern'}>
            <div class="flex items-center justify-between">
              <span class="text-sm font-bold" style="color: var(--text-primary)">{pat.name || 'Unnamed'}</span>
              {pat.type && (
                <span class="text-xs px-2 py-0.5 rounded-full" style="background: var(--bg-surface-raised); color: var(--text-secondary)">{pat.type}</span>
              )}
            </div>
            {pat.description && <p class="text-xs mt-1" style="color: var(--text-secondary)">{pat.description}</p>}
            <div class="flex gap-4 mt-1 text-xs" style="color: var(--text-tertiary)">
              {pat.confidence != null && <span>Confidence: {Math.round(pat.confidence * 100)}%</span>}
              {pat.frequency && <span>{pat.frequency}</span>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function ShadowBrief({ accuracy }) {
  if (!accuracy) return null;

  const dailyTrend = accuracy.daily_trend || [];
  const last7 = dailyTrend.slice(-7);
  const avg7d = last7.length > 0
    ? last7.reduce((sum, day) => sum + (day.accuracy ?? 0), 0) / last7.length
    : null;

  const total = accuracy.predictions_total ?? 0;
  const stage = accuracy.stage || 'backtest';

  return (
    <Section title="Shadow Accuracy" subtitle="Predict-compare-score loop measuring forecast quality." summary={avg7d != null ? `${Math.round(avg7d * 100)}% (7d avg)` : stage}>
      <div class="t-frame p-3" data-label="shadow">
        <div class="flex items-center gap-4 text-sm">
          <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{stage}</span>
          {avg7d != null ? (
            <span style="color: var(--text-secondary)">
              <span class="font-bold" style={`color: ${Math.round(avg7d * 100) >= 70 ? 'var(--status-healthy)' : Math.round(avg7d * 100) >= 40 ? 'var(--status-warning)' : 'var(--status-error)'}`}>
                {Math.round(avg7d * 100)}%
              </span> trailing 7-day accuracy ({total} predictions)
            </span>
          ) : (
            <span style="color: var(--text-tertiary)">No predictions yet</span>
          )}
        </div>
      </div>
    </Section>
  );
}

export default function Understand() {
  const intelligence = useCache('intelligence');
  const patternsCache = useCache('patterns');

  const [anomalies, setAnomalies] = useState(null);
  const [drift, setDrift] = useState(null);
  const [shap, setShap] = useState(null);
  const [shadowAccuracy, setShadowAccuracy] = useState(null);

  useEffect(() => {
    fetchJson('/api/ml/anomalies').then(setAnomalies).catch(() => {});
    fetchJson('/api/ml/drift').then(setDrift).catch(() => {});
    fetchJson('/api/ml/shap').then(setShap).catch(() => {});
    fetchJson('/api/shadow/accuracy').then(setShadowAccuracy).catch(() => {});
  }, []);

  const intel = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data;
  }, [intelligence.data]);

  const patterns = useComputed(() => {
    if (!patternsCache.data || !patternsCache.data.data) return [];
    return patternsCache.data.data.patterns || [];
  }, [patternsCache.data]);

  const loading = intelligence.loading;
  const error = intelligence.error;

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <PageBanner page="UNDERSTAND" subtitle="Anomalies ARIA has found, patterns it's learned, and how accurate its predictions are — with the reasoning behind each." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="UNDERSTAND" subtitle="Anomalies ARIA has found, patterns it's learned, and how accurate its predictions are — with the reasoning behind each." />
        <ErrorState error={error} onRetry={intelligence.refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-8 animate-page-enter">
      <PageBanner page="UNDERSTAND" subtitle="Anomalies ARIA has found, patterns it's learned, and how accurate its predictions are — with the reasoning behind each." />

      <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/anomaly/all'; }} style="cursor: pointer;">
        <AnomalyAlerts anomalies={anomalies} />
      </div>
      <PatternsList patterns={patterns} />

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {intel && (
          <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/prediction/all'; }} style="cursor: pointer;">
            <PredictionsVsActuals predictions={intel.predictions} intradayTrend={intel.intraday_trend} />
          </div>
        )}
        <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/drift/all'; }} style="cursor: pointer;">
          <DriftStatus drift={drift} />
        </div>
      </div>

      <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/anomaly/shap'; }} style="cursor: pointer;">
        <ShapAttributions shap={shap} />
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {intel && (
          <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/baseline/all'; }} style="cursor: pointer;">
            <Baselines baselines={intel.baselines} />
          </div>
        )}
        {intel && <TrendsOverTime trendData={intel.trend_data} intradayTrend={intel.intraday_trend} />}
      </div>
      {intel && (
        <div class="clickable-data" onClick={() => { window.location.hash = '#/detail/correlation/all'; }} style="cursor: pointer;">
          <Correlations correlations={intel.entity_correlations?.top_co_occurrences} />
        </div>
      )}

      <ShadowBrief accuracy={shadowAccuracy} />

      <InlineSettings
        categories={['Anomaly Detection', 'Shadow Engine', 'Drift Detection', 'Forecaster']}
        title="Analysis Settings"
        subtitle="Fine-tune how ARIA detects patterns, anomalies, and forecast accuracy. Most users won't need to change these."
      />
      <DataSourceConfig
        module="anomaly"
        title="Anomaly Sources"
        subtitle="Toggle which detection methods feed the anomaly pipeline."
      />
      <DataSourceConfig
        module="shadow"
        title="Shadow Sources"
        subtitle="Toggle which capabilities are shadow-predicted."
      />
    </div>
  );
}
