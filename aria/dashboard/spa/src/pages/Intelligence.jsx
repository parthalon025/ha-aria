import { useState, useEffect, useMemo } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { fetchJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import HeroCard from '../components/HeroCard.jsx';
import { Callout, Section } from './intelligence/utils.jsx';
import { LearningProgress } from './intelligence/LearningProgress.jsx';
import { HomeRightNow } from './intelligence/HomeRightNow.jsx';
import { ActivitySection } from './intelligence/ActivitySection.jsx';
import { TrendsOverTime } from './intelligence/TrendsOverTime.jsx';
import { PredictionsVsActuals } from './intelligence/PredictionsVsActuals.jsx';
import { Baselines } from './intelligence/Baselines.jsx';
import { DailyInsight } from './intelligence/DailyInsight.jsx';
import { Correlations } from './intelligence/Correlations.jsx';
import { SystemStatus } from './intelligence/SystemStatus.jsx';

function ShadowBrief({ shadowAccuracy, pipeline }) {
  if (!shadowAccuracy && !pipeline) return null;

  const total = shadowAccuracy?.predictions_total ?? 0;
  const correct = shadowAccuracy?.predictions_correct ?? 0;
  const acc = shadowAccuracy?.overall_accuracy ?? 0;
  const stage = pipeline?.current_stage || shadowAccuracy?.stage || 'backtest';

  const accStyle = acc >= 70
    ? 'color: var(--status-healthy)'
    : acc >= 40
      ? 'color: var(--status-warning)'
      : 'color: var(--status-error)';

  const summaryText = total > 0 ? `${Math.round(acc)}% accuracy` : stage;

  return (
    <Section title="Shadow Engine" subtitle="Predict-compare-score loop running alongside the main engine." summary={summaryText}>
      <div class="t-frame p-4" data-label="shadow">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-4">
            <span class="text-xs font-medium rounded-full px-2.5 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{stage}</span>
            {total > 0 ? (
              <span class="text-sm" style="color: var(--text-secondary)">
                <span class="font-bold" style={accStyle}>{Math.round(acc)}%</span> accuracy ({correct}/{total})
              </span>
            ) : (
              <span class="text-sm" style="color: var(--text-tertiary)">No predictions yet</span>
            )}
          </div>
          <a href="#/shadow" class="text-sm font-medium" style="color: var(--accent)">Full details &rarr;</a>
        </div>
        {/* Gate progress toward next stage */}
        {pipeline && (() => {
          const stg = pipeline.current_stage || 'backtest';
          // Gate thresholds — must stay in sync with PIPELINE_GATES in hub/api.py
          const gates = {
            backtest: { field: 'backtest_accuracy', threshold: 0.40, label: 'backtest accuracy' },
            shadow: { field: 'shadow_accuracy_7d', threshold: 0.50, label: '7-day shadow accuracy' },
            suggest: { field: 'suggest_approval_rate_14d', threshold: 0.70, label: '14-day approval rate' },
          };
          const gate = gates[stg];
          if (!gate) return null; // autonomous = no next gate
          const current = pipeline[gate.field] ?? 0;
          const pct = Math.min(100, Math.round((current / gate.threshold) * 100));
          const met = current >= gate.threshold;
          return (
            <div class="mt-3 pt-3" style="border-top: 1px solid var(--border-subtle)">
              <div class="flex items-center justify-between text-xs mb-1">
                <span style="color: var(--text-tertiary)">Gate: {gate.label} &ge; {Math.round(gate.threshold * 100)}%</span>
                <span class={met ? 'font-medium' : ''} style={met ? 'color: var(--status-healthy)' : 'color: var(--text-tertiary)'}>{Math.round(current * 100)}%</span>
              </div>
              <div class="h-1.5 rounded-full" style="background: var(--bg-inset)">
                <div class="h-1.5 rounded-full" style={`background: ${met ? 'var(--status-healthy)' : 'var(--accent)'}; width: ${pct}%`} />
              </div>
            </div>
          );
        })()}
      </div>
    </Section>
  );
}

export default function Intelligence() {
  const { data, loading, error, refetch } = useCache('intelligence');
  const [shadowAccuracy, setShadowAccuracy] = useState(null);
  const [pipeline, setPipeline] = useState(null);

  useEffect(() => {
    fetchJson('/api/shadow/accuracy').then(setShadowAccuracy).catch(() => {});
    fetchJson('/api/pipeline').then(setPipeline).catch(() => {});
  }, []);

  const intel = useComputed(() => {
    if (!data || !data.data) return null;
    return data.data;
  }, [data]);

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Intelligence</h1>
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Intelligence</h1>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  if (!intel) {
    return (
      <div class="space-y-6">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Intelligence</h1>
        <Callout>Intelligence data is loading. The engine collects its first snapshot automatically via cron.</Callout>
      </div>
    );
  }

  // Compute hero metric: current power draw
  const latestIntraday = intel.intraday_trend?.length > 0 ? intel.intraday_trend[intel.intraday_trend.length - 1] : null;
  const currentPower = latestIntraday?.power_watts ?? null;

  // Compute delta vs baseline if available
  let powerDelta = null;
  if (currentPower != null && intel.baselines) {
    const today = new Date().toLocaleDateString('en-US', { weekday: 'long' });
    const baseline = intel.baselines[today];
    if (baseline?.power_watts?.mean != null) {
      const diff = currentPower - baseline.power_watts.mean;
      const pct = baseline.power_watts.mean > 0 ? Math.round((diff / baseline.power_watts.mean) * 100) : 0;
      if (Math.abs(pct) >= 10) {
        powerDelta = `${pct > 0 ? '+' : ''}${pct}% vs ${today}`;
      } else {
        powerDelta = `typical for ${today}`;
      }
    }
  }

  // Compute sparkline from intraday power trend (entries have {hour: 0-23, power_watts, ...})
  const powerSparkData = useMemo(() => {
    const trend = intel.intraday_trend;
    if (!trend || trend.length < 2) return null;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayStart = Math.floor(today.getTime() / 1000);
    const ts = [];
    const vals = [];
    for (const d of trend) {
      ts.push(dayStart + (d.hour ?? 0) * 3600);
      vals.push(d.power_watts ?? null);
    }
    return [ts, vals];
  }, [intel.intraday_trend]);

  return (
    <div class="space-y-8 animate-page-enter">
      <div class="t-section-header" style="padding-bottom: 8px;">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Intelligence</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">Your home's learning engine — data maturity, real-time activity, baselines, and ML insights.</p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Row 1: HeroCard (full width) */}
        <div class="sm:col-span-2">
          <HeroCard
            value={currentPower}
            label="home right now"
            unit="W"
            delta={powerDelta}
            sparkData={powerSparkData}
          />
        </div>

        {/* Row 2: LearningProgress + ShadowBrief (2-col on tablet+) */}
        <LearningProgress maturity={intel.data_maturity} shadowStage={pipeline?.current_stage} shadowAccuracy={shadowAccuracy?.overall_accuracy} />
        <ShadowBrief shadowAccuracy={shadowAccuracy} pipeline={pipeline} />

        {/* Row 3: HomeRightNow (full width) */}
        <div class="sm:col-span-2">
          <HomeRightNow intraday={intel.intraday_trend} baselines={intel.baselines} />
        </div>

        {/* Row 4: ActivitySection (full width) */}
        <div class="sm:col-span-2">
          <ActivitySection activity={intel.activity} />
        </div>

        {/* Row 5: TrendsOverTime + PredictionsVsActuals (2-col on tablet+) */}
        <TrendsOverTime trendData={intel.trend_data} intradayTrend={intel.intraday_trend} />
        <PredictionsVsActuals predictions={intel.predictions} intradayTrend={intel.intraday_trend} />

        {/* Row 6: Baselines + DailyInsight (2-col on tablet+) */}
        <Baselines baselines={intel.baselines} />
        <DailyInsight insight={intel.daily_insight} />

        {/* Row 7: Correlations + SystemStatus (2-col on tablet+) */}
        <Correlations correlations={intel.correlations} />
        <SystemStatus runLog={intel.run_log} mlModels={intel.ml_models} metaLearning={intel.meta_learning} />

        {/* Row 8: Configuration redirect (full width) */}
        <div class="sm:col-span-2">
          <Section title="Configuration" subtitle="Engine parameters are now managed in Settings.">
            <div class="t-callout p-3 text-sm flex items-center justify-between">
              <span>Engine settings have moved to the dedicated Settings page.</span>
              <a href="#/settings" class="font-medium whitespace-nowrap" style="color: var(--accent)">Settings &rarr;</a>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}
