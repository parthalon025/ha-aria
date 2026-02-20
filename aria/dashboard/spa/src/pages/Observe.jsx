import { useState, useEffect } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import { wsConnected } from '../store.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import PageBanner from '../components/PageBanner.jsx';
import PresenceCard from '../components/PresenceCard.jsx';
import { HomeRightNow } from './intelligence/HomeRightNow.jsx';
import { ActivitySection } from './intelligence/ActivitySection.jsx';
import InlineSettings from '../components/InlineSettings.jsx';
import DataSourceConfig from '../components/DataSourceConfig.jsx';

export default function Observe() {
  const intelligence = useCache('intelligence');
  const activity = useCache('activity_summary');

  const intel = useComputed(() => {
    if (!intelligence.data || !intelligence.data.data) return null;
    return intelligence.data.data;
  }, [intelligence.data]);

  const actInner = useComputed(() => {
    if (!activity.data || !activity.data.data) return null;
    return activity.data.data;
  }, [activity.data]);

  const loading = intelligence.loading || activity.loading;
  const error = intelligence.error || activity.error;

  if (loading && !intelligence.data) {
    return (
      <div class="space-y-6">
        <PageBanner page="OBSERVE" subtitle="Live view of who's home, what devices are active, and how your home is behaving right now." />
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <PageBanner page="OBSERVE" subtitle="Live view of who's home, what devices are active, and how your home is behaving right now." />
        <ErrorState error={error} onRetry={() => { intelligence.refetch(); activity.refetch(); }} />
      </div>
    );
  }

  const ws = actInner ? (actInner.websocket || null) : null;
  const actRate = actInner ? (actInner.activity_rate || null) : null;
  const evRate = actRate ? actRate.current : null;
  const occ = actInner ? (actInner.occupancy || null) : null;

  const intraday = intel ? intel.intraday_trend : null;
  const latest = Array.isArray(intraday) && intraday.length > 0 ? intraday[intraday.length - 1] : null;
  const lightsOn = latest ? (latest.lights_on ?? null) : null;
  const powerW = latest ? (latest.power_watts ?? null) : null;

  const connected = wsConnected.value;

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="OBSERVE" subtitle="Live view of who's home, what devices are active, and how your home is behaving right now." />

      {/* Live metrics strip */}
      <div class="t-frame" data-label="live metrics">
        <div class="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <a href="#/detail/entity/occupancy" class="clickable-data flex items-center gap-1.5" style="text-decoration: none; color: inherit;">
            <span style="color: var(--text-tertiary)">Occupancy</span>
            <span class="font-medium" style="color: var(--text-primary)">{occ && occ.anyone_home ? 'Home' : occ ? 'Away' : '\u2014'}</span>
          </a>
          <a href="#/detail/module/activity" class="clickable-data flex items-center gap-1.5" style="text-decoration: none; color: inherit;">
            <span style="color: var(--text-tertiary)">Events</span>
            <span class="data-mono font-medium" style="color: var(--text-primary)">{evRate != null ? `${evRate}/min` : '\u2014'}</span>
          </a>
          <a href="#/detail/entity/lights" class="clickable-data flex items-center gap-1.5" style="text-decoration: none; color: inherit;">
            <span style="color: var(--text-tertiary)">Lights</span>
            <span class="data-mono font-medium" style="color: var(--text-primary)">{lightsOn != null ? `${lightsOn} on` : '\u2014'}</span>
          </a>
          <a href="#/detail/entity/power" class="clickable-data flex items-center gap-1.5" style="text-decoration: none; color: inherit;">
            <span style="color: var(--text-tertiary)">Power</span>
            <span class="data-mono font-medium" style="color: var(--text-primary)">{powerW != null ? `${Math.round(powerW)} W` : '\u2014'}</span>
          </a>
          <div class="flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full" style={`background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`} />
            <span style="color: var(--text-tertiary)">WebSocket</span>
            <span class="font-medium" style="color: var(--text-primary)">{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </div>

      <PresenceCard />

      {intel && (
        <HomeRightNow intraday={intel.intraday_trend} baselines={intel.baselines} />
      )}

      {intel && (
        <ActivitySection activity={intel.activity} />
      )}

      <InlineSettings
        categories={['Activity Monitor']}
        title="Observation Settings"
        subtitle="Control how ARIA monitors your home — event rates, occupancy tracking, and WebSocket connection sensitivity."
      />
      <DataSourceConfig
        module="presence"
        title="Presence Sources"
        subtitle="Toggle which signal types feed room occupancy detection."
      />
      <DataSourceConfig
        module="activity"
        title="Activity Sources"
        subtitle="Toggle which entity domains are tracked for activity monitoring."
      />
    </div>
  );
}
