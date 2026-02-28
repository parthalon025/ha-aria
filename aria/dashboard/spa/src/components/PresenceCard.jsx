import useCache from '../hooks/useCache.js';

/**
 * Probability bar with color-coded fill.
 * Green >= 70%, amber 30-69%, red < 30%.
 */
function ProbBar({ value }) {
  const pct = Math.round((value || 0) * 100);
  const color = pct >= 70 ? 'var(--status-healthy)' : pct >= 30 ? 'var(--status-warning)' : 'var(--status-error)';
  return (
    <div class="flex items-center gap-2">
      <div class="flex-1 h-1.5 rounded-full" style="background: var(--bg-inset)">
        <div class="h-1.5 rounded-full" style={`width: ${pct}%; background: ${color}; min-width: ${pct > 0 ? '3px' : '0'}; transition: width 0.3s ease;`} />
      </div>
      <span class="data-mono text-xs" style={`color: ${color}; min-width: 28px; text-align: right;`}>{pct}%</span>
    </div>
  );
}

/**
 * PresenceCard — compact summary card for room occupancy.
 *
 * Shows: overall home probability (hero metric), per-room probability bars,
 * identified persons count, camera signal count, and MQTT connection status.
 *
 * Subscribes to the "presence" cache category via useCache (WebSocket-driven).
 */
export default function PresenceCard() {
  const { data, loading } = useCache('presence');

  const presence = (data && data.data) || {};
  const rooms = presence.rooms || {};
  const persons = presence.identified_persons || {};
  const cameraRooms = presence.camera_rooms || {};
  const mqttConnected = presence.mqtt_connected;
  const occupiedRooms = presence.occupied_rooms || [];

  const roomEntries = Object.entries(rooms).sort((a, b) => (b[1].probability || 0) - (a[1].probability || 0));
  const personCount = Object.keys(persons).length;
  const cameraCount = Object.keys(cameraRooms).length;
  const totalSignals = roomEntries.reduce((sum, [, r]) => sum + (r.signals || []).length, 0);

  // Overall probability: use the max room probability as the home occupancy indicator
  const overallProb = presence.overall_probability !== null && presence.overall_probability !== undefined
    ? presence.overall_probability
    : (roomEntries.length > 0 ? Math.max(...roomEntries.map(([, r]) => r.probability || 0)) : 0);
  const overallPct = Math.round(overallProb * 100);
  const overallColor = overallPct >= 70 ? 'var(--status-healthy)' : overallPct >= 30 ? 'var(--status-warning)' : 'var(--status-error)';

  // Loading skeleton
  if (loading && !data) {
    return (
      <div class="t-frame" data-label="presence" style="min-height: 120px;">
        <div class="flex items-center gap-2">
          <div class="w-12 h-8 rounded" style="background: var(--bg-inset); animation: pulse 1.5s infinite;" />
          <div class="flex-1 h-3 rounded" style="background: var(--bg-inset); animation: pulse 1.5s infinite;" />
        </div>
      </div>
    );
  }

  return (
    <div class="t-frame" data-label="presence">
      {/* Hero: overall occupancy */}
      <div class="flex items-start justify-between mb-3">
        <div>
          <div class="flex items-baseline gap-2">
            <span class="data-mono" style={`font-size: var(--type-hero); font-weight: 600; color: ${overallColor}; line-height: 1;`}>
              {overallPct}%
            </span>
          </div>
          <div class="text-xs mt-1" style="color: var(--text-tertiary)">
            {occupiedRooms.length === 0
              ? 'No rooms occupied'
              : occupiedRooms.length === 1
                ? '1 room occupied'
                : `${occupiedRooms.length} rooms occupied`}
          </div>
        </div>

        {/* MQTT status */}
        <div class="flex items-center gap-1.5 text-xs flex-shrink-0">
          <span class="inline-block w-2 h-2 rounded-full" style={`background: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`} />
          <span style={`color: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`}>
            MQTT
          </span>
        </div>
      </div>

      {/* Stats row */}
      <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs mb-3" style="color: var(--text-tertiary)">
        <span>{personCount} person{personCount !== 1 ? 's' : ''}</span>
        <span>{cameraCount} camera{cameraCount !== 1 ? 's' : ''}</span>
        <span>{totalSignals} signal{totalSignals !== 1 ? 's' : ''}</span>
      </div>

      {/* Per-room occupancy bars */}
      {roomEntries.length > 0 ? (
        <div class="space-y-2">
          {roomEntries.slice(0, 6).map(([name, room]) => (
            <a key={name} href={`#/detail/room/${name}`} class="clickable-data block" style="text-decoration: none; color: inherit;">
              <div class="text-xs mb-0.5" style="color: var(--text-secondary); text-transform: capitalize;">
                {name.replace(/_/g, ' ')}
              </div>
              <ProbBar value={room.probability} />
            </a>
          ))}
          {roomEntries.length > 6 && (
            <div class="text-xs" style="color: var(--text-tertiary)">
              +{roomEntries.length - 6} more room{roomEntries.length - 6 !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      ) : (
        <div class="text-xs" style="color: var(--text-tertiary)">
          No rooms detected yet.
        </div>
      )}

      {/* Link to full page */}
      <a
        href="#/presence"
        class="block text-xs mt-3"
        style="color: var(--accent); text-decoration: none;"
      >
        View full presence data {'\u2192'}
      </a>
    </div>
  );
}
