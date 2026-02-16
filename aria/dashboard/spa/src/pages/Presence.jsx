import useCache from '../hooks/useCache.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import PageBanner from '../components/PageBanner.jsx';
import { Section, Callout, relativeTime } from './intelligence/utils.jsx';
import { confidenceBadgeStyle } from '../constants.js';

// Signal type labels and colors
const SIGNAL_LABELS = {
  camera_person: 'Camera',
  camera_face: 'Face ID',
  motion: 'Motion',
  light_interaction: 'Light',
  dimmer_press: 'Dimmer',
  door: 'Door',
  device_tracker: 'Tracker',
};

const SIGNAL_COLORS = {
  camera_person: 'var(--accent)',
  camera_face: 'var(--status-healthy)',
  motion: 'var(--accent-purple)',
  light_interaction: 'var(--accent-warm)',
  dimmer_press: 'var(--accent-warm)',
  door: 'var(--text-secondary)',
  device_tracker: 'var(--status-healthy)',
};

/** Probability bar with gradient fill */
function ProbabilityBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? 'var(--status-healthy)' : pct >= 30 ? 'var(--status-warning)' : 'var(--accent)';
  return (
    <div class="flex items-center gap-2">
      <div class="flex-1 h-2 rounded-full" style="background: var(--bg-inset)">
        <div class="h-2 rounded-full transition-all" style={`width: ${pct}%; background: ${color}; min-width: ${pct > 0 ? '4px' : '0'}`} />
      </div>
      <span class="text-xs font-bold" style={`color: ${color}; min-width: 32px; text-align: right;`}>{pct}%</span>
    </div>
  );
}

/** Room card showing probability, signal count, and recent signals */
function RoomCard({ name, room }) {
  const signalCount = (room.signals || []).length;
  const personCount = (room.persons || []).length;
  const isOccupied = room.probability >= 0.5;
  const latestSignal = room.signals && room.signals.length > 0 ? room.signals[room.signals.length - 1] : null;

  // Group signals by type for the breakdown
  const signalGroups = {};
  (room.signals || []).forEach(s => {
    const type = s.type || 'unknown';
    if (!signalGroups[type]) signalGroups[type] = 0;
    signalGroups[type]++;
  });

  return (
    <div class="t-frame p-4" style={isOccupied ? 'border-left: 4px solid var(--status-healthy)' : ''}>
      <div class="flex items-start justify-between mb-2">
        <div>
          <div class="text-sm font-bold" style="color: var(--text-primary); text-transform: capitalize;">{name.replace(/_/g, ' ')}</div>
          <div class="text-xs mt-0.5" style="color: var(--text-tertiary)">
            {signalCount} signal{signalCount !== 1 ? 's' : ''}
            {personCount > 0 && ` \u00B7 ${personCount} identified`}
          </div>
        </div>
        <span class="text-xs font-medium rounded px-1.5 py-0.5" style={confidenceBadgeStyle(room.probability)}>
          {room.confidence || (room.probability >= 0.7 ? 'high' : room.probability >= 0.3 ? 'medium' : 'low')}
        </span>
      </div>

      <ProbabilityBar value={room.probability} />

      {/* Signal type breakdown */}
      {Object.keys(signalGroups).length > 0 && (
        <div class="flex flex-wrap gap-1.5 mt-2">
          {Object.entries(signalGroups).map(([type, count]) => (
            <span key={type} class="text-xs rounded px-1.5 py-0.5" style={`background: var(--bg-surface-raised); color: ${SIGNAL_COLORS[type] || 'var(--text-tertiary)'}`}>
              {SIGNAL_LABELS[type] || type} ({count})
            </span>
          ))}
        </div>
      )}

      {/* Latest signal detail */}
      {latestSignal && (
        <div class="text-xs mt-2 truncate" style="color: var(--text-tertiary)" title={latestSignal.detail}>
          {latestSignal.detail}
        </div>
      )}
    </div>
  );
}

/** Identified person row */
function PersonRow({ name, info }) {
  return (
    <div class="flex items-center gap-3 py-2" style="border-bottom: 1px solid var(--border-subtle)">
      {/* Avatar circle with initial */}
      <div class="flex items-center justify-center rounded-full" style="width: 32px; height: 32px; background: var(--accent-glow); color: var(--accent); font-weight: 700; font-size: 14px; flex-shrink: 0;">
        {(name || '?')[0].toUpperCase()}
      </div>
      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium truncate" style="color: var(--text-primary); text-transform: capitalize;">{name}</div>
        <div class="text-xs" style="color: var(--text-tertiary)">
          {info.room ? info.room.replace(/_/g, ' ') : 'Unknown room'}
          {info.confidence != null && ` \u00B7 ${Math.round(info.confidence * 100)}%`}
        </div>
      </div>
      {info.last_seen && (
        <div class="text-xs flex-shrink-0" style="color: var(--text-tertiary)">{relativeTime(info.last_seen)}</div>
      )}
    </div>
  );
}

/** Camera status strip */
function CameraStrip({ cameraRooms }) {
  if (!cameraRooms || Object.keys(cameraRooms).length === 0) return null;

  return (
    <div class="flex flex-wrap gap-2">
      {Object.entries(cameraRooms).map(([camera, room]) => (
        <div key={camera} class="text-xs rounded px-2 py-1" style="background: var(--bg-surface-raised); color: var(--text-secondary);">
          <span style="color: var(--status-healthy);">{'\u25CF'} </span>
          <span style="text-transform: capitalize;">{camera.replace(/_/g, ' ')}</span>
          <span style="color: var(--text-tertiary);"> \u2192 {room.replace(/_/g, ' ')}</span>
        </div>
      ))}
    </div>
  );
}

/** Signal feed — recent signals across all rooms, newest first */
function SignalFeed({ rooms }) {
  // Collect all signals with room context
  const allSignals = [];
  Object.entries(rooms || {}).forEach(([roomName, room]) => {
    (room.signals || []).forEach((s, i) => {
      allSignals.push({ ...s, room: roomName, _idx: i });
    });
  });

  // Show latest 15 (signals are already in order, take from each room's tail)
  const recent = allSignals.slice(-15).reverse();

  if (recent.length === 0) {
    return <p class="text-sm" style="color: var(--text-tertiary)">No signals received yet. Walk past a camera or trigger a motion sensor.</p>;
  }

  return (
    <div class="space-y-1">
      {recent.map((s, i) => {
        const color = SIGNAL_COLORS[s.type] || 'var(--text-tertiary)';
        return (
          <div key={i} class="flex items-center gap-2 py-1 px-1 rounded text-sm">
            <span class="w-2 h-2 rounded-full flex-shrink-0" style={`background: ${color}`} />
            <span class="flex-1 truncate" style="color: var(--text-secondary)" title={s.detail}>{s.detail || s.type}</span>
            <span class="text-xs flex-shrink-0 rounded px-1" style="background: var(--bg-surface-raised); color: var(--text-tertiary); text-transform: capitalize;">
              {s.room.replace(/_/g, ' ')}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function Presence() {
  const { data, loading, error, refetch } = useCache('presence');

  if (loading && !data) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const presence = data || {};
  const rooms = presence.rooms || {};
  const occupiedRooms = presence.occupied_rooms || [];
  const persons = presence.identified_persons || {};
  const cameraRooms = presence.camera_rooms || {};
  const mqttConnected = presence.mqtt_connected;
  const roomNames = Object.keys(rooms);
  const personNames = Object.keys(persons);
  const totalSignals = roomNames.reduce((sum, r) => sum + (rooms[r].signals || []).length, 0);

  // Hero metric: number of occupied rooms
  const heroValue = occupiedRooms.length;
  const heroLabel = heroValue === 1 ? '1 room occupied' : `${heroValue} rooms occupied`;

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="PRESENCE" subtitle="Room-level occupancy from cameras, motion sensors, and device signals fused through Bayesian probability." />

      {/* Connection status */}
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2 text-xs">
          <span class="inline-block w-2.5 h-2.5 rounded-full" style={`background: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`} />
          <span style={`color: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`}>
            MQTT {mqttConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div class="text-xs" style="color: var(--text-tertiary)">
          {roomNames.length} room{roomNames.length !== 1 ? 's' : ''} tracked \u00B7 {totalSignals} signal{totalSignals !== 1 ? 's' : ''} \u00B7 {Object.keys(cameraRooms).length} camera{Object.keys(cameraRooms).length !== 1 ? 's' : ''}
        </div>
      </div>

      {!mqttConnected && (
        <Callout>MQTT is disconnected. Camera-based presence signals are unavailable. Check that Frigate and Mosquitto are running.</Callout>
      )}

      {/* Hero metric + Persons */}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Hero: Occupied rooms */}
        <div class="t-frame p-4" style={`border-left: 4px solid ${heroValue > 0 ? 'var(--status-healthy)' : 'var(--border-subtle)'}`}>
          <div class="text-4xl font-bold" style={`color: ${heroValue > 0 ? 'var(--status-healthy)' : 'var(--text-tertiary)'}`}>
            {heroValue}
          </div>
          <div class="text-sm mt-1" style="color: var(--text-tertiary)">{heroLabel}</div>
          {occupiedRooms.length > 0 && (
            <div class="text-xs mt-1" style="color: var(--text-secondary); text-transform: capitalize;">
              {occupiedRooms.map(r => r.replace(/_/g, ' ')).join(', ')}
            </div>
          )}
        </div>

        {/* Identified Persons */}
        <div class="lg:col-span-2 t-frame p-4" data-label="who's home">
          {personNames.length === 0 ? (
            <div>
              <p class="text-sm" style="color: var(--text-tertiary)">No identified persons yet.</p>
              <p class="text-xs mt-1" style="color: var(--text-tertiary)">Face recognition builds over time. Frigate collects unknown faces automatically — label them in the Frigate UI to enable identification.</p>
            </div>
          ) : (
            <div>
              {personNames.map(name => (
                <PersonRow key={name} name={name} info={persons[name]} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Room cards */}
      <Section title="Room Occupancy" subtitle="Per-room probability from fused sensor signals. Rooms above 50% are considered occupied." summary={`${roomNames.length} rooms`}>
        {roomNames.length === 0 ? (
          <Callout>No rooms detected yet. Rooms appear when cameras detect people or HA sensors report activity in mapped areas.</Callout>
        ) : (
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {roomNames
              .sort((a, b) => (rooms[b].probability || 0) - (rooms[a].probability || 0))
              .map(name => (
                <RoomCard key={name} name={name} room={rooms[name]} />
              ))}
          </div>
        )}
      </Section>

      {/* Signal Feed */}
      <Section title="Signal Feed" subtitle="Recent detection events across all rooms. Each signal decays over time based on its type." summary={`${totalSignals} active`} defaultOpen={totalSignals > 0}>
        <div class="t-frame p-4" data-label="signals">
          <SignalFeed rooms={rooms} />
        </div>
        {/* Legend */}
        <div class="flex flex-wrap items-center gap-3 mt-3" style="font-size: 10px; color: var(--text-tertiary);">
          {Object.entries(SIGNAL_LABELS).map(([type, label]) => (
            <div key={type} class="flex items-center gap-1">
              <span class="inline-block w-2 h-2 rounded-full" style={`background: ${SIGNAL_COLORS[type]}`} />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </Section>

      {/* Camera Status */}
      <Section title="Cameras" subtitle="Active camera-to-room mappings. Cameras publish person detections to MQTT via Frigate." summary={`${Object.keys(cameraRooms).length} mapped`} defaultOpen={false}>
        <div class="t-frame p-4" data-label="camera map">
          <CameraStrip cameraRooms={cameraRooms} />
        </div>
      </Section>
    </div>
  );
}
