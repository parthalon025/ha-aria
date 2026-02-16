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

/** Identified person row with camera and room info */
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

/** Recent person detection card with thumbnail from any camera */
function DetectionCard({ detection }) {
  const hasThumbnail = detection.event_id && detection.has_snapshot;
  const thumbnailUrl = hasThumbnail ? `/api/frigate/thumbnail/${detection.event_id}` : null;

  return (
    <div class="t-frame p-2 flex gap-3" style="min-width: 0;">
      {/* Thumbnail or placeholder */}
      <div class="flex-shrink-0 rounded overflow-hidden" style="width: 64px; height: 64px; background: var(--bg-inset);">
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt={`Detection on ${detection.camera}`}
            style="width: 100%; height: 100%; object-fit: cover;"
            loading="lazy"
          />
        ) : (
          <div class="flex items-center justify-center w-full h-full" style="color: var(--text-tertiary); font-size: 20px;">
            {'\u{1F464}'}
          </div>
        )}
      </div>
      {/* Detail */}
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          {detection.sub_label ? (
            <span class="text-sm font-bold" style="color: var(--status-healthy); text-transform: capitalize;">{detection.sub_label}</span>
          ) : (
            <span class="text-sm font-medium" style="color: var(--text-secondary)">Unknown person</span>
          )}
          <span class="text-xs rounded px-1" style="background: var(--bg-surface-raised); color: var(--text-tertiary);">
            {Math.round((detection.score || 0) * 100)}%
          </span>
        </div>
        <div class="text-xs mt-0.5" style="color: var(--text-tertiary); text-transform: capitalize;">
          {detection.camera.replace(/_/g, ' ')} \u2192 {detection.room.replace(/_/g, ' ')}
        </div>
        {detection.timestamp && (
          <div class="text-xs mt-0.5" style="color: var(--text-tertiary)">{relativeTime(detection.timestamp)}</div>
        )}
      </div>
    </div>
  );
}

/** Face Recognition status and configuration display */
function FaceRecognitionStatus({ faceConfig, labeledFaces }) {
  const enabled = faceConfig && faceConfig.enabled;
  const config = (faceConfig && faceConfig.config) || {};
  const labeledCount = faceConfig ? faceConfig.labeled_count || 0 : 0;
  const labeledNames = labeledFaces || {};

  return (
    <div class="space-y-3">
      {/* Status indicators */}
      <div class="flex flex-wrap gap-3">
        <div class="t-frame p-3 flex-1" style="min-width: 120px;">
          <div class="text-xs" style="color: var(--text-tertiary)">Status</div>
          <div class="flex items-center gap-1.5 mt-1">
            <span class="inline-block w-2 h-2 rounded-full" style={`background: ${enabled ? 'var(--status-healthy)' : 'var(--status-error)'}`} />
            <span class="text-sm font-medium" style={`color: ${enabled ? 'var(--status-healthy)' : 'var(--status-error)'}`}>
              {enabled ? 'Active' : 'Disabled'}
            </span>
          </div>
        </div>
        <div class="t-frame p-3 flex-1" style="min-width: 120px;">
          <div class="text-xs" style="color: var(--text-tertiary)">Labeled Faces</div>
          <div class="text-xl font-bold mt-1" style={`color: ${labeledCount > 0 ? 'var(--accent)' : 'var(--text-tertiary)'}`}>
            {labeledCount}
          </div>
        </div>
        <div class="t-frame p-3 flex-1" style="min-width: 120px;">
          <div class="text-xs" style="color: var(--text-tertiary)">Threshold</div>
          <div class="text-sm font-bold mt-1" style="color: var(--text-primary)">
            {config.recognition_threshold ? `${Math.round(config.recognition_threshold * 100)}%` : '--'}
          </div>
        </div>
        <div class="t-frame p-3 flex-1" style="min-width: 120px;">
          <div class="text-xs" style="color: var(--text-tertiary)">Model</div>
          <div class="text-sm font-medium mt-1" style="color: var(--text-primary)">
            {config.model_size || '--'}
          </div>
        </div>
      </div>

      {/* Labeled persons list */}
      {Object.keys(labeledNames).length > 0 && (
        <div>
          <div class="text-xs font-medium mb-1.5" style="color: var(--text-secondary)">Known Faces</div>
          <div class="flex flex-wrap gap-2">
            {Object.entries(labeledNames).map(([name, count]) => (
              <span key={name} class="text-xs rounded px-2 py-1" style="background: var(--accent-glow); color: var(--accent); text-transform: capitalize;">
                {name} ({count} sample{count !== 1 ? 's' : ''})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Labeling guide when no faces labeled */}
      {labeledCount === 0 && enabled && (
        <Callout>
          No faces labeled yet. Frigate automatically collects face samples from person detections.
          To enable face recognition: open Frigate UI, go to a person event, and assign a name to the detected face.
          After labeling {config.min_faces || 2}+ samples per person, ARIA will identify them automatically across all cameras.
        </Callout>
      )}
    </div>
  );
}

export default function Presence() {
  const { data, loading, error, refetch } = useCache('presence');

  if (loading && !data) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  // data = API envelope { category, data: { ... }, version, ... }
  const presence = (data && data.data) || {};
  const rooms = presence.rooms || {};
  const occupiedRooms = presence.occupied_rooms || [];
  const persons = presence.identified_persons || {};
  const cameraRooms = presence.camera_rooms || {};
  const mqttConnected = presence.mqtt_connected;
  const faceRecognition = presence.face_recognition || {};
  const recentDetections = presence.recent_detections || [];
  const roomNames = Object.keys(rooms);
  const personNames = Object.keys(persons);
  const totalSignals = roomNames.reduce((sum, r) => sum + (rooms[r].signals || []).length, 0);
  const cameraCount = Object.keys(cameraRooms).length;

  // Hero metric: number of occupied rooms
  const heroValue = occupiedRooms.length;
  const heroLabel = heroValue === 1 ? '1 room occupied' : `${heroValue} rooms occupied`;

  // Count unique cameras with recent detections
  const activeCameras = new Set(recentDetections.map(d => d.camera));

  return (
    <div class="space-y-6 animate-page-enter">
      <PageBanner page="PRESENCE" subtitle="Room-level occupancy from cameras, motion sensors, and device signals fused through Bayesian probability." />

      {/* Connection status */}
      <div class="flex items-center gap-3 flex-wrap">
        <div class="flex items-center gap-2 text-xs">
          <span class="inline-block w-2.5 h-2.5 rounded-full" style={`background: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`} />
          <span style={`color: ${mqttConnected ? 'var(--status-healthy)' : 'var(--status-error)'}`}>
            MQTT {mqttConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div class="text-xs" style="color: var(--text-tertiary)">
          {roomNames.length} room{roomNames.length !== 1 ? 's' : ''} \u00B7 {totalSignals} signal{totalSignals !== 1 ? 's' : ''} \u00B7 {cameraCount} camera{cameraCount !== 1 ? 's' : ''}
          {activeCameras.size > 0 && ` \u00B7 ${activeCameras.size} detecting`}
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
              <p class="text-xs mt-1" style="color: var(--text-tertiary)">
                Face recognition builds over time. Frigate collects unknown faces automatically
                {faceRecognition.labeled_count > 0 ? ` \u2014 ${faceRecognition.labeled_count} face${faceRecognition.labeled_count !== 1 ? 's' : ''} labeled so far.` : ' \u2014 label them in the Frigate UI to enable identification.'}
              </p>
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

      {/* Recent Detections — cross-camera person sightings with thumbnails */}
      <Section title="Recent Detections" subtitle="Person detections across all cameras, newest first. Thumbnails from Frigate event snapshots." summary={`${recentDetections.length} events`} defaultOpen={recentDetections.length > 0}>
        {recentDetections.length === 0 ? (
          <Callout>No person detections yet. Walk past a camera to trigger a detection event.</Callout>
        ) : (
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {recentDetections.slice(0, 12).map((det, i) => (
              <DetectionCard key={det.event_id || i} detection={det} />
            ))}
          </div>
        )}
      </Section>

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

      {/* Face Recognition */}
      <Section title="Face Recognition" subtitle="Frigate collects face samples from person detections. Label faces in the Frigate UI to enable cross-camera identification." summary={faceRecognition.enabled ? `${faceRecognition.labeled_count || 0} labeled` : 'disabled'} defaultOpen={false}>
        <FaceRecognitionStatus faceConfig={faceRecognition} labeledFaces={faceRecognition.labeled_faces || {}} />
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
      <Section title="Cameras" subtitle="Active camera-to-room mappings. Cameras publish person detections to MQTT via Frigate." summary={`${cameraCount} mapped`} defaultOpen={false}>
        <div class="t-frame p-4" data-label="camera map">
          <CameraStrip cameraRooms={cameraRooms} />
        </div>
      </Section>
    </div>
  );
}
