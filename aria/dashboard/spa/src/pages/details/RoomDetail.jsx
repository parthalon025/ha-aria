/**
 * RoomDetail — Detail renderer for room presence data.
 * Three-section layout: Summary (occupancy), Explanation (signals + persons), History (detections).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import HeroCard from '../../components/HeroCard.jsx';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime, confidenceColor } from '../intelligence/utils.jsx';

/** Staleness threshold: 30 minutes in ms. */
const STALE_MS = 30 * 60 * 1000;

function formatPct(val) {
  if (val === null || val === undefined) return '\u2014';
  return `${(val * 100).toFixed(0)}%`;
}

export default function RoomDetail({ id, type: _type }) {
  const [room, setRoom] = useState(null);
  const [detections, setDetections] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetchJson('/api/cache/presence')
      .then((result) => {
        const data = result?.data || result || {};
        const rooms = data.rooms || {};
        const roomData = rooms[id] || null;
        setRoom(roomData);
        setLastUpdated(result?.last_updated || data.last_updated || null);

        // Filter recent detections to this room
        const allDetections = data.recent_detections || [];
        const roomDetections = allDetections.filter(
          (det) => det.room === id || det.room_id === id
        );
        setDetections(roomDetections);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!room) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No room data found for: {id}
        </p>
      </div>
    );
  }

  const probability = room.occupancy_probability ?? room.probability ?? null;
  const confidence = room.confidence || 'low';
  const signals = room.signals || room.active_signals || [];
  const persons = room.persons || room.identified_persons || [];
  const isOccupied = room.occupied || (probability !== null && probability !== undefined && probability > 0.5);
  const isStale = lastUpdated && (Date.now() - new Date(lastUpdated).getTime()) > STALE_MS;

  const statsItems = [
    { label: 'Room', value: id },
    { label: 'Signals', value: String(signals.length) },
    { label: 'Occupied', value: isOccupied ? 'Yes' : 'No', warning: !isOccupied },
  ];

  return (
    <div class={`space-y-6 ${isStale ? 'sh-frozen' : ''}`}>
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <HeroCard
          value={formatPct(probability)}
          label="Occupancy"
          timestamp={lastUpdated}
        />
        <div style="margin-top: 12px;">
          <span
            style={`display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); ${confidenceColor(confidence)}`}
          >
            {confidence} confidence
          </span>
        </div>
        <div style="margin-top: 12px;">
          <StatsGrid items={statsItems} />
        </div>
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {/* Signal breakdown */}
        {signals.length > 0 ? (
          <div class="space-y-2" style="margin-bottom: 16px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Signal Breakdown
            </span>
            {signals.map((sig, idx) => {
              const sigValue = sig.value ?? sig.weight ?? sig.confidence ?? 0;
              const pct = (sigValue * 100).toFixed(0);
              return (
                <div key={idx} style="font-family: var(--font-mono); font-size: var(--type-label);">
                  <div class="flex items-center gap-2">
                    <span style="color: var(--text-secondary); min-width: 100px; flex-shrink: 0;">
                      {sig.type || sig.signal_type || 'signal'}
                    </span>
                    <div style="flex: 1; height: 6px; background: var(--bg-inset); border-radius: 3px; overflow: hidden;">
                      <div style={`width: ${pct}%; height: 100%; background: var(--accent); border-radius: 3px;`} />
                    </div>
                    <span style="color: var(--text-secondary); min-width: 40px; text-align: right;">{pct}%</span>
                  </div>
                  {sig.detail && (
                    <div style="color: var(--text-tertiary); font-size: var(--type-label); margin-left: 100px; margin-top: 2px;">
                      {sig.detail}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 16px;">
            No active signals
          </p>
        )}

        {/* Identified persons */}
        {persons.length > 0 && (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Persons Detected
            </span>
            {persons.map((person, idx) => (
              <div key={idx} class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-secondary);">{person.name || person.person_id || 'Unknown'}</span>
                <span style="color: var(--text-tertiary);">
                  {person.last_seen ? relativeTime(person.last_seen) : ''}
                  {person.confidence !== null && person.confidence !== undefined && ` (${(person.confidence * 100).toFixed(0)}%)`}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {detections.length > 0 ? (
          <div class="space-y-1">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Recent Detections
            </span>
            {detections.map((det, idx) => (
              <div key={idx} class="flex gap-3" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(det.timestamp || det.time)}
                </span>
                <span style="color: var(--text-secondary);">
                  {det.signal_type || det.type || 'detection'}
                </span>
                <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                  {det.detail || ''}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No recent detections for this room
          </p>
        )}
      </div>
    </div>
  );
}
