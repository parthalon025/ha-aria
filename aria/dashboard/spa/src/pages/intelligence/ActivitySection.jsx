import { Section, Callout, durationSince, describeEvent, EVENT_ICONS, DOMAIN_LABELS } from './utils.jsx';

function ActivityTimeline({ windows }) {
  if (!windows || windows.length === 0) return null;

  const now = new Date();
  const sixHoursAgo = new Date(now - 6 * 60 * 60 * 1000).toISOString();
  const recent = windows.filter(w => w.window_start >= sixHoursAgo);

  if (recent.length === 0) return null;

  const maxCount = Math.max(...recent.map(w => w.event_count), 1);
  const totalEvents = recent.reduce((sum, w) => sum + w.event_count, 0);
  const occupiedWindows = recent.filter(w => w.occupancy).length;
  const pctOccupied = recent.length > 0 ? Math.round((occupiedWindows / recent.length) * 100) : 0;

  return (
    <div class="space-y-2">
      <div class="flex justify-between items-baseline">
        <div class="text-xs font-medium text-gray-600">Activity Timeline (6h)</div>
        <div class="text-xs text-gray-400">{totalEvents} events, home {pctOccupied}% of the time</div>
      </div>
      <div class="flex items-end gap-0.5 h-16">
        {recent.map((w, i) => {
          const height = Math.max((w.event_count / maxCount) * 100, 4);
          const color = w.occupancy ? '#7c3aed' : '#9ca3af';
          const time = w.window_start.slice(11, 16);
          return (
            <div
              key={i}
              class="flex-1 rounded-t transition-all"
              style={{ height: `${height}%`, backgroundColor: color, minWidth: '3px' }}
              title={`${time} \u2014 ${w.event_count} events${w.occupancy ? '' : ' (away)'}`}
            />
          );
        })}
      </div>
      <div class="flex justify-between text-[10px] text-gray-400">
        <span>{recent[0]?.window_start?.slice(11, 16)}</span>
        <div class="flex items-center gap-2">
          <span class="inline-block w-2 h-2 rounded-sm" style={{ backgroundColor: '#7c3aed' }} /> home
          <span class="inline-block w-2 h-2 rounded-sm" style={{ backgroundColor: '#9ca3af' }} /> away
        </div>
        {recent.length > 1 && <span>{recent[recent.length - 1]?.window_start?.slice(11, 16)}</span>}
      </div>
    </div>
  );
}

function WsHealthIndicator({ ws }) {
  if (!ws) return null;
  const connected = ws.connected;
  return (
    <div class="flex items-center gap-2 text-xs">
      <span class={`inline-block w-2.5 h-2.5 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
      <span class={connected ? 'text-green-700' : 'text-red-700'}>
        {connected ? 'Connected' : 'Disconnected'}
      </span>
      {ws.disconnect_count > 0 && (
        <span class="text-gray-400">({ws.disconnect_count} reconnect{ws.disconnect_count !== 1 ? 's' : ''})</span>
      )}
      {ws.total_disconnect_s > 0 && (
        <span class="text-gray-400">{Math.round(ws.total_disconnect_s)}s downtime</span>
      )}
    </div>
  );
}

export function ActivitySection({ activity }) {
  if (!activity) {
    return (
      <Section
        title="Activity Monitor"
        subtitle="Waiting for WebSocket to connect to Home Assistant..."
      >
        <Callout color="gray">Activity monitoring is starting up. State changes will appear here once the WebSocket connection is established.</Callout>
      </Section>
    );
  }

  const summary = activity.activity_summary;
  const log = activity.activity_log;

  if (!summary && !log) {
    return (
      <Section
        title="Activity Monitor"
        subtitle="Waiting for first events..."
      >
        <Callout color="gray">Activity monitoring is starting up. State changes will appear here once the WebSocket connection is established.</Callout>
      </Section>
    );
  }

  const occ = summary?.occupancy || {};
  const rate = summary?.activity_rate || {};
  const snap = summary?.snapshot_status || {};
  const domains = summary?.domains_active_1h || {};
  const recentEvents = summary?.recent_activity || [];
  const windows = log?.windows || [];
  const ws = summary?.websocket || {};
  const eventPredictions = summary?.event_predictions || {};
  const patterns = summary?.patterns || [];
  const occPrediction = summary?.occupancy_prediction || {};
  const anomalies = summary?.anomalies || [];

  // Build contextual subtitle
  const parts = [];
  if (occ.anyone_home) {
    const dur = durationSince(occ.since);
    parts.push(dur ? `${occ.people.join(' & ')} home for ${dur}` : `${occ.people.join(' & ')} home`);
  } else {
    parts.push('Nobody home');
  }
  if (rate.trend === 'increasing') parts.push('activity picking up');
  else if (rate.trend === 'decreasing') parts.push('quieting down');
  const eventsToday = log?.events_today;
  if (eventsToday != null) parts.push(`${eventsToday} events today`);
  const contextSubtitle = parts.join(' \u00B7 ');

  // Determine snapshot status message
  let snapMsg = '';
  if (snap.today_count > 0) {
    snapMsg = `${snap.today_count} adaptive snapshot${snap.today_count !== 1 ? 's' : ''} captured today \u2014 the system saw enough activity to grab extra data points.`;
  } else if (occ.anyone_home) {
    snapMsg = 'Watching for sustained activity to trigger an adaptive snapshot (needs 5+ events with 30m cooldown).';
  } else {
    snapMsg = 'Adaptive snapshots only trigger when someone is home.';
  }

  return (
    <Section title="Activity Monitor" subtitle={contextSubtitle}>
      <div class="space-y-4">

        {/* WebSocket health + anomaly alerts */}
        <div class="flex items-center justify-between">
          <WsHealthIndicator ws={ws} />
          {eventPredictions.predicted_next_domain && (
            <div class="text-xs text-gray-400">
              Next likely: <span class="font-medium text-gray-600">{DOMAIN_LABELS[eventPredictions.predicted_next_domain] || eventPredictions.predicted_next_domain}</span>
              {' '}({Math.round(eventPredictions.probability * 100)}%)
            </div>
          )}
        </div>

        {/* Anomaly callouts */}
        {anomalies.length > 0 && anomalies.map((a, i) => (
          <Callout key={i} color="amber">{a.message}</Callout>
        ))}

        {/* Status bar */}
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Occupancy */}
          <div class={`bg-white rounded-lg shadow-sm p-4 ${occ.anyone_home ? 'border-l-4 border-green-400' : 'border-l-4 border-gray-200'}`}>
            <div class={`text-2xl font-bold ${occ.anyone_home ? 'text-green-600' : 'text-gray-400'}`}>
              {occ.anyone_home ? 'Home' : 'Away'}
            </div>
            <div class="text-sm text-gray-500 mt-1">Occupancy</div>
            {occ.anyone_home && occ.since && (
              <div class="text-xs text-gray-400 mt-1">for {durationSince(occ.since)}</div>
            )}
            {!occ.anyone_home && occPrediction.predicted_arrival && occPrediction.status === 'predicted' && (
              <div class="text-xs text-blue-500 mt-1">
                ETA ~{occPrediction.predicted_arrival}
                <span class="text-gray-400"> ({occPrediction.confidence}, {occPrediction.based_on} samples)</span>
              </div>
            )}
            {!occ.anyone_home && occPrediction.status === 'past_predicted' && (
              <div class="text-xs text-gray-400 mt-1">
                Typical arrival ({occPrediction.predicted_arrival}) passed
              </div>
            )}
          </div>

          {/* Current window */}
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="text-2xl font-bold text-blue-500">
              {rate.current != null ? rate.current : '\u2014'}
            </div>
            <div class="text-sm text-gray-500 mt-1">Events (15m window)</div>
            {rate.avg_1h > 0 && (
              <div class={`text-xs mt-1 ${
                rate.trend === 'increasing' ? 'text-amber-500' :
                rate.trend === 'decreasing' ? 'text-blue-500' : 'text-gray-400'
              }`}>
                {rate.trend === 'increasing' ? 'Above' : rate.trend === 'decreasing' ? 'Below' : 'Near'} avg ({rate.avg_1h}/window)
              </div>
            )}
          </div>

          {/* Today total */}
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="text-2xl font-bold text-blue-500">
              {eventsToday != null ? eventsToday : '\u2014'}
            </div>
            <div class="text-sm text-gray-500 mt-1">Events Today</div>
          </div>

          {/* Snapshots */}
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="text-2xl font-bold text-blue-500">
              {snap.today_count != null ? `${snap.today_count}/${snap.daily_cap}` : '\u2014'}
            </div>
            <div class="text-sm text-gray-500 mt-1">Adaptive Snapshots</div>
            {snap.cooldown_remaining_s > 0 && (
              <div class="text-xs text-gray-400 mt-1">Next eligible in {Math.ceil(snap.cooldown_remaining_s / 60)}m</div>
            )}
          </div>
        </div>

        {/* Snapshot context */}
        <div class="text-xs text-gray-400 italic px-1">{snapMsg}</div>

        {/* Recent Activity + Domains side by side */}
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">

          {/* Recent Activity -- 2/3 width */}
          <div class="md:col-span-2 bg-white rounded-lg shadow-sm p-4">
            <div class="text-xs font-bold text-gray-500 uppercase mb-2">What Just Happened</div>
            {recentEvents.length === 0 ? (
              <p class="text-sm text-gray-400">Waiting for state changes...</p>
            ) : (
              <div class="space-y-0.5 max-h-64 overflow-y-auto">
                {recentEvents.map((evt, i) => {
                  const desc = describeEvent(evt);
                  const icon = EVENT_ICONS[desc.icon] || '\u00B7';
                  const isSignificant = ['lock', 'person', 'device_tracker'].includes(evt.domain)
                    || (evt.domain === 'binary_sensor' && ['door', 'window'].includes(evt.device_class));
                  const evtKey = `${evt.entity || evt.friendly_name}-${evt.time}-${i}`;
                  return (
                    <div key={evtKey} class={`flex items-center gap-2 py-1 px-1 rounded ${isSignificant ? 'bg-amber-50' : ''}`}>
                      <span class="w-5 text-center text-sm flex-shrink-0">{icon}</span>
                      <span class={`flex-1 text-sm ${isSignificant ? 'font-medium text-gray-900' : 'text-gray-600'}`}>
                        {desc.text}
                      </span>
                      <span class="text-xs text-gray-400 flex-shrink-0">{evt.time}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Domain breakdown -- 1/3 width */}
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="text-xs font-bold text-gray-500 uppercase mb-2">Active Domains (1h)</div>
            {Object.keys(domains).length === 0 ? (
              <p class="text-sm text-gray-400">No activity yet.</p>
            ) : (
              <div class="space-y-2">
                {(() => {
                  const maxDomain = Math.max(...Object.values(domains), 1);
                  return Object.entries(domains).map(([domain, count]) => {
                    const label = DOMAIN_LABELS[domain] || domain;
                    const pct = (count / maxDomain) * 100;
                    return (
                      <div key={domain} class="space-y-0.5">
                        <div class="flex justify-between text-xs">
                          <span class="text-gray-600">{label}</span>
                          <span class="text-gray-400">{count}</span>
                        </div>
                        <div class="h-1.5 bg-gray-100 rounded-full">
                          <div class="h-1.5 bg-purple-400 rounded-full" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            )}
          </div>
        </div>

        {/* Detected Patterns */}
        {patterns.length > 0 && (
          <div class="bg-white rounded-lg shadow-sm p-4">
            <div class="text-xs font-bold text-gray-500 uppercase mb-2">Detected Patterns (24h)</div>
            <p class="text-xs text-gray-400 mb-2">Recurring event sequences that suggest routines or automations.</p>
            <div class="space-y-1.5">
              {patterns.map((p, i) => (
                <div key={i} class="flex items-center gap-2 text-sm">
                  <div class="flex items-center gap-1">
                    {p.sequence.map((s, j) => (
                      <span key={j}>
                        <span class="bg-purple-100 text-purple-700 rounded px-1.5 py-0.5 text-xs font-medium">
                          {DOMAIN_LABELS[s] || s}
                        </span>
                        {j < p.sequence.length - 1 && <span class="text-gray-300 mx-0.5">&rarr;</span>}
                      </span>
                    ))}
                  </div>
                  <span class="text-xs text-gray-400 ml-auto flex-shrink-0">{p.count}x{p.last_seen ? `, last ${p.last_seen.slice(11, 16) || p.last_seen}` : ''}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timeline */}
        <div class="bg-white rounded-lg shadow-sm p-4">
          <ActivityTimeline windows={windows} />
          {windows.length === 0 && (
            <p class="text-sm text-gray-400">Timeline will appear after the first 15-minute window.</p>
          )}
        </div>

        {/* Snapshot Log */}
        {snap.log_today && snap.log_today.length > 0 && (
          <details class="bg-white rounded-lg shadow-sm">
            <summary class="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700 hover:bg-gray-50">
              Snapshot Log \u2014 {snap.log_today.length} adaptive snapshot{snap.log_today.length !== 1 ? 's' : ''} today
            </summary>
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-100 text-left text-xs text-gray-500">
                    <th class="px-4 py-1">#</th>
                    <th class="px-4 py-1">Time</th>
                    <th class="px-4 py-1">Events</th>
                    <th class="px-4 py-1">People</th>
                    <th class="px-4 py-1">What Triggered It</th>
                  </tr>
                </thead>
                <tbody>
                  {snap.log_today.map((entry, i) => (
                    <tr key={i} class="border-b border-gray-50">
                      <td class="px-4 py-1.5 text-gray-400">{entry.number}</td>
                      <td class="px-4 py-1.5 text-gray-600">{entry.timestamp?.slice(11, 16)}</td>
                      <td class="px-4 py-1.5">{entry.buffered_events} buffered</td>
                      <td class="px-4 py-1.5 text-gray-600">{(entry.people || []).join(', ') || '\u2014'}</td>
                      <td class="px-4 py-1.5">
                        <div class="flex flex-wrap gap-1">
                          {Object.entries(entry.domains || {}).slice(0, 4).map(([d, c]) => (
                            <span key={d} class="bg-gray-100 rounded px-1.5 py-0.5 text-xs">
                              {DOMAIN_LABELS[d] || d}: {c}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </div>
    </Section>
  );
}
