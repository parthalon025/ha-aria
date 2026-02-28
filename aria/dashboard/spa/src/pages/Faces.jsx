import { useState, useEffect, useRef } from 'preact/hooks';
import { fetchJson, postJson, baseUrl } from '../api.js';
import PageBanner from '../components/PageBanner.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

function formatAgo(isoStr) {
  if (!isoStr) return 'Never';
  const diff = Math.floor((Date.now() - new Date(isoStr)) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

/** Convert internal cluster labels to human-readable display names. */
function formatCandidate(name) {
  if (!name || name === 'unknown') return 'Unidentified';
  if (name.startsWith('cluster_')) return `Group ${name.slice(8)}`;
  return name;
}

/**
 * Color tier for a confidence score (Cleveland & McGill: length + color encoding).
 * ≥85% = green (high confidence), 50-84% = amber (review needed), <50% = muted.
 */
function confColor(confidence) {
  if (confidence >= 0.85) return 'var(--status-healthy)';
  if (confidence >= 0.50) return 'var(--status-active)';
  return 'var(--text-tertiary)';
}

/** DELETE helper — no deleteJson exists in api.js so we wrap fetch directly. */
function deleteJson(path) {
  return fetch(`${baseUrl}${path}`, { method: 'DELETE' })
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      return res.json();
    });
}

/** Source badge color — bootstrap=muted, manual=accent, live=green */
function sourceBadgeColor(source) {
  if (source === 'live') return 'var(--status-healthy)';
  if (source === 'manual') return 'var(--accent)';
  return 'var(--text-tertiary)';
}

export default function Faces() {
  const [stats, setStats] = useState(null);
  const [queue, setQueue] = useState([]);
  const [people, setPeople] = useState([]);
  const [labelInput, setLabelInput] = useState({});
  const [labeledItems, setLabeledItems] = useState({});
  const [labelingItems, setLabelingItems] = useState({});
  const [bootstrapStatus, setBootstrapStatus] = useState({
    running: false, processed: 0, total: 0, startedAt: null, lastRan: null,
  });
  const [deployMsg, setDeployMsg] = useState(null);
  const [deployed, setDeployed] = useState(false);
  const [deployedPeopleCount, setDeployedPeopleCount] = useState(0);
  const [deployedSampleCount, setDeployedSampleCount] = useState(0);
  const [frigateRestarting, setFrigateRestarting] = useState(false);
  const [frigateRestarted, setFrigateRestarted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Pagination: fetch up to 100, display in chunks of 20
  const [displayLimit, setDisplayLimit] = useState(20);
  // Clear queue confirmation
  const [confirmClear, setConfirmClear] = useState(false);
  // Keyboard navigation: ID of the currently focused queue item
  const [focusedItemId, setFocusedItemId] = useState(null);
  // Per-cluster-group name input for bulk labeling (key = "Group N", value = typed name)
  const [clusterGroupInput, setClusterGroupInput] = useState({});

  // Person management state
  const [expandedPerson, setExpandedPerson] = useState(null);
  const [personSamples, setPersonSamples] = useState({});
  const [renamingPerson, setRenamingPerson] = useState(null);
  const [renameInput, setRenameInput] = useState('');
  const [deletingPerson, setDeletingPerson] = useState(null);
  const [deletingSample, setDeletingSample] = useState(null);
  const [exporting, setExporting] = useState(false);

  // Presence data for "last seen" in person cards
  const [presencePersons, setPresencePersons] = useState({});

  // Ref for the auto-refresh interval
  const queuePollRef = useRef(null);

  async function fetchData() {
    try {
      const [s, q, p] = await Promise.all([
        fetchJson('/api/faces/stats'),
        fetchJson('/api/faces/queue?limit=100'),
        fetchJson('/api/faces/people'),
      ]);
      setStats(s);
      setQueue(q.items || []);
      setPeople(p.people || []);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  // Fetch presence data on mount (best-effort — ignore failures)
  useEffect(() => {
    const controller = new AbortController();
    fetchJson('/api/cache/presence')
      .then(d => {
        if (controller.signal.aborted) return;
        const presence = d?.data?.data || d?.data || {};
        setPresencePersons(presence.identified_persons || {});
      })
      .catch(e => { if (e.name !== 'AbortError') {} }); // best-effort, suppress all
    return () => controller.abort();
  }, []);

  // Fetch initial data and bootstrap status on mount
  useEffect(() => {
    const controller = new AbortController();
    fetchData();
    fetchJson('/api/faces/bootstrap/status')
      .then(s => {
        if (controller.signal.aborted) return;
        setBootstrapStatus({
          running: s.running, processed: s.processed, total: s.total,
          startedAt: s.started_at, lastRan: s.last_ran,
        });
      })
      .catch(e => { if (e.name !== 'AbortError') console.warn('Bootstrap status fetch failed:', e); });
    return () => controller.abort();
  }, []);

  // Auto-focus first queue item when queue first loads
  useEffect(() => {
    if (queue.length > 0 && focusedItemId === null) {
      setFocusedItemId(queue[0].id);
    }
  }, [queue.length]);

  // Poll bootstrap status every 2s while running; refresh data on completion
  useEffect(() => {
    if (!bootstrapStatus.running) return;
    const id = setInterval(() => {
      fetchJson('/api/faces/bootstrap/status')
        .then(s => {
          setBootstrapStatus({
            running: s.running, processed: s.processed, total: s.total,
            startedAt: s.started_at, lastRan: s.last_ran,
          });
          if (!s.running) fetchData();
        })
        .catch(e => console.warn('Bootstrap status poll failed:', e));
    }, 2000);
    return () => clearInterval(id);
  }, [bootstrapStatus.running]);

  // Auto-refresh queue every 5s when queue has items OR bootstrap is running
  useEffect(() => {
    const shouldPoll = queue.length > 0 || bootstrapStatus.running;
    if (queuePollRef.current) {
      clearInterval(queuePollRef.current);
      queuePollRef.current = null;
    }
    if (shouldPoll) {
      queuePollRef.current = setInterval(() => {
        fetchData();
      }, 5000);
    }
    return () => {
      if (queuePollRef.current) {
        clearInterval(queuePollRef.current);
        queuePollRef.current = null;
      }
    };
  }, [queue.length, bootstrapStatus.running]);

  // Keyboard shortcuts: j/k navigate, 1/2/3 select candidates, Enter submits
  useEffect(() => {
    function onKey(evt) {
      // Never capture while user is typing in an input field
      if (document.activeElement?.tagName === 'INPUT') return;
      const visible = queue.slice(0, displayLimit);
      const idx = visible.findIndex(item => item.id === focusedItemId);

      if (evt.key === 'j' || evt.key === 'ArrowDown') {
        evt.preventDefault();
        if (idx < visible.length - 1) setFocusedItemId(visible[idx + 1].id);
        else if (displayLimit < queue.length) setDisplayLimit(prev => prev + 20);
      } else if (evt.key === 'k' || evt.key === 'ArrowUp') {
        evt.preventDefault();
        if (idx > 0) setFocusedItemId(visible[idx - 1].id);
      } else if (['1', '2', '3'].includes(evt.key) && focusedItemId !== null) {
        evt.preventDefault();
        const item = visible[idx];
        if (!item) return;
        const candIndex = parseInt(evt.key, 10) - 1;
        const visibleCands = item.top_candidates?.filter(c => c.confidence >= 0.30) || [];
        const picked = visibleCands[candIndex];
        if (picked) handleLabel(item.id, formatCandidate(picked.person_name));
      } else if (evt.key === 'Enter' && focusedItemId !== null) {
        evt.preventDefault();
        handleLabel(focusedItemId);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [queue, focusedItemId, displayLimit, labelInput]);

  /**
   * Label a queue item. Accepts an optional overrideName so candidate chips
   * can submit directly without going through the text input state.
   * Pass skipRefresh=true to suppress the per-item fetchData (for bulk ops).
   */
  async function handleLabel(queueId, overrideName, { skipRefresh = false } = {}) {
    const name = (overrideName ?? labelInput[queueId])?.trim();
    if (!name || labelingItems[queueId]) return;
    setLabelingItems(prev => ({ ...prev, [queueId]: true }));
    try {
      await postJson('/api/faces/label', { queue_id: queueId, person_name: name });
      setLabeledItems(prev => ({ ...prev, [queueId]: name }));
      setLabelInput(prev => ({ ...prev, [queueId]: '' }));
      // Advance focus to next item after labeling
      setFocusedItemId(prev => {
        const visible = queue.slice(0, displayLimit);
        const idx = visible.findIndex(item => item.id === queueId);
        return visible[idx + 1]?.id ?? prev;
      });
      if (!skipRefresh) {
        setTimeout(() => {
          setLabeledItems(prev => { const n = { ...prev }; delete n[queueId]; return n; });
          fetchData();
        }, 1500);
      }
    } catch (e) {
      setError(e);
    } finally {
      setLabelingItems(prev => { const n = { ...prev }; delete n[queueId]; return n; });
    }
  }

  /** Confirm all unlabeled items in a group — parallel requests, single refresh at end. */
  async function handleConfirmGroup(items, personName) {
    const unlabeled = items.filter(item => !labeledItems[item.id]);
    if (!unlabeled.length) return;
    // Mark all as labeling immediately for visual feedback
    setLabelingItems(prev => {
      const n = { ...prev };
      unlabeled.forEach(item => { n[item.id] = true; });
      return n;
    });
    let failCount = 0;
    try {
      // Fire all label requests in parallel (backend handles duplicates via UNIQUE constraint)
      await Promise.all(unlabeled.map(item =>
        postJson('/api/faces/label', { queue_id: item.id, person_name: personName })
          .then(() => setLabeledItems(prev => ({ ...prev, [item.id]: personName })))
          .catch(() => { failCount++; })
      ));
      // Surface error if all items in the batch failed (network or server issue)
      if (failCount === unlabeled.length) {
        setError(new Error(`Failed to label ${failCount} item${failCount > 1 ? 's' : ''} — check connection`));
      } else if (failCount > 0) {
        setError(new Error(`${failCount} of ${unlabeled.length} items failed to label — refreshing`));
      }
    } finally {
      setLabelingItems(prev => {
        const n = { ...prev };
        unlabeled.forEach(item => { delete n[item.id]; });
        return n;
      });
      // Single refresh for the whole group after all requests complete
      setTimeout(() => {
        setLabeledItems(prev => {
          const n = { ...prev };
          unlabeled.forEach(item => { delete n[item.id]; });
          return n;
        });
        fetchData();
      }, 1200);
    }
  }

  /** Dismiss a single queue item without labeling it. Optimistic update. */
  async function handleDismiss(itemId) {
    // Optimistic remove from queue
    setQueue(prev => prev.filter(item => item.id !== itemId));
    try {
      await postJson(`/api/faces/queue/${itemId}/dismiss`, {});
    } catch (e) {
      // On error, re-fetch to restore true state
      fetchData();
      setError(e);
    }
  }

  /** Clear queue — requires a confirmation click first. */
  async function handleClearQueue() {
    if (!confirmClear) {
      setConfirmClear(true);
      return;
    }
    setConfirmClear(false);
    try {
      await postJson('/api/faces/queue/clear', {});
      fetchData();
    } catch (e) {
      setError(e);
    }
  }

  async function handleBootstrap() {
    try {
      await postJson('/api/faces/bootstrap', {});
      setBootstrapStatus(prev => ({ ...prev, running: true }));
      // Reset all stale queue-side UI state from prior run
      setClusterGroupInput({});
      setLabeledItems({});
      setLabelInput({});
    } catch (e) {
      setError(e);
    }
  }

  async function handleDeploy() {
    try {
      await postJson('/api/faces/deploy', {});
      setDeployed(true);
      setDeployedPeopleCount(people.length);
      setDeployedSampleCount(people.reduce((s, p) => s + p.count, 0));
      setDeployMsg('Deployed — restarting Frigate…');
      await handleRestartFrigate();
    } catch (e) {
      setError(e);
    }
  }

  async function handleRestartFrigate() {
    setFrigateRestarting(true);
    setFrigateRestarted(false);
    try {
      await postJson('/api/faces/restart-frigate', {});
      setFrigateRestarted(true);
      setDeployMsg('Deployed and Frigate restarted — face recognition is live.');
      setTimeout(() => setDeployMsg(null), 8000);
    } catch {
      setDeployMsg('Deployed. Frigate restart failed — run: docker restart frigate');
      setTimeout(() => setDeployMsg(null), 10000);
    } finally {
      setFrigateRestarting(false);
    }
  }

  /** Toggle expand/collapse for a person panel. Fetches samples on expand. */
  async function handleTogglePerson(personName) {
    if (expandedPerson === personName) {
      setExpandedPerson(null);
      return;
    }
    setExpandedPerson(personName);
    if (!personSamples[personName]) {
      try {
        const data = await fetchJson(`/api/faces/people/${encodeURIComponent(personName)}/samples`);
        setPersonSamples(prev => ({ ...prev, [personName]: data.samples || [] }));
      } catch (e) {
        console.warn(`Failed to fetch samples for ${personName}:`, e);
        setPersonSamples(prev => ({ ...prev, [personName]: [] }));
      }
    }
  }

  /** Delete a single training sample. Optimistic update; re-fetch on success and error. */
  async function handleDeleteSample(personName, sampleId) {
    setPersonSamples(prev => ({
      ...prev,
      [personName]: (prev[personName] || []).filter(s => s.id !== sampleId),
    }));
    setDeletingSample(null);
    try {
      await deleteJson(`/api/faces/people/${encodeURIComponent(personName)}/samples/${sampleId}`);
      // Re-fetch to confirm server state matches optimistic update
      const fresh = await fetchJson(`/api/faces/people/${encodeURIComponent(personName)}/samples`);
      setPersonSamples(prev => ({ ...prev, [personName]: fresh.samples || [] }));
    } catch (e) {
      // Re-fetch to restore correct state (undo the optimistic update)
      try {
        const restored = await fetchJson(`/api/faces/people/${encodeURIComponent(personName)}/samples`);
        setPersonSamples(prev => ({ ...prev, [personName]: restored.samples || [] }));
      } catch { /* restore failed, keep previous optimistic state */ }
      setError(e);
    }
  }

  /** Rename a person. On success: refresh people list, collapse panel. */
  async function handleRename(personName) {
    const newName = renameInput.trim();
    if (!newName || newName === personName) {
      setRenamingPerson(null);
      setRenameInput('');
      return;
    }
    try {
      await postJson(`/api/faces/people/${encodeURIComponent(personName)}/rename`, { new_name: newName });
      setRenamingPerson(null);
      setRenameInput('');
      // Move expanded panel to new name, migrate cached samples
      setExpandedPerson(newName);
      setPersonSamples(prev => {
        const updated = { ...prev };
        if (updated[personName]) updated[newName] = updated[personName];
        delete updated[personName];
        return updated;
      });
      fetchData();
    } catch (e) {
      setError(e);
    }
  }

  /** Delete an entire person. Requires a confirm click. */
  async function handleDeletePerson(personName) {
    if (deletingPerson !== personName) {
      setDeletingPerson(personName);
      return;
    }
    // Second click — confirmed
    setDeletingPerson(null);
    try {
      await deleteJson(`/api/faces/people/${encodeURIComponent(personName)}`);
      setExpandedPerson(null);
      setPersonSamples(prev => {
        const updated = { ...prev };
        delete updated[personName];
        return updated;
      });
      fetchData();
    } catch (e) {
      setError(e);
    }
  }

  const pct = bootstrapStatus.total > 0
    ? Math.round(bootstrapStatus.processed / bootstrapStatus.total * 100)
    : 0;

  // Paginated visible slice of the queue
  const visibleQueue = queue.slice(0, displayLimit);
  const hasMore = queue.length > displayLimit || (stats?.queue_depth ?? 0) > queue.length;

  // Client-side grouping:
  //  - Real person names → own group (DBSCAN already matched to a known person)
  //  - cluster_* → own "Group N" group (visually similar unknowns — bulk-label together)
  //  - unknown / no candidates → "Unidentified"
  const queueGroups = visibleQueue.reduce((groups, item) => {
    const primaryName = item.top_candidates?.[0]?.person_name;
    let key;
    if (primaryName && primaryName !== 'unknown' && !primaryName.startsWith('cluster_')) {
      key = formatCandidate(primaryName);        // known person
    } else if (primaryName && primaryName.startsWith('cluster_')) {
      key = formatCandidate(primaryName);        // "Group N" — visually similar cluster
    } else {
      key = 'Unidentified';                      // no usable candidate
    }
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
    return groups;
  }, {});
  // Sort: real person groups first → cluster groups (Group N) → Unidentified last
  const groupKeys = Object.keys(queueGroups).sort((a, b) => {
    if (a === 'Unidentified') return 1;
    if (b === 'Unidentified') return -1;
    const aCluster = a.startsWith('Group ');
    const bCluster = b.startsWith('Group ');
    if (aCluster && !bCluster) return 1;
    if (!aCluster && bCluster) return -1;
    return a.localeCompare(b);
  });

  // Bootstrap is "done" if the in-memory tracker recorded a run, OR if there's
  // already data in the store (queue items or known people) — handles hub restarts
  // that wipe the in-memory lastRan field (Cluster C: cold-start state loss).
  const bootstrapDone = !!bootstrapStatus.lastRan
    || queue.length > 0
    || people.length > 0
    || (stats?.queue_depth ?? 0) > 0;
  const canDeploy = (stats?.known_people ?? 0) >= 1;

  // Show re-deploy banner when new labels exist since last deploy
  const totalSamples = people.reduce((s, p) => s + p.count, 0);
  const newLabelsSinceDeploy = deployed && (people.length > deployedPeopleCount || totalSamples > deployedSampleCount);

  if (loading) return <LoadingState type="stats" />;

  return (
    <div style="max-width: 56rem; margin: 0 auto;">
      <PageBanner page="FACES" subtitle="Face recognition — bootstrap clusters, label identities, deploy to Frigate." />

      {error && (
        <div style="margin-bottom: 1.5rem;">
          <ErrorState error={error} onRetry={fetchData} />
        </div>
      )}

      {/* Stats — 5 tiles */}
      <div class="grid gap-4" style="grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr)); margin-bottom: 1.5rem;">
        <div class="t-card" style="padding: 1rem; text-align: center;">
          <div style="font-size: 2rem; font-weight: 700; color: var(--status-active);">
            {stats?.queue_depth ?? 0}
          </div>
          <div style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 0.25rem;">
            Pending review
          </div>
        </div>
        <div class="t-card" style="padding: 1rem; text-align: center;">
          <div style="font-size: 2rem; font-weight: 700; color: var(--accent);">
            {stats?.auto_label_rate !== null && stats?.auto_label_rate !== undefined
              ? `${Math.round(stats.auto_label_rate * 100)}%`
              : '—'}
          </div>
          <div style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 0.25rem;">
            Auto-labeled
          </div>
        </div>
        <div class="t-card" style="padding: 1rem; text-align: center;">
          <div style="font-size: 2rem; font-weight: 700; color: var(--status-healthy);">
            {stats?.known_people ?? 0}
          </div>
          <div style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 0.25rem;">
            Known people
          </div>
        </div>
        <div class="t-card" style="padding: 1rem; text-align: center;">
          <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary);">
            {formatAgo(stats?.last_face_processed_at)}
          </div>
          <div style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 0.25rem;">
            Last processed
          </div>
        </div>
        <div class="t-card" style="padding: 1rem; text-align: center;">
          <div style={`font-size: 2rem; font-weight: 700; color: ${(stats?.face_pipeline_errors ?? 0) > 0 ? 'var(--status-error)' : 'var(--status-healthy)'};`}>
            {stats?.face_pipeline_errors ?? 0}
          </div>
          <div style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 0.25rem;">
            Pipeline errors
          </div>
        </div>
      </div>

      {/* Re-deploy banner — appears when new labels exist since last deploy */}
      {newLabelsSinceDeploy && (
        <div style="margin-bottom: 1.5rem; padding: 0.75rem 1rem; background: var(--bg-inset); border-left: 3px solid var(--accent); border-radius: var(--radius); display: flex; align-items: center; justify-content: space-between; gap: 1rem;">
          <span style="font-size: var(--type-label); color: var(--text-primary);">
            {people.length > deployedPeopleCount
              ? `${people.length - deployedPeopleCount} new person${people.length - deployedPeopleCount !== 1 ? 's' : ''} labeled since last deploy — Frigate won't recognize them until you re-deploy.`
              : `${totalSamples - deployedSampleCount} new sample${totalSamples - deployedSampleCount !== 1 ? 's' : ''} added since last deploy — re-deploy to improve recognition accuracy.`
            }
          </span>
          <button
            class="t-btn t-btn-primary"
            onClick={handleDeploy}
            style="padding: 0.375rem 0.75rem; font-size: 0.8125rem; white-space: nowrap; flex-shrink: 0;"
          >
            Re-deploy
          </button>
        </div>
      )}

      {/* 4-step workflow guide — always visible, shows current position */}
      <div class="t-frame" data-label="WORKFLOW" style="padding: 1rem; margin-bottom: 1.5rem;">

        {/* Step 1 — Bootstrap */}
        <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem; align-items: flex-start;">
          <div style={`
            width: 1.75rem; height: 1.75rem; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700;
            background: ${bootstrapDone ? 'var(--status-healthy)' : 'var(--accent)'};
            color: var(--bg-base);
          `}>{bootstrapDone ? '✓' : '1'}</div>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.875rem; margin-bottom: 0.125rem;">
              Run Bootstrap
            </div>
            <div style="font-size: var(--type-label); color: var(--text-secondary); margin-bottom: 0.5rem;">
              Scans all Frigate clips, extracts face images, and groups similar faces together.
              {bootstrapDone && <span style="color: var(--status-healthy);"> Last ran {formatAgo(bootstrapStatus.lastRan)}.</span>}
            </div>
            <button
              class="t-btn t-btn-primary"
              onClick={handleBootstrap}
              disabled={bootstrapStatus.running}
              style={`padding: 0.375rem 0.75rem; font-size: 0.8125rem; opacity: ${bootstrapStatus.running ? 0.5 : 1};`}
            >
              {bootstrapStatus.running ? 'Running…' : bootstrapDone ? 'Re-run Bootstrap' : 'Start Bootstrap'}
            </button>
            {bootstrapStatus.running && (
              <div style="margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: var(--type-label); color: var(--text-secondary); margin-bottom: 0.25rem;">
                  {bootstrapStatus.total === 0
                    ? <span>Scanning Frigate clips for face crops… (30–90s)</span>
                    : <><span>Extracting face images…</span><span>{bootstrapStatus.processed} / {bootstrapStatus.total}</span></>
                  }
                </div>
                <div style="background: var(--bg-inset); border-radius: 2px; overflow: hidden; height: 4px;">
                  <div style={`background: var(--accent); width: ${pct}%; height: 100%; transition: width 0.5s ease;`} />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Step 2 — Label */}
        <div style={`display: flex; gap: 0.75rem; margin-bottom: 1rem; align-items: flex-start; opacity: ${bootstrapDone ? 1 : 0.35};`}>
          <div style={`
            width: 1.75rem; height: 1.75rem; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700;
            background: ${queue.length === 0 && people.length > 0 ? 'var(--status-healthy)' : bootstrapDone ? 'var(--accent)' : 'var(--bg-inset)'};
            color: var(--bg-base);
          `}>{queue.length === 0 && people.length > 0 ? '✓' : '2'}</div>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.875rem; margin-bottom: 0.125rem;">
              Label Faces
            </div>
            <div style="font-size: var(--type-label); color: var(--text-secondary); margin-bottom: 0.375rem;">
              {!bootstrapDone
                ? 'Unlocks after bootstrap. You\'ll see face photos — click a name chip or type a new name.'
                : queue.length > 0
                  ? <>{queue.length} face{queue.length !== 1 ? 's' : ''} waiting below. Click a chip or type a name. Keyboard: <kbd style="background:var(--bg-inset);padding:0 3px;border-radius:2px;font-size:0.7rem;">j/k</kbd> navigate, <kbd style="background:var(--bg-inset);padding:0 3px;border-radius:2px;font-size:0.7rem;">1 2 3</kbd> pick candidate, <kbd style="background:var(--bg-inset);padding:0 3px;border-radius:2px;font-size:0.7rem;">Enter</kbd> submit.</>
                  : people.length > 0
                    ? 'Queue cleared — all done.'
                    : 'No faces found. Try re-running bootstrap.'
              }
            </div>
            {queue.length > 0 && (
              <div style="display: flex; gap: 0.5rem; align-items: center;">
                {confirmClear ? (
                  <>
                    <span style="font-size: var(--type-label); color: var(--status-warning);">
                      Clear all {queue.length} unlabeled faces?
                    </span>
                    <button
                      class="t-btn t-btn-secondary"
                      onClick={handleClearQueue}
                      style="padding: 0.25rem 0.625rem; font-size: 0.8125rem; color: var(--status-warning); border-color: var(--status-warning);"
                    >
                      Yes, clear
                    </button>
                    <button
                      class="t-btn t-btn-secondary"
                      onClick={() => setConfirmClear(false)}
                      style="padding: 0.25rem 0.625rem; font-size: 0.8125rem;"
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    class="t-btn t-btn-secondary"
                    onClick={handleClearQueue}
                    style="padding: 0.375rem 0.75rem; font-size: 0.8125rem; color: var(--text-secondary);"
                  >
                    Clear Queue
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Step 3 — Deploy */}
        <div style={`display: flex; gap: 0.75rem; margin-bottom: 1rem; align-items: flex-start; opacity: ${canDeploy ? 1 : 0.35};`}>
          <div style={`
            width: 1.75rem; height: 1.75rem; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700;
            background: ${deployed ? 'var(--status-healthy)' : canDeploy ? 'var(--accent)' : 'var(--bg-inset)'};
            color: var(--bg-base);
          `}>{deployed ? '✓' : '3'}</div>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.875rem; margin-bottom: 0.125rem;">
              Deploy to Frigate
            </div>
            <div style="font-size: var(--type-label); color: var(--text-secondary); margin-bottom: 0.5rem;">
              {!canDeploy
                ? 'Unlocks after you label at least one person. Copies your best face photos into Frigate\'s recognition library.'
                : `${people.length} person${people.length !== 1 ? 's' : ''} ready to deploy. Copies labeled face images into Frigate so it can recognize people in live footage.`
              }
              {deployMsg && <span style="color: var(--status-healthy);"> {deployMsg}</span>}
            </div>
            <button
              class="t-btn t-btn-primary"
              onClick={handleDeploy}
              disabled={!canDeploy}
              style={`padding: 0.375rem 0.75rem; font-size: 0.8125rem; opacity: ${canDeploy ? 1 : 0.4};`}
            >
              {deployed ? 'Re-deploy' : 'Deploy to Frigate'}
            </button>
          </div>
        </div>

        {/* Step 4 — Restart Frigate */}
        <div style={`display: flex; gap: 0.75rem; align-items: flex-start; opacity: ${deployed ? 1 : 0.35};`}>
          <div style={`
            width: 1.75rem; height: 1.75rem; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 700;
            background: ${frigateRestarted ? 'var(--status-healthy)' : deployed ? 'var(--accent)' : 'var(--bg-inset)'};
            color: var(--bg-base);
          `}>{frigateRestarted ? '✓' : '4'}</div>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-primary); font-size: 0.875rem; margin-bottom: 0.125rem;">
              Restart Frigate
            </div>
            <div style="font-size: var(--type-label); color: var(--text-secondary); margin-bottom: 0.5rem;">
              {!deployed
                ? 'Unlocks after deploy. Frigate reads its face library on startup — a restart is required to activate recognition.'
                : frigateRestarting
                  ? 'Restarting Frigate container… (~5s)'
                  : frigateRestarted
                    ? 'Frigate is running with the new face library. Recognition is live.'
                    : 'Frigate needs a restart to load the new faces. This happens automatically after deploy, or use the button below.'
              }
            </div>
            {deployed && !frigateRestarted && (
              <button
                class="t-btn t-btn-secondary"
                onClick={handleRestartFrigate}
                disabled={frigateRestarting}
                style={`padding: 0.375rem 0.75rem; font-size: 0.8125rem; opacity: ${frigateRestarting ? 0.5 : 1};`}
              >
                {frigateRestarting ? 'Restarting…' : 'Restart Frigate'}
              </button>
            )}
          </div>
        </div>

      </div>

      {/* Review queue — grouped by primary candidate, paginated */}
      {visibleQueue.length > 0 && (
        <div style="margin-bottom: 1.5rem;">
          <div class="t-section-header" style="margin-bottom: 0.75rem;">
            Review Queue ({stats?.queue_depth ?? queue.length})
          </div>
          {groupKeys.map(groupName => {
            const groupItems = queueGroups[groupName];
            const isRealPerson = groupName !== 'Unidentified' && !groupName.startsWith('Group ');
            const isCluster = groupName.startsWith('Group ');
            return (
              <div key={groupName} style="margin-bottom: 1rem;">
                {/* Group header */}
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
                  <span style="font-size: var(--type-label); color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">
                    {groupName} — {groupItems.length} face{groupItems.length !== 1 ? 's' : ''}
                    {isCluster && <span style="color: var(--text-tertiary); font-size: 0.7rem; margin-left: 0.25rem; text-transform: none;">(visually similar)</span>}
                  </span>
                  {isRealPerson && (
                    <button
                      class="t-btn t-btn-secondary"
                      onClick={() => handleConfirmGroup(groupItems, groupName)}
                      style="padding: 0.125rem 0.5rem; font-size: 0.75rem;"
                    >
                      Confirm all as {groupName}
                    </button>
                  )}
                  {isCluster && (
                    <div style="display: flex; align-items: center; gap: 0.375rem;">
                      <input
                        class="t-input"
                        type="text"
                        placeholder="Who is this?"
                        value={clusterGroupInput[groupName] || ''}
                        onInput={evt => setClusterGroupInput(prev => ({ ...prev, [groupName]: evt.target.value }))}
                        onKeyDown={evt => {
                          if (evt.key === 'Enter') {
                            const name = clusterGroupInput[groupName]?.trim();
                            if (name) {
                              handleConfirmGroup(groupItems, name);
                              setClusterGroupInput(prev => ({ ...prev, [groupName]: '' }));
                            }
                          }
                        }}
                        style="width: 9rem; font-size: 0.8125rem; padding: 0.125rem 0.5rem;"
                      />
                      <button
                        class="t-btn t-btn-secondary"
                        onClick={() => {
                          const name = clusterGroupInput[groupName]?.trim();
                          if (name) {
                            handleConfirmGroup(groupItems, name);
                            setClusterGroupInput(prev => ({ ...prev, [groupName]: '' }));
                          }
                        }}
                        disabled={!clusterGroupInput[groupName]?.trim()}
                        style={`padding: 0.125rem 0.5rem; font-size: 0.75rem; opacity: ${clusterGroupInput[groupName]?.trim() ? 1 : 0.4};`}
                      >
                        Label all
                      </button>
                    </div>
                  )}
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                  {groupItems.map(item => {
                    const isFocused = item.id === focusedItemId;
                    return (
                      <div
                        key={item.id}
                        class="t-card"
                        onClick={() => setFocusedItemId(item.id)}
                        style={`
                          padding: 0.75rem; display: flex; gap: 0.75rem; align-items: flex-start;
                          cursor: pointer;
                          ${isFocused ? 'border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent);' : ''}
                        `}
                      >
                        <img
                          src={`${baseUrl}/api/faces/image/${item.id}`}
                          alt={`Face from event ${item.event_id}`}
                          style="width: 5rem; height: 5rem; object-fit: cover; border-radius: var(--radius); flex-shrink: 0; background: var(--bg-inset);"
                          onError={evt => { evt.target.style.display = 'none'; }}
                        />
                        <div style="flex: 1; min-width: 0;">
                          {/* Camera, time, priority */}
                          <div style="font-size: var(--type-label); color: var(--text-tertiary); margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                            <span>Priority: {item.priority?.toFixed(2)}</span>
                            {item.camera && (
                              <span style="color: var(--text-secondary);">{item.camera}</span>
                            )}
                            <span style="color: var(--text-tertiary);">{formatAgo(item.created_at)}</span>
                            {isFocused && <span style="color: var(--accent);">● focused</span>}
                          </div>

                          {/* Candidate chips — click to confirm directly (P1 + P2) */}
                          <div style="display: flex; flex-wrap: wrap; gap: 0.375rem; margin-bottom: 0.5rem;">
                            {item.top_candidates?.filter(c => c.confidence >= 0.30).map((cand, ci) => (
                              <button
                                key={cand.person_name}
                                class="t-btn t-btn-secondary"
                                onClick={evt => { evt.stopPropagation(); handleLabel(item.id, formatCandidate(cand.person_name)); }}
                                disabled={!!labelingItems[item.id]}
                                title={`Confirm as ${formatCandidate(cand.person_name)} (key: ${ci + 1})`}
                                style={`
                                  padding: 0.25rem 0.5rem; font-size: 0.8125rem;
                                  border-color: ${confColor(cand.confidence)};
                                  display: inline-flex; align-items: center; gap: 0.375rem;
                                `}
                              >
                                {isFocused && ci < 3 && (
                                  <span style="font-size: 0.7rem; color: var(--text-tertiary); font-family: monospace;">{ci + 1}</span>
                                )}
                                <span style={`color: ${confColor(cand.confidence)}; font-weight: 600;`}>
                                  {formatCandidate(cand.person_name)}
                                </span>
                                {cand.confidence > 0 && (
                                  <span style="display: inline-flex; align-items: center; gap: 0.25rem;">
                                    <span style={`
                                      display: inline-block; width: ${Math.round(cand.confidence * 48)}px;
                                      max-width: 48px; height: 3px; border-radius: 1px;
                                      background: ${confColor(cand.confidence)};
                                    `} />
                                    <span style={`font-size: 0.75rem; color: ${confColor(cand.confidence)};`}>
                                      {(cand.confidence * 100).toFixed(0)}%
                                    </span>
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>

                          {/* Label input or post-label confirmation */}
                          {labeledItems[item.id] ? (
                            <div style="font-size: 0.875rem; color: var(--status-healthy); font-weight: 500;">
                              Labeled as {labeledItems[item.id]} ✓
                            </div>
                          ) : (
                            <div style="display: flex; gap: 0.5rem; align-items: center;">
                              <input
                                class="t-input"
                                type="text"
                                placeholder="New name…"
                                value={labelInput[item.id] || ''}
                                onInput={evt => setLabelInput(prev => ({ ...prev, [item.id]: evt.target.value }))}
                                onKeyDown={evt => evt.key === 'Enter' && handleLabel(item.id)}
                                onClick={evt => evt.stopPropagation()}
                                style="flex: 1; font-size: 0.875rem; padding: 0.25rem 0.5rem;"
                              />
                              <button
                                class="t-btn t-btn-primary"
                                onClick={evt => { evt.stopPropagation(); handleLabel(item.id); }}
                                disabled={!!labelingItems[item.id]}
                                style={`padding: 0.25rem 0.75rem; font-size: 0.875rem; opacity: ${labelingItems[item.id] ? 0.5 : 1};`}
                              >
                                {labelingItems[item.id] ? '…' : 'Label'}
                              </button>
                              <button
                                class="t-btn t-btn-secondary"
                                onClick={evt => { evt.stopPropagation(); handleDismiss(item.id); }}
                                style="padding: 0.25rem 0.5rem; font-size: 0.8125rem; color: var(--text-tertiary);"
                                title="Dismiss without labeling"
                              >
                                Dismiss
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* Load more / pagination */}
          {hasMore && (
            <div style="text-align: center; margin-top: 0.75rem;">
              <button
                class="t-btn t-btn-secondary"
                onClick={() => setDisplayLimit(prev => prev + 20)}
                style="padding: 0.5rem 1.5rem; font-size: 0.875rem;"
              >
                Load more ({Math.min(queue.length, displayLimit + 20) - displayLimit > 0
                  ? `+${Math.min(queue.length - displayLimit, 20)} of ${(stats?.queue_depth ?? queue.length) - displayLimit} remaining`
                  : `${(stats?.queue_depth ?? queue.length) - displayLimit} remaining`})
              </button>
            </div>
          )}
        </div>
      )}

      {/* Known People — Frigate-tier gallery with person management */}
      {people.length > 0 && (
        <div style="margin-bottom: 1.5rem;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
            <div class="t-section-header">
              Known People ({people.length})
            </div>
            <button
              class="t-btn t-btn-secondary"
              style={`padding: 0.25rem 0.625rem; font-size: 0.75rem; opacity: ${exporting ? 0.5 : 1};`}
              disabled={exporting}
              onClick={async () => {
                if (exporting) return;
                setExporting(true);
                try {
                  const data = await fetchJson('/api/faces/export');
                  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `aria-faces-export-${new Date().toISOString().slice(0,10)}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                } catch (e) { setError(e); } finally { setExporting(false); }
              }}
            >
              {exporting ? 'Exporting…' : 'Export JSON'}
            </button>
          </div>
          <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            {people.map(person => {
              const isExpanded = expandedPerson === person.person_name;
              const samples = personSamples[person.person_name] || [];
              const presence = presencePersons[person.person_name];
              const isRenaming = renamingPerson === person.person_name;
              const isDeleting = deletingPerson === person.person_name;

              // Source breakdown from loaded samples
              const sources = samples.reduce((acc, s) => {
                acc[s.source] = (acc[s.source] || 0) + 1;
                return acc;
              }, {});
              const sourceBreakdown = ['bootstrap', 'manual', 'live']
                .filter(src => sources[src] > 0)
                .map(src => `${sources[src]} ${src}`)
                .join(' · ');

              return (
                <div key={person.person_name} class="t-card" style="overflow: hidden;">
                  {/* Collapsed header row — always visible */}
                  <div
                    style="padding: 0.75rem 1rem; display: flex; align-items: center; gap: 0.75rem; cursor: pointer;"
                    onClick={() => handleTogglePerson(person.person_name)}
                  >
                    {/* Expand indicator */}
                    <span style={`font-size: 0.75rem; color: var(--text-tertiary); transition: transform 0.15s; display: inline-block; transform: ${isExpanded ? 'rotate(90deg)' : 'rotate(0deg)'};`}>
                      ▶
                    </span>

                    {/* Name + counts */}
                    <div style="flex: 1; min-width: 0;">
                      <div style="display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap;">
                        <span style="font-weight: 600; color: var(--text-primary); font-size: 0.9375rem;">
                          {person.person_name}
                        </span>
                        <span style="font-size: var(--type-label); color: var(--text-secondary);">
                          {isExpanded && samples.length > 0
                            ? `${samples.length} samples`
                            : `${person.count} samples`}
                        </span>
                        {isExpanded && sourceBreakdown && (
                          <span style="font-size: var(--type-label); color: var(--text-tertiary);">
                            {sourceBreakdown}
                          </span>
                        )}
                      </div>
                      {/* Last seen from presence data */}
                      {presence?.room && (
                        <div style="font-size: var(--type-label); color: var(--text-tertiary); margin-top: 0.125rem;">
                          Last seen: {presence.room}{presence.last_seen ? ` · ${formatAgo(presence.last_seen)}` : ''}
                        </div>
                      )}
                    </div>

                    {/* Action buttons — stop propagation so they don't toggle expand */}
                    {isExpanded && (
                      <div style="display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0;" onClick={evt => evt.stopPropagation()}>
                        {/* Rename flow */}
                        {isRenaming ? (
                          <>
                            <input
                              class="t-input"
                              type="text"
                              value={renameInput}
                              onInput={evt => setRenameInput(evt.target.value)}
                              onKeyDown={evt => {
                                if (evt.key === 'Enter') handleRename(person.person_name);
                                if (evt.key === 'Escape') { setRenamingPerson(null); setRenameInput(''); }
                              }}
                              style="font-size: 0.8125rem; padding: 0.25rem 0.5rem; width: 8rem;"
                              autoFocus
                            />
                            <button
                              class="t-btn t-btn-primary"
                              onClick={() => handleRename(person.person_name)}
                              style="padding: 0.25rem 0.625rem; font-size: 0.8125rem;"
                            >
                              Save
                            </button>
                            <button
                              class="t-btn t-btn-secondary"
                              onClick={() => { setRenamingPerson(null); setRenameInput(''); }}
                              style="padding: 0.25rem 0.625rem; font-size: 0.8125rem;"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <button
                            class="t-btn t-btn-secondary"
                            onClick={() => { setRenamingPerson(person.person_name); setRenameInput(person.person_name); setDeletingPerson(null); }}
                            style="padding: 0.25rem 0.625rem; font-size: 0.8125rem;"
                          >
                            Rename
                          </button>
                        )}

                        {/* Delete person flow */}
                        {!isRenaming && (
                          isDeleting ? (
                            <>
                              <span style="font-size: var(--type-label); color: var(--status-error); white-space: nowrap;">
                                Delete {person.person_name} and all {person.count} samples?
                              </span>
                              <button
                                class="t-btn t-btn-secondary"
                                onClick={() => handleDeletePerson(person.person_name)}
                                style="padding: 0.25rem 0.625rem; font-size: 0.8125rem; color: var(--status-error); border-color: var(--status-error);"
                              >
                                Confirm
                              </button>
                              <button
                                class="t-btn t-btn-secondary"
                                onClick={() => setDeletingPerson(null)}
                                style="padding: 0.25rem 0.625rem; font-size: 0.8125rem;"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              class="t-btn t-btn-secondary"
                              onClick={() => { handleDeletePerson(person.person_name); setRenamingPerson(null); }}
                              style="padding: 0.25rem 0.625rem; font-size: 0.8125rem; color: var(--status-error); border-color: var(--status-error);"
                            >
                              Delete all
                            </button>
                          )
                        )}
                      </div>
                    )}
                  </div>

                  {/* Expanded panel — sample thumbnail gallery */}
                  {isExpanded && (() => {
                    const verifiedCount = samples.filter(s => s.verified).length;
                    const threshold = Math.max(0.50, 0.85 - (0.005 * verifiedCount)).toFixed(2);
                    return (
                    <div style="padding: 0 1rem 1rem; border-top: 1px solid var(--bg-inset);">
                      {samples.length === 0 ? (
                        <div style="font-size: var(--type-label); color: var(--text-tertiary); padding-top: 0.75rem;">
                          No samples loaded.
                        </div>
                      ) : (
                        <>
                        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; padding-top: 0.75rem;">
                          {samples.map(sample => {
                            const isBeingDeleted = deletingSample?.person === person.person_name && deletingSample?.sampleId === sample.id;
                            return (
                              <div
                                key={sample.id}
                                style={`
                                  position: relative; width: 4rem; height: 4rem;
                                  border-radius: var(--radius); overflow: visible;
                                  background: var(--bg-inset);
                                  border: 1px solid var(--bg-surface-raised);
                                  flex-shrink: 0;
                                `}
                              >
                                <img
                                  src={`${baseUrl}/api/faces/embedding-image/${sample.id}`}
                                  alt={`Sample ${sample.id}`}
                                  style="width: 4rem; height: 4rem; object-fit: cover; border-radius: var(--radius); background: var(--bg-inset);"
                                  onError={evt => { evt.target.style.display = 'none'; }}
                                />

                                {/* Source badge */}
                                <div style={`
                                  position: absolute; bottom: -0.375rem; left: 50%; transform: translateX(-50%);
                                  font-size: 0.5rem; font-weight: 700; text-transform: uppercase;
                                  color: ${sourceBadgeColor(sample.source)};
                                  background: var(--bg-base); padding: 0 0.2rem;
                                  border-radius: 2px; white-space: nowrap; letter-spacing: 0.03em;
                                `}>
                                  {sample.source}
                                </div>

                                {/* Delete button */}
                                {isBeingDeleted ? (
                                  <button
                                    style="position: absolute; top: -0.375rem; right: -0.375rem; width: 1.25rem; height: 1.25rem; border-radius: 50%; background: var(--status-error); border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: white; font-weight: 700;"
                                    onClick={() => handleDeleteSample(person.person_name, sample.id)}
                                    title="Confirm delete"
                                  >
                                    !
                                  </button>
                                ) : (
                                  <button
                                    style="position: absolute; top: -0.375rem; right: -0.375rem; width: 1.25rem; height: 1.25rem; border-radius: 50%; background: var(--bg-surface-raised); border: 1px solid var(--bg-inset); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 0.6rem; color: var(--text-secondary);"
                                    onClick={() => setDeletingSample({ person: person.person_name, sampleId: sample.id })}
                                    title="Remove this training sample"
                                  >
                                    ✕
                                  </button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                        <div style="font-size: var(--type-label); color: var(--text-tertiary); margin-top: 0.5rem;">
                          Recognition threshold: {(threshold * 100).toFixed(0)}% — auto-tightens as verified samples grow (currently {verifiedCount} verified).
                        </div>
                        </>
                      )}
                    </div>
                    );
                  })()}
                </div>
              );
            })}
          </div>
        </div>
      )}

    </div>
  );
}
