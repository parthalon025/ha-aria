import { Section, relativeTime } from './utils.jsx';

export function SystemStatus({ runLog, mlModels, metaLearning }) {
  // Determine overall health
  const hasErrors = runLog && runLog.some(r => r.status === 'error');
  const lastRun = runLog && runLog.length > 0 ? runLog[0] : null;
  const healthNote = hasErrors
    ? 'Errors detected in recent runs \u2014 the pipeline may be missing data.'
    : lastRun
      ? `Last run ${relativeTime(lastRun.timestamp)}. Everything healthy.`
      : 'No runs recorded yet.';

  return (
    <Section
      title="System Status"
      subtitle={`If something is red here, the intelligence pipeline is broken and won't catch issues in your home. ${healthNote}`}
      summary={hasErrors ? "errors" : "healthy"}
    >
      <div class="space-y-4">
        {/* Run Log */}
        <div class="t-frame overflow-x-auto" data-label="run-log">
          <div class="px-4 py-2 text-xs font-bold uppercase" style="border-bottom: 1px solid var(--border-subtle); color: var(--text-tertiary)">Run Log</div>
          {(!runLog || runLog.length === 0) ? (
            <div class="px-4 py-3 text-sm" style="color: var(--text-tertiary)">No runs recorded yet.</div>
          ) : (
            <table class="w-full text-sm">
              <thead>
                <tr class="text-left text-xs" style="border-bottom: 1px solid var(--border-subtle); color: var(--text-tertiary)">
                  <th class="px-4 py-1">When</th>
                  <th class="px-4 py-1">Type</th>
                  <th class="px-4 py-1">Status</th>
                </tr>
              </thead>
              <tbody>
                {runLog.map((r, i) => (
                  <tr key={i} style="border-bottom: 1px solid var(--border-subtle)">
                    <td class="px-4 py-1.5" style="color: var(--text-secondary)" title={r.timestamp}>{relativeTime(r.timestamp)}</td>
                    <td class="px-4 py-1.5">
                      <span class="rounded px-1.5 py-0.5 text-xs" style="background: var(--bg-surface-raised)">{r.type}</span>
                    </td>
                    <td class="px-4 py-1.5">
                      <span class="inline-block w-2 h-2 rounded-full" style={`background: ${r.status === 'ok' ? 'var(--status-healthy)' : 'var(--status-error)'}`} />
                      {r.message && <span class="ml-1 text-xs" style="color: var(--status-error)">{r.message}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* ML Models */}
        <div class="t-frame p-4" data-label="ml-models">
          <div class="text-xs font-bold uppercase mb-2" style="color: var(--text-tertiary)">ML Models</div>
          {(!mlModels || mlModels.count === 0) ? (
            <p class="text-sm" style="color: var(--text-tertiary)">ML models activate after 14 days of data. Until then, predictions use statistical baselines only.</p>
          ) : (
            <div class="space-y-2">
              <div class="flex gap-3 text-sm">
                <span class="rounded px-2 py-0.5" style="background: var(--bg-surface-raised)">{mlModels.count} model{mlModels.count !== 1 ? 's' : ''}</span>
                {mlModels.last_trained && <span style="color: var(--text-tertiary)">Last trained: {relativeTime(mlModels.last_trained)}</span>}
              </div>
              {mlModels.scores && Object.keys(mlModels.scores).length > 0 && (
                <table class="w-full text-xs">
                  <thead>
                    <tr class="text-left" style="color: var(--text-tertiary)"><th class="py-1">Model</th><th>R2</th><th>MAE</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(mlModels.scores).map(([name, s]) => (
                      <tr key={name}>
                        <td class="py-1 data-mono">{name}</td>
                        <td>{s.r2 !== null && s.r2 !== undefined ? s.r2.toFixed(3) : '\u2014'}</td>
                        <td>{s.mae !== null && s.mae !== undefined ? s.mae.toFixed(2) : '\u2014'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>

        {/* Meta-Learning */}
        <div class="t-frame p-4" data-label="meta-learning">
          <div class="text-xs font-bold uppercase mb-2" style="color: var(--text-tertiary)">Meta-Learning</div>
          {(!metaLearning || metaLearning.applied_count === 0) ? (
            <p class="text-sm" style="color: var(--text-tertiary)">Meta-learning reviews model performance weekly and auto-tunes feature selection. Activates after the first training cycle.</p>
          ) : (
            <div class="space-y-2">
              <div class="flex gap-3 text-sm">
                <span class="rounded px-2 py-0.5" style="background: var(--bg-surface-raised)">{metaLearning.applied_count} suggestion{metaLearning.applied_count !== 1 ? 's' : ''} applied</span>
                {metaLearning.last_applied && <span style="color: var(--text-tertiary)">Last: {relativeTime(metaLearning.last_applied)}</span>}
              </div>
              {metaLearning.suggestions && metaLearning.suggestions.length > 0 && (
                <ul class="text-xs space-y-1 list-disc ml-4" style="color: var(--text-secondary)">
                  {metaLearning.suggestions.map((s, i) => (
                    <li key={i}>{s.description || s.action || JSON.stringify(s)}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}
