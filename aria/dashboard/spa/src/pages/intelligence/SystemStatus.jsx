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
    >
      <div class="space-y-4">
        {/* Run Log */}
        <div class="bg-white rounded-lg shadow-sm overflow-x-auto">
          <div class="px-4 py-2 border-b border-gray-200 text-xs font-bold text-gray-500 uppercase">Run Log</div>
          {(!runLog || runLog.length === 0) ? (
            <div class="px-4 py-3 text-sm text-gray-400">No runs recorded yet.</div>
          ) : (
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-gray-100 text-left text-xs text-gray-500">
                  <th class="px-4 py-1">When</th>
                  <th class="px-4 py-1">Type</th>
                  <th class="px-4 py-1">Status</th>
                </tr>
              </thead>
              <tbody>
                {runLog.map((r, i) => (
                  <tr key={i} class="border-b border-gray-50">
                    <td class="px-4 py-1.5 text-gray-600" title={r.timestamp}>{relativeTime(r.timestamp)}</td>
                    <td class="px-4 py-1.5">
                      <span class="bg-gray-100 rounded px-1.5 py-0.5 text-xs">{r.type}</span>
                    </td>
                    <td class="px-4 py-1.5">
                      <span class={`inline-block w-2 h-2 rounded-full ${r.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
                      {r.message && <span class="ml-1 text-xs text-red-600">{r.message}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* ML Models */}
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-xs font-bold text-gray-500 uppercase mb-2">ML Models</div>
          {(!mlModels || mlModels.count === 0) ? (
            <p class="text-sm text-gray-400">ML models activate after 14 days of data. Until then, predictions use statistical baselines only.</p>
          ) : (
            <div class="space-y-2">
              <div class="flex gap-3 text-sm">
                <span class="bg-gray-100 rounded px-2 py-0.5">{mlModels.count} model{mlModels.count !== 1 ? 's' : ''}</span>
                {mlModels.last_trained && <span class="text-gray-500">Last trained: {relativeTime(mlModels.last_trained)}</span>}
              </div>
              {mlModels.scores && Object.keys(mlModels.scores).length > 0 && (
                <table class="w-full text-xs">
                  <thead>
                    <tr class="text-left text-gray-500"><th class="py-1">Model</th><th>R2</th><th>MAE</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(mlModels.scores).map(([name, s]) => (
                      <tr key={name}>
                        <td class="py-1 font-mono">{name}</td>
                        <td>{s.r2 != null ? s.r2.toFixed(3) : '\u2014'}</td>
                        <td>{s.mae != null ? s.mae.toFixed(2) : '\u2014'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>

        {/* Meta-Learning */}
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="text-xs font-bold text-gray-500 uppercase mb-2">Meta-Learning</div>
          {(!metaLearning || metaLearning.applied_count === 0) ? (
            <p class="text-sm text-gray-400">Meta-learning reviews model performance weekly and auto-tunes feature selection. Activates after the first training cycle.</p>
          ) : (
            <div class="space-y-2">
              <div class="flex gap-3 text-sm">
                <span class="bg-gray-100 rounded px-2 py-0.5">{metaLearning.applied_count} suggestion{metaLearning.applied_count !== 1 ? 's' : ''} applied</span>
                {metaLearning.last_applied && <span class="text-gray-500">Last: {relativeTime(metaLearning.last_applied)}</span>}
              </div>
              {metaLearning.suggestions && metaLearning.suggestions.length > 0 && (
                <ul class="text-xs text-gray-600 space-y-1 list-disc ml-4">
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
