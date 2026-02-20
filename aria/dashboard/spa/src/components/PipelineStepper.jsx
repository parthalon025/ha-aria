import { useState, useMemo } from 'preact/hooks';
import { SOURCES, INTAKE, PROCESSING, ENRICHMENT, OUTPUTS, getNodeMetric, ACTION_CONDITIONS } from '../lib/pipelineGraph.js';

const STAGES = [
  { label: 'SOURCES', nodes: SOURCES },
  { label: 'INTAKE', nodes: INTAKE },
  { label: 'PROCESSING', nodes: PROCESSING },
  { label: 'ENRICHMENT', nodes: ENRICHMENT },
  { label: 'OUTPUTS', nodes: OUTPUTS },
];

const STATUS_COLORS = {
  healthy: 'var(--status-healthy)',
  warning: 'var(--status-warning)',
  blocked: 'var(--status-error)',
  waiting: 'var(--status-waiting)',
};

function getModuleStatus(moduleStatuses, nodeId) {
  const s = moduleStatuses?.[nodeId];
  if (s === 'running') return 'healthy';
  if (s === 'failed') return 'blocked';
  if (s === 'starting') return 'waiting';
  return 'waiting';
}

export default function PipelineStepper({ moduleStatuses, cacheData }) {
  const [expandedStage, setExpandedStage] = useState(-1);

  const action = useMemo(() => {
    for (const cond of ACTION_CONDITIONS) {
      if (cond.test(cacheData)) {
        const text = typeof cond.text === 'function' ? cond.text(cacheData) : cond.text;
        return { text, href: cond.href };
      }
    }
    return null;
  }, [cacheData]);

  return (
    <section class="t-frame" data-label="pipeline">
      <div class="space-y-1">
        {STAGES.map((stage, idx) => {
          const isExpanded = expandedStage === idx;
          return (
            <div key={stage.label}>
              {/* Stage row */}
              <button
                class="w-full flex items-center gap-2 py-2 px-1 text-left"
                style="background: none; border: none; cursor: pointer;"
                onClick={() => setExpandedStage(isExpanded ? -1 : idx)}
              >
                <span
                  class="w-2 h-2 rounded-full flex-shrink-0"
                  style={`background: ${(() => {
                    const statuses = stage.nodes
                      .filter((nd) => nd.column > 0)
                      .map((nd) => getModuleStatus(moduleStatuses, nd.id));
                    if (statuses.includes('blocked')) return 'var(--status-error)';
                    if (statuses.includes('warning')) return 'var(--status-warning)';
                    if (statuses.some((s) => s === 'healthy')) return 'var(--status-healthy)';
                    return statuses.length === 0 ? 'var(--status-healthy)' : 'var(--status-waiting)';
                  })()};`}
                />
                <span class="flex-1 text-xs font-semibold" style="color: var(--text-primary); font-family: var(--font-mono);">
                  {stage.label}
                </span>
                <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
                  {stage.nodes.length} modules
                </span>
              </button>

              {/* Expanded nodes */}
              {isExpanded && (
                <div class="pl-4 pb-2 space-y-1">
                  {stage.nodes.map((node) => {
                    const status = node.column === 0 ? 'healthy' : getModuleStatus(moduleStatuses, node.id);
                    const metric = getNodeMetric(cacheData, node.id);
                    const href = node.column === 4 && node.page
                      ? node.page
                      : node.column > 0 && node.column < 4
                        ? `#/detail/module/${node.id}`
                        : null;

                    return href ? (
                      <a
                        key={node.id}
                        href={href}
                        class="clickable-data flex items-center gap-2 py-1"
                        style="text-decoration: none; color: inherit;"
                      >
                        <span
                          class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                          style={`background: ${STATUS_COLORS[status]};`}
                        />
                        <span class="flex-1 text-xs" style="color: var(--text-secondary); font-family: var(--font-mono);">
                          {node.label}
                        </span>
                        <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
                          {metric}
                        </span>
                      </a>
                    ) : (
                      <div key={node.id} class="flex items-center gap-2 py-1">
                        <span
                          class="w-1.5 h-1.5 rounded-full flex-shrink-0"
                          style={`background: ${STATUS_COLORS[status]};`}
                        />
                        <span class="flex-1 text-xs" style="color: var(--text-secondary); font-family: var(--font-mono);">
                          {node.label}
                        </span>
                        <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
                          {metric}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Connector line between stages */}
              {idx < STAGES.length - 1 && idx !== 1 && (
                <div class="ml-1 h-3 border-l" style="border-color: var(--border-primary);" />
              )}

              {/* Bus bar between Intake and Processing */}
              {idx === 1 && (
                <div
                  class="my-1 mx-1 px-2 py-1 rounded text-center text-xs font-bold"
                  style="background: var(--bg-terminal); border: 1px solid var(--status-healthy); color: var(--status-healthy); font-family: var(--font-mono);"
                >
                  HUB CACHE
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Action strip */}
      {action && (
        <div class="mt-3 pt-3" style="border-top: 1px solid var(--border-primary);">
          {action.href ? (
            <a href={action.href} class="text-xs font-medium" style="color: var(--accent); font-family: var(--font-mono); text-decoration: none;">
              {action.text}
            </a>
          ) : (
            <span class="text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
              {action.text}
            </span>
          )}
        </div>
      )}
    </section>
  );
}
