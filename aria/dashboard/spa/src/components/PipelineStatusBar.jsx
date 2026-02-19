import { useState, useEffect } from 'preact/hooks';
import { wsConnected } from '../store.js';
import { fetchJson } from '../api.js';

/**
 * Compact one-line pipeline status bar for the Home page.
 * Shows: Pipeline stage · Shadow stage · WebSocket status.
 * Applies superhot data attributes when modules fail or WS is down.
 */
export default function PipelineStatusBar() {
  const [health, setHealth] = useState(null);
  const [pipeline, setPipeline] = useState(null);
  const connected = wsConnected.value;

  useEffect(() => {
    fetchJson('/health').then(setHealth).catch(() => {});
    fetchJson('/api/pipeline').then(setPipeline).catch(() => {});
  }, []);

  const pipelineStage = pipeline?.current_stage || 'starting';
  const shadowStage = pipeline?.current_stage || 'backtest';
  const hasFailed = health?.modules && Object.values(health.modules).some((status) => status === 'failed');

  const attrs = {};
  if (hasFailed) {
    attrs['data-sh-effect'] = 'glitch';
  }
  if (!connected) {
    attrs['data-sh-mantra'] = 'OFFLINE';
  }

  return (
    <div
      class="t-frame"
      style="padding: 8px 16px;"
      {...attrs}
    >
      <div class="flex items-center gap-3 text-xs" style="color: var(--text-tertiary); font-family: var(--font-mono);">
        <span>
          Pipeline: <span style={`color: ${hasFailed ? 'var(--status-error)' : 'var(--text-secondary)'}`}>{pipelineStage}</span>
        </span>
        <span style="color: var(--border-subtle);">&middot;</span>
        <span>
          Shadow: <span style="color: var(--text-secondary)">{shadowStage}</span>
        </span>
        <span style="color: var(--border-subtle);">&middot;</span>
        <span class="flex items-center gap-1">
          <span
            class="inline-block w-1.5 h-1.5 rounded-full"
            style={`background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`}
          />
          WebSocket: <span style={`color: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'}`}>{connected ? 'connected' : 'disconnected'}</span>
        </span>
      </div>
    </div>
  );
}
