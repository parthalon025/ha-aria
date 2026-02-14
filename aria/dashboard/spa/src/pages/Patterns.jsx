import { useState } from 'preact/hooks';
import useCache from '../hooks/useCache.js';
import useComputed from '../hooks/useComputed.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Type badge style by pattern type. */
function typeBadgeStyle(type) {
  switch ((type || '').toLowerCase()) {
    case 'temporal': return 'background: var(--accent-glow); color: var(--accent);';
    case 'correlation': return 'background: var(--accent-purple-glow); color: var(--accent-purple);';
    case 'sequence': return 'background: var(--status-warning-glow); color: var(--status-warning);';
    case 'anomaly': return 'background: var(--status-error-glow); color: var(--status-error);';
    default: return 'background: var(--bg-surface-raised); color: var(--text-secondary);';
  }
}

function PatternCard({ pattern }) {
  const [expanded, setExpanded] = useState(false);

  const confidence = pattern.confidence ?? 0;
  const pct = Math.round(confidence * 100);
  const entities = pattern.entities || [];

  return (
    <div class="t-card" style="padding: 1.25rem;">
      {/* Header */}
      <div class="flex items-center justify-between mb-2">
        <h3 class="text-base font-bold" style="color: var(--text-primary)">{pattern.name || 'Unnamed pattern'}</h3>
        {pattern.type && (
          <span class="inline-block px-2 py-0.5 rounded-full text-xs font-medium" style={typeBadgeStyle(pattern.type)}>
            {pattern.type}
          </span>
        )}
      </div>

      {/* Description */}
      {pattern.description && (
        <p class="text-sm mb-3" style="color: var(--text-secondary)">{pattern.description}</p>
      )}

      {/* Details grid */}
      <div class="grid grid-cols-2 gap-x-6 gap-y-2 text-sm mb-3">
        {pattern.frequency && (
          <div>
            <span style="color: var(--text-tertiary)">Frequency</span>
            <div class="font-medium" style="color: var(--text-secondary)">{pattern.frequency}</div>
          </div>
        )}
        {entities.length > 0 && (
          <div>
            <span style="color: var(--text-tertiary)">Entities</span>
            <div class="font-medium" style="color: var(--text-secondary)">{entities.length}</div>
          </div>
        )}
        <div>
          <span style="color: var(--text-tertiary)">Confidence</span>
          <div class="font-medium" style="color: var(--text-secondary)">{pct}%</div>
        </div>
        {pattern.time_window && (
          <div>
            <span style="color: var(--text-tertiary)">Time window</span>
            <div class="font-medium" style="color: var(--text-secondary)">{pattern.time_window}</div>
          </div>
        )}
        {pattern.support != null && (
          <div>
            <span style="color: var(--text-tertiary)">Support</span>
            <div class="font-medium" style="color: var(--text-secondary)">{Math.round(pattern.support * 100)}%</div>
          </div>
        )}
      </div>

      {/* Entity list */}
      {entities.length > 0 && (
        <div class="flex flex-wrap gap-1 mb-3">
          {entities.map((eid) => (
            <span key={eid} class="inline-block px-1.5 py-0.5 text-xs data-mono" style="background: var(--bg-surface-raised); border-radius: var(--radius); color: var(--text-secondary)">
              {eid}
            </span>
          ))}
        </div>
      )}

      {/* Expandable raw data */}
      {pattern.raw_data && (
        <div>
          <button
            onClick={() => setExpanded(!expanded)}
            class="text-sm cursor-pointer"
            style="color: var(--accent)"
          >
            {expanded ? 'Hide raw data' : 'Show raw data'}
          </button>
          {expanded && (
            <pre class="mt-2 p-3 text-xs overflow-x-auto" style="background: var(--bg-inset); color: var(--text-primary); border-radius: var(--radius); font-family: var(--font-mono)">
              {JSON.stringify(pattern.raw_data, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function Patterns() {
  const { data, loading, error, refetch } = useCache('patterns');

  const { metadata, patterns } = useComputed(() => {
    if (!data || !data.data) return { metadata: null, patterns: [] };
    const inner = data.data;
    return {
      metadata: inner.metadata || null,
      patterns: inner.patterns || [],
    };
  }, [data]);

  const pageSubtitle = "Recurring behaviors the system has found — things that happen at the same time, in the same sequence, or that correlate with each other.";

  if (loading && !data) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Patterns</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <LoadingState type="cards" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Patterns</h1>
          <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
        </div>
        <ErrorState error={error} onRetry={refetch} />
      </div>
    );
  }

  return (
    <div class="space-y-6 animate-page-enter">
      <div class="t-section-header animate-fade-in-up" style="padding-bottom: 8px;">
        <h1 class="text-2xl font-bold" style="color: var(--text-primary)">Patterns</h1>
        <p class="text-sm" style="color: var(--text-tertiary)">{pageSubtitle}</p>
      </div>

      {/* Metadata summary */}
      {metadata && (
        <div class="flex flex-wrap gap-3 text-sm animate-fade-in-up delay-100" style="color: var(--text-tertiary)">
          {metadata.time_range && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">Range: {metadata.time_range}</span>
          )}
          {metadata.pattern_count != null && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">{metadata.pattern_count} patterns</span>
          )}
          {metadata.analyzed_at && (
            <span style="background: var(--bg-surface-raised); border-radius: var(--radius); padding: 0.25rem 0.5rem;">
              Analyzed: {new Date(metadata.analyzed_at).toLocaleString()}
            </span>
          )}
        </div>
      )}

      {patterns.length === 0 ? (
        <div class="t-callout animate-fade-in-up delay-200" style="padding: 0.75rem;">
          <span class="text-sm" style="color: var(--text-secondary)">No patterns detected yet. The pattern recognition module analyzes HA logbook sequences to find temporal, correlation, and sequence patterns. It needs several days of logbook data with meaningful device events to identify reliable patterns.</span>
        </div>
      ) : (
        <div class="space-y-4 stagger-children">
          {patterns.map((pat, i) => (
            <PatternCard key={pat.name || i} pattern={pat} />
          ))}
        </div>
      )}
    </div>
  );
}
