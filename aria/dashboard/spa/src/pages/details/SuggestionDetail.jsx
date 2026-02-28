/**
 * SuggestionDetail — Detail renderer for automation suggestions.
 * Three-section layout: Summary (title + status), Explanation (rule details), History (feedback).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

function statusColor(status) {
  if (status === 'approved') return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (status === 'rejected') return 'background: var(--status-error-glow); color: var(--status-error);';
  return 'background: var(--bg-inset); color: var(--text-secondary);';
}

export default function SuggestionDetail({ id, type: _type }) {
  const [suggestion, setSuggestion] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/cache/automation_suggestions'),
      fetchJson('/api/automations/feedback').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([suggestionsResult, feedbackResult]) => {
        const data = suggestionsResult?.data || suggestionsResult || {};
        const list = Array.isArray(data) ? data : (data.suggestions || []);
        const match = list.find(
          (s) => s.id === id || s.suggestion_id === id || String(s.id) === id
        );
        setSuggestion(match || null);

        if (feedbackResult) {
          const fbData = feedbackResult?.data || feedbackResult || {};
          const allFb = Array.isArray(fbData) ? fbData : (fbData.feedback || fbData.entries || []);
          const matched = allFb.filter(
            (fb) => fb.suggestion_id === id || fb.id === id || String(fb.suggestion_id) === id
          );
          matched.sort((a, b) => {
            const ta = new Date(a.timestamp || a.created_at || 0).getTime();
            const tb = new Date(b.timestamp || b.created_at || 0).getTime();
            return tb - ta;
          });
          setFeedback(matched);
        }
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!suggestion) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No suggestion found for: {id}
        </p>
      </div>
    );
  }

  const status = suggestion.status || 'pending';
  const confidence = suggestion.confidence ?? suggestion.confidence_score ?? null;

  const statsItems = [
    { label: 'Title', value: suggestion.title || suggestion.name || id },
    { label: 'Status', value: status },
  ];
  if (confidence !== null && confidence !== undefined) {
    statsItems.push({ label: 'Confidence', value: `${(confidence * 100).toFixed(0)}%` });
  }
  if (suggestion.trigger_entity) {
    statsItems.push({ label: 'Trigger', value: suggestion.trigger_entity });
  }

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            style={`display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); text-transform: uppercase; ${statusColor(status)}`}
          >
            {status}
          </span>
        </div>
        {suggestion.description && (
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 12px;">
            {suggestion.description}
          </p>
        )}
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        <div class="space-y-1">
          <span
            style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
          >
            Automation Rule
          </span>
          {suggestion.trigger_entity && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Trigger Entity</span>
              <span style="color: var(--text-secondary);">{suggestion.trigger_entity}</span>
            </div>
          )}
          {suggestion.conditions && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Conditions</span>
              <span style="color: var(--text-secondary); max-width: 60%; text-align: right; word-break: break-word;">
                {typeof suggestion.conditions === 'object' ? JSON.stringify(suggestion.conditions) : String(suggestion.conditions)}
              </span>
            </div>
          )}
          {suggestion.action && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Action</span>
              <span style="color: var(--text-secondary); max-width: 60%; text-align: right; word-break: break-word;">
                {typeof suggestion.action === 'object' ? JSON.stringify(suggestion.action) : String(suggestion.action)}
              </span>
            </div>
          )}
          {suggestion.expected_benefit && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Expected Benefit</span>
              <span style="color: var(--text-secondary);">{suggestion.expected_benefit}</span>
            </div>
          )}
          {suggestion.rule && (
            <div style="font-family: var(--font-mono); font-size: var(--type-label); margin-top: 8px;">
              <span style="color: var(--text-tertiary);">Rule</span>
              <pre style="color: var(--text-secondary); margin: 4px 0; white-space: pre-wrap; word-break: break-word;">
                {typeof suggestion.rule === 'object' ? JSON.stringify(suggestion.rule, null, 2) : String(suggestion.rule)}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {feedback.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Feedback History ({feedback.length})
            </span>
            {feedback.map((fb, idx) => (
              <div key={idx} class="flex gap-3 items-center" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                  {relativeTime(fb.timestamp || fb.created_at)}
                </span>
                <span
                  style={`padding: 1px 6px; border-radius: 3px; font-size: var(--type-label); ${statusColor(fb.action || fb.status || 'pending')}`}
                >
                  {fb.action || fb.status || 'update'}
                </span>
                <span style="color: var(--text-tertiary); flex: 1; text-align: right;">
                  {fb.decided_by || fb.user || ''}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No feedback history available
          </p>
        )}
      </div>
    </div>
  );
}
