/**
 * CorrelationDetail — Detail renderer for entity correlations.
 * Three-section layout: Summary (pair + coefficient), Explanation (meaning), History (limited).
 * The id is formatted as "entity1--entity2" (double dash separator).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';

function strengthLabel(coeff) {
  const abs = Math.abs(coeff);
  if (abs >= 0.7) return 'strong';
  if (abs >= 0.4) return 'moderate';
  return 'weak';
}

function strengthColor(label) {
  if (label === 'strong') return 'color: var(--accent);';
  if (label === 'moderate') return 'color: var(--status-warning);';
  return 'color: var(--text-tertiary);';
}

export default function CorrelationDetail({ id, type: _type }) {
  const [correlation, setCorrelation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  // Split id on double-dash to get entity pair
  const parts = id.split('--');
  const entity1 = parts[0] || id;
  const entity2 = parts[1] || '';

  useEffect(() => {
    setLoading(true);
    setError(null);

    fetchJson('/api/cache/intelligence')
      .then((result) => {
        const data = result?.data || result || {};
        // Look for correlations in intelligence data
        const correlations = data.correlations || data.entity_correlations || [];
        let found = null;

        if (Array.isArray(correlations)) {
          found = correlations.find(
            (c) =>
              (c.entity_1 === entity1 && c.entity_2 === entity2) ||
              (c.entity_1 === entity2 && c.entity_2 === entity1) ||
              c.id === id ||
              c.pair === id
          );
        } else if (typeof correlations === 'object') {
          found = correlations[id] || correlations[`${entity1}--${entity2}`] || correlations[`${entity2}--${entity1}`] || null;
        }

        setCorrelation(found);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, entity1, entity2, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (!correlation) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No correlation found for: {entity1} / {entity2 || id}
        </p>
      </div>
    );
  }

  const coefficient = correlation.correlation ?? correlation.coefficient ?? correlation.value ?? 0;
  const strength = strengthLabel(coefficient);
  const direction = coefficient >= 0 ? 'positive' : 'negative';

  const statsItems = [
    { label: 'Entity 1', value: correlation.entity_1 || entity1 },
    { label: 'Entity 2', value: correlation.entity_2 || entity2 },
    { label: 'Coefficient', value: Number(coefficient).toFixed(3) },
    { label: 'Strength', value: strength },
  ];

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        <div style="margin-bottom: 12px;">
          <span
            class="data-mono"
            style={`font-size: var(--type-hero); font-weight: 600; line-height: 1; ${strengthColor(strength)}`}
          >
            {Number(coefficient).toFixed(3)}
          </span>
          <span
            style={`display: inline-block; margin-left: 12px; padding: 2px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: var(--type-label); text-transform: uppercase; background: var(--bg-inset); ${strengthColor(strength)}`}
          >
            {strength}
          </span>
        </div>
        <StatsGrid items={statsItems} />
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        <div class="space-y-1">
          <p style="color: var(--text-secondary); font-family: var(--font-mono); font-size: var(--type-label); margin-bottom: 8px;">
            {strength === 'strong'
              ? `These two entities show a ${direction} correlation, indicating their states frequently change together.`
              : strength === 'moderate'
                ? `These entities have a ${direction} correlation, suggesting some relationship in their state changes.`
                : `These entities show a weak ${direction} correlation, with limited relationship between their state changes.`
            }
          </p>

          {(correlation.entity_1_state || correlation.state_1) && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Entity 1 State</span>
              <span style="color: var(--text-secondary);">{correlation.entity_1_state || correlation.state_1}</span>
            </div>
          )}
          {(correlation.entity_2_state || correlation.state_2) && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Entity 2 State</span>
              <span style="color: var(--text-secondary);">{correlation.entity_2_state || correlation.state_2}</span>
            </div>
          )}
          {(correlation.time_range || correlation.period) && (
            <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
              <span style="color: var(--text-tertiary);">Time Range</span>
              <span style="color: var(--text-secondary);">{correlation.time_range || correlation.period}</span>
            </div>
          )}
        </div>
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          Correlation computed from intelligence data
        </p>
      </div>
    </div>
  );
}
