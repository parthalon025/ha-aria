/**
 * PredictionDetail — Detail renderer for shadow mode predictions.
 * Three-section layout: Summary (latest prediction), Explanation (Thompson stats), History (prediction list).
 */
import { useState, useEffect } from 'preact/hooks';
import { fetchJson } from '../../api.js';
import HeroCard from '../../components/HeroCard.jsx';
import StatsGrid from '../../components/StatsGrid.jsx';
import LoadingState from '../../components/LoadingState.jsx';
import ErrorState from '../../components/ErrorState.jsx';
import { relativeTime } from '../intelligence/utils.jsx';

export default function PredictionDetail({ id, type: _type }) {
  const [predictions, setPredictions] = useState([]);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      fetchJson('/api/shadow/predictions'),
      fetchJson('/api/shadow/accuracy').catch(err => { console.warn('Optional fetch failed:', err.message); return null; }),
    ])
      .then(([predResult, accResult]) => {
        const allPreds = Array.isArray(predResult)
          ? predResult
          : (predResult?.predictions || predResult?.data || []);

        // Filter predictions matching this id (could be capability or entity)
        const matched = allPreds.filter(
          (pred) =>
            pred.entity_id === id ||
            pred.capability === id ||
            pred.id === id ||
            pred.name === id
        );

        // Sort newest first
        matched.sort((a, b) => {
          const ta = new Date(a.timestamp || a.created_at || 0).getTime();
          const tb = new Date(b.timestamp || b.created_at || 0).getTime();
          return tb - ta;
        });

        setPredictions(matched);
        setAccuracy(accResult?.data || accResult || null);
      })
      .catch((err) => setError(err))
      .finally(() => setLoading(false));
  }, [id, retryCount]);

  if (loading) return <LoadingState type="cards" />;
  if (error) return <ErrorState error={error} onRetry={() => setRetryCount((prev) => prev + 1)} />;
  if (predictions.length === 0) {
    return (
      <div class="t-frame" data-label="not found">
        <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
          No predictions found for: {id}
        </p>
      </div>
    );
  }

  const latest = predictions[0];
  const wasCorrect = latest.was_correct ?? latest.correct ?? null;
  const confidence = latest.confidence ?? latest.confidence_score ?? null;

  const statsItems = [
    { label: 'Predicted', value: String(latest.predicted ?? latest.predicted_value ?? '\u2014') },
    { label: 'Actual', value: String(latest.actual ?? latest.actual_value ?? '\u2014') },
  ];
  if (wasCorrect !== null && wasCorrect !== undefined) {
    statsItems.push({
      label: 'Correct',
      value: wasCorrect ? 'Yes' : 'No',
      warning: !wasCorrect,
    });
  }

  // Thompson Sampling stats — look in accuracy data or latest prediction
  const thompson = latest.thompson || latest.thompson_sampling || accuracy?.thompson?.[id] || null;
  const method = latest.method || latest.model || latest.prediction_method || null;

  return (
    <div class="space-y-6">
      {/* Summary */}
      <div class="t-frame" data-label="summary">
        {confidence !== null && confidence !== undefined && (
          <HeroCard
            value={`${(confidence * 100).toFixed(0)}%`}
            label="Confidence"
            timestamp={latest.timestamp || latest.created_at}
          />
        )}
        <div style={confidence !== null && confidence !== undefined ? 'margin-top: 12px;' : ''}>
          <StatsGrid items={statsItems} />
        </div>
      </div>

      {/* Explanation */}
      <div class="t-frame" data-label="explanation">
        {thompson ? (
          <div class="space-y-1" style="margin-bottom: 16px;">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Thompson Sampling
            </span>
            {thompson.alpha !== null && thompson.alpha !== undefined && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Alpha</span>
                <span style="color: var(--text-secondary);">{Number(thompson.alpha).toFixed(2)}</span>
              </div>
            )}
            {thompson.beta !== null && thompson.beta !== undefined && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Beta</span>
                <span style="color: var(--text-secondary);">{Number(thompson.beta).toFixed(2)}</span>
              </div>
            )}
            {thompson.expected_accuracy !== null && thompson.expected_accuracy !== undefined && (
              <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
                <span style="color: var(--text-tertiary);">Expected Accuracy</span>
                <span style="color: var(--text-secondary);">{(thompson.expected_accuracy * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        ) : null}

        {method && (
          <div class="flex justify-between" style="font-family: var(--font-mono); font-size: var(--type-label);">
            <span style="color: var(--text-tertiary);">Method</span>
            <span style="color: var(--text-secondary);">{method}</span>
          </div>
        )}

        {!thompson && !method && (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No model details available
          </p>
        )}
      </div>

      {/* History */}
      <div class="t-frame" data-label="history">
        {predictions.length > 0 ? (
          <div class="space-y-2">
            <span
              style="font-size: var(--type-label); color: var(--text-tertiary); font-family: var(--font-mono); text-transform: uppercase;"
            >
              Prediction History ({predictions.length})
            </span>
            {predictions.map((pred, idx) => {
              const correct = pred.was_correct ?? pred.correct ?? null;
              const ts = pred.timestamp || pred.created_at;
              return (
                <div key={idx} class="flex gap-3 items-center" style="font-family: var(--font-mono); font-size: var(--type-label);">
                  <span style="color: var(--text-tertiary); min-width: 60px; flex-shrink: 0;">
                    {relativeTime(ts)}
                  </span>
                  <span style="color: var(--text-secondary); flex: 1;">
                    {String(pred.predicted ?? pred.predicted_value ?? '?')} / {String(pred.actual ?? pred.actual_value ?? '?')}
                  </span>
                  {correct !== null && correct !== undefined && (
                    <span
                      style={`padding: 1px 6px; border-radius: 3px; font-size: var(--type-label); ${
                        correct
                          ? 'background: var(--status-healthy-glow); color: var(--status-healthy);'
                          : 'background: var(--status-error-glow); color: var(--status-error);'
                      }`}
                    >
                      {correct ? 'correct' : 'incorrect'}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p style="color: var(--text-tertiary); font-family: var(--font-mono); font-size: var(--type-label);">
            No prediction history available
          </p>
        )}
      </div>
    </div>
  );
}
