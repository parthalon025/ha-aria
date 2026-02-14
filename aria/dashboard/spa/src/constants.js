/**
 * Frontend constants — UI-only values extracted from components.
 * Backend-driven values should come from /api/config or /api/pipeline.
 */

// ── Display thresholds ──────────────────────────────────────────────────────

/** Baseline comparison: deltas below this % are labeled "typical". */
export const SIGNIFICANCE_PCT = 10;

/** Unavailable entity count above which a warning is shown. */
export const UNAVAILABLE_WARNING_THRESHOLD = 100;

// ── Accuracy color thresholds (percentage, 0-100) ──────────────────────────

export const ACCURACY_HEALTHY_PCT = 70;
export const ACCURACY_WARNING_PCT = 40;

/** Return CSS color style string for an accuracy percentage (0-100). */
export function accuracyColor(pct) {
  if (pct >= ACCURACY_HEALTHY_PCT) return 'color: var(--status-healthy)';
  if (pct >= ACCURACY_WARNING_PCT) return 'color: var(--status-warning)';
  return 'color: var(--status-error)';
}

/** Return CSS inline style string for confidence badge (0-1 scale). */
export function confidenceBadgeStyle(confidence) {
  if (confidence >= 0.7) return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (confidence >= 0.4) return 'background: var(--status-warning-glow); color: var(--status-warning);';
  return 'background: var(--status-error-glow); color: var(--status-error);';
}

// ── Pagination / display limits ─────────────────────────────────────────────

export const CURATION_GROUP_PREVIEW = 20;
export const CURATION_TABLE_MAX = 100;
export const SHADOW_PREDICTIONS_LIMIT = 20;
export const SHADOW_DISAGREEMENTS_LIMIT = 10;

// ── Time windows ────────────────────────────────────────────────────────────

export const ACTIVITY_TIMELINE_MS = 6 * 60 * 60 * 1000;
export const SWIM_LANE_MS = 60 * 60 * 1000;

// ── ML training threshold (matches config: incremental.data_window_days) ────

export const ML_TRAINING_MIN_DAYS = 14;

// ── Learning phases ─────────────────────────────────────────────────────────

export const LEARNING_PHASES = ['collecting', 'baselines', 'ml-training', 'ml-active'];
export const LEARNING_PHASE_LABELS = ['Collecting', 'Baselines', 'ML Training', 'ML Active'];
