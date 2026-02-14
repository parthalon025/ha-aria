import { Section } from './utils.jsx';
import { LEARNING_PHASES, LEARNING_PHASE_LABELS } from '../../constants.js';

export function LearningProgress({ maturity, shadowStage, shadowAccuracy }) {
  if (!maturity) return null;
  const idx = LEARNING_PHASES.indexOf(maturity.phase);
  const pct = Math.max(((idx + 1) / LEARNING_PHASES.length) * 100, 10);

  const whyText = idx < 2
    ? 'The system needs enough data to tell the difference between "normal Tuesday" and "something unusual." More days = better predictions.'
    : 'The system has enough data to predict your home\'s behavior and flag anomalies.';

  return (
    <Section title="Learning Progress" summary={maturity.phase}>
      <div class="t-frame p-4 space-y-4" data-label="learning">
        <div>
          <div class="flex justify-between text-xs mb-1" style="color: var(--text-tertiary)">
            {LEARNING_PHASE_LABELS.map((label, i) => {
              const style = i < idx
                ? 'color: var(--status-healthy)'
                : i === idx
                  ? 'color: var(--accent); font-weight: 700'
                  : '';
              return <span key={label} style={style}>{label}</span>;
            })}
          </div>
          <div class="h-3 rounded-full" style="background: var(--bg-inset)">
            <div
              class="h-3 rounded-full transition-all"
              style={`background: var(--accent); width: ${pct}%`}
            />
          </div>
        </div>

        <p class="text-sm" style="color: var(--text-secondary)">{maturity.description}</p>
        <p class="text-xs italic" style="color: var(--text-tertiary)">{whyText}</p>

        <div class="flex flex-wrap gap-3 text-sm">
          <span class="rounded px-2 py-1" style="background: var(--bg-surface-raised)">{maturity.days_of_data} day{maturity.days_of_data !== 1 ? 's' : ''} of data</span>
          <span class="rounded px-2 py-1" style="background: var(--bg-surface-raised)">{maturity.intraday_count} intraday snapshot{maturity.intraday_count !== 1 ? 's' : ''}</span>
          {maturity.first_date && (
            <span class="rounded px-2 py-1" style="background: var(--bg-surface-raised)">Since {maturity.first_date}</span>
          )}
        </div>

        {maturity.next_milestone && maturity.phase !== 'ml-active' && (
          <p class="text-xs" style="color: var(--text-tertiary)">Next: {maturity.next_milestone}</p>
        )}

        {shadowStage && (
          <div class="flex items-center gap-2 text-xs pt-1" style="color: var(--text-tertiary); border-top: 1px solid var(--border-subtle)">
            <span class="font-medium rounded-full px-2 py-0.5 capitalize" style="background: var(--accent-glow); color: var(--accent)">{shadowStage}</span>
            <span>Shadow engine{shadowAccuracy != null ? ` \u2014 ${Math.round(shadowAccuracy)}% accuracy` : ''}</span>
          </div>
        )}
      </div>
    </Section>
  );
}
