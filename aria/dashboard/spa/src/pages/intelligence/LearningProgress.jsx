import { Section } from './utils.jsx';

const PHASES = ['collecting', 'baselines', 'ml-training', 'ml-active'];
const PHASE_LABELS = ['Collecting', 'Baselines', 'ML Training', 'ML Active'];

export function LearningProgress({ maturity, shadowStage, shadowAccuracy }) {
  if (!maturity) return null;
  const idx = PHASES.indexOf(maturity.phase);
  const pct = Math.max(((idx + 1) / PHASES.length) * 100, 10);

  const whyText = idx < 2
    ? 'The system needs enough data to tell the difference between "normal Tuesday" and "something unusual." More days = better predictions.'
    : 'The system has enough data to predict your home\'s behavior and flag anomalies.';

  return (
    <Section title="Learning Progress">
      <div class="bg-white rounded-lg shadow-sm p-4 space-y-4">
        <div>
          <div class="flex justify-between text-xs text-gray-500 mb-1">
            {PHASE_LABELS.map((label, i) => (
              <span key={label} class={i <= idx ? 'font-bold text-blue-600' : ''}>{label}</span>
            ))}
          </div>
          <div class="h-3 rounded-full bg-gray-200">
            <div
              class="h-3 rounded-full bg-blue-500 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        <p class="text-sm text-gray-700">{maturity.description}</p>
        <p class="text-xs text-gray-400 italic">{whyText}</p>

        <div class="flex flex-wrap gap-3 text-sm">
          <span class="bg-gray-100 rounded px-2 py-1">{maturity.days_of_data} day{maturity.days_of_data !== 1 ? 's' : ''} of data</span>
          <span class="bg-gray-100 rounded px-2 py-1">{maturity.intraday_count} intraday snapshot{maturity.intraday_count !== 1 ? 's' : ''}</span>
          {maturity.first_date && (
            <span class="bg-gray-100 rounded px-2 py-1">Since {maturity.first_date}</span>
          )}
        </div>

        {maturity.next_milestone && maturity.phase !== 'ml-active' && (
          <p class="text-xs text-gray-500">Next: {maturity.next_milestone}</p>
        )}

        {shadowStage && (
          <div class="flex items-center gap-2 text-xs text-gray-500 pt-1 border-t border-gray-100">
            <span class="font-medium bg-blue-100 text-blue-700 rounded-full px-2 py-0.5 capitalize">{shadowStage}</span>
            <span>Shadow engine{shadowAccuracy != null ? ` \u2014 ${Math.round(shadowAccuracy)}% accuracy` : ''}</span>
          </div>
        )}
      </div>
    </Section>
  );
}
