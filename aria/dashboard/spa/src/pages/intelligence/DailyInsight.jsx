import { Section, Callout } from './utils.jsx';

export function DailyInsight({ insight }) {
  if (!insight) {
    return (
      <Section
        title="Daily Insight"
        subtitle="An AI-generated analysis of your home's patterns. Generated each night at 11:30 PM from the day's data."
      >
        <Callout>No insight report yet. The first report is generated after the first full pipeline run.</Callout>
      </Section>
    );
  }

  const lines = (insight.report || '').split('\n');

  return (
    <Section
      title="Daily Insight"
      subtitle="AI analysis of what happened yesterday and what to watch for. Generated nightly from your full data set."
    >
      <div class="bg-white rounded-lg shadow-sm p-4">
        <span class="inline-block bg-gray-100 rounded px-2 py-0.5 text-xs text-gray-500 mb-3">{insight.date}</span>
        <div class="prose prose-sm max-w-none text-gray-700 space-y-2">
          {lines.map((line, i) => {
            if (line.startsWith('###')) return <h3 key={i} class="text-sm font-bold text-gray-900 mt-3">{line.replace(/^###\s*/, '')}</h3>;
            if (line.trim() === '') return null;
            return <p key={i} class="text-sm leading-relaxed">{line}</p>;
          })}
        </div>
      </div>
    </Section>
  );
}
