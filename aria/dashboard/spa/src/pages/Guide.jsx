import AriaLogo from '../components/AriaLogo.jsx';

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const JOURNEY_STEPS = [
  {
    day: 'Day 1',
    title: 'Collecting',
    desc: "Connects to Home Assistant and starts watching. Every light switch, door sensor, thermostat change, and motion event gets recorded. Zero configuration.",
  },
  {
    day: 'Days 2\u20133',
    title: 'Baselines',
    desc: 'After 24 hours, ARIA knows what "normal" looks like. Builds hourly averages for every metric \u2014 lights on at 8pm, power draw at noon, motion patterns.',
  },
  {
    day: 'Days 3\u20137',
    title: 'ML Training',
    desc: 'Six ML algorithms start training \u2014 predicting which room activates next, when devices turn on, energy patterns. Models improve with every snapshot.',
  },
  {
    day: 'Week 2+',
    title: 'Shadow Mode',
    desc: 'Predicts silently and scores itself. "What I predicted" vs "what happened." Accuracy climbs daily. Nothing changes in your home yet.',
  },
  {
    day: 'When ready',
    title: 'Suggestions',
    desc: 'Once accuracy gates pass, generates automation YAML. You review and approve each one. Nothing runs without your sign-off.',
  },
];

const PAGE_GUIDE = [
  { title: 'Home', path: '/', desc: 'Live pipeline flowchart \u2014 data flow, module health, status chips.' },
  { title: 'Discovery', path: '/discovery', desc: 'Every entity, device, and area HA knows about. Unavailable devices flagged.' },
  { title: 'Capabilities', path: '/capabilities', desc: 'Detected capabilities \u2014 lighting, climate, presence. Scopes automations to your setup.' },
  { title: 'Data Curation', path: '/data-curation', desc: 'Include or exclude entities. Filter noisy sensors that confuse models.' },
  { title: 'Intelligence', path: '/intelligence', desc: 'Baselines, predictions vs actuals, trends, correlations, daily LLM insights.' },
  { title: 'Predictions', path: '/predictions', desc: 'ML outputs with confidence scores. Green/yellow/red. Improves daily.' },
  { title: 'Patterns', path: '/patterns', desc: 'Recurring event sequences \u2014 basis for automation suggestions.' },
  { title: 'Shadow Mode', path: '/shadow', desc: 'Accuracy tracking, disagreements, pipeline stage progression.' },
  { title: 'Automations', path: '/automations', desc: 'Ready-to-use YAML from detected patterns. Review, approve, or reject.' },
  { title: 'Settings', path: '/settings', desc: 'Engine parameters \u2014 thresholds, schedules, weights. Defaults work well.' },
];

const FAQ = [
  {
    q: 'Does ARIA send my data anywhere?',
    a: 'No. scikit-learn for ML, optionally Ollama for LLM. Zero data leaves your network.',
  },
  {
    q: 'Will it change my HA setup?',
    a: 'Never without permission. Read-only from HA. Suggestions require manual approval.',
  },
  {
    q: "How long until it's useful?",
    a: 'Baselines within 24h. Patterns in 2\u20133 days. ML predictions after a week. Solid suggestions in 2\u20133 weeks.',
  },
  {
    q: 'Why does everything show "Blocked"?',
    a: 'Normal for the first few days. Each module waits for upstream data. Home page shows exactly what\u2019s needed.',
  },
  {
    q: 'What is Shadow Mode?',
    a: 'Silent predictions compared to reality. Like a co-pilot building a track record before acting.',
  },
  {
    q: 'Do I need to configure anything?',
    a: 'No. Auto-discovers devices, classifies entities, builds models. Data Curation lets you fine-tune if needed.',
  },
];

const CONCEPTS = [
  { term: 'Entity', def: 'Anything HA tracks \u2014 light, sensor, lock, motion detector.' },
  { term: 'Baseline', def: 'Hourly averages of "normal." Flags when something is unusual for that time of day.' },
  { term: 'Shadow Prediction', def: 'Silent prediction scored against reality. Builds track record before action.' },
  { term: 'Pipeline', def: 'Data path: collection \u2192 learning \u2192 action. Each stage gates the next.' },
  { term: 'Curation', def: 'Choosing which entities ARIA watches. Filters noise so models learn clean signals.' },
  { term: 'Accuracy Gate', def: 'Quality checkpoint. No suggestions until predictions hit minimum accuracy.' },
];

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function HeroSection() {
  return (
    <div class="relative overflow-hidden px-6 py-8 mb-6 t-terminal-bg" style="background: var(--bg-surface); border-radius: var(--radius);">
      <div class="absolute inset-0 pointer-events-none" style="opacity: 0.04">
        <div class="t1-scan-line" style="width: 100%; height: 2px; background: var(--accent);" />
      </div>

      <div class="relative">
        <AriaLogo className="w-36 mb-3 animate-fade-in" color="var(--accent)" />
        <p class="text-xs font-medium mb-4 animate-fade-in delay-100" style="color: var(--accent-dim); letter-spacing: 0.08em;">
          ADAPTIVE RESIDENCE INTELLIGENCE ARCHITECTURE<span class="animate-blink" style="color: var(--accent);">_</span>
        </p>
        <p class="text-lg font-semibold leading-snug mb-3 animate-fade-in delay-200" style="color: var(--text-primary);">
          Your home generates 22,000+ events every day.{' '}
          <span style="color: var(--accent);">ARIA learns what they mean.</span>
        </p>
        <p class="text-sm leading-relaxed animate-fade-in delay-300" style="color: var(--text-tertiary);">
          Local ML. No cloud. No subscriptions. Watches Home Assistant, learns patterns, generates automation suggestions.
        </p>
      </div>
    </div>
  );
}

function PrivacyNote() {
  return (
    <div
      class="px-4 py-3 mb-6"
      style="background: var(--bg-surface-raised); border-left: 3px solid var(--accent); border-radius: var(--radius);"
    >
      <p class="text-sm" style="color: var(--text-secondary);">
        <span class="font-semibold" style="color: var(--accent);">Fully local.</span>{' '}
        ML via scikit-learn, optional LLM via Ollama. No API calls, no telemetry, no cloud accounts.
      </p>
    </div>
  );
}

function JourneyTimeline() {
  return (
    <div class="mb-6 overflow-hidden" style="background: var(--bg-surface-raised); border-radius: var(--radius);">
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider" style="color: var(--accent); letter-spacing: 0.06em;">
          How ARIA Learns
        </h2>
      </div>

      <div class="px-5 pb-4">
        <div class="space-y-0 stagger-children">
          {JOURNEY_STEPS.map((step, i) => (
            <div
              key={i}
              class="flex gap-4 items-start py-3"
              style={i < JOURNEY_STEPS.length - 1 ? 'border-bottom: 1px solid var(--border-subtle);' : ''}
            >
              <div
                class="shrink-0 w-6 h-6 flex items-center justify-center text-xs font-bold"
                style="background: var(--accent); color: var(--bg-surface); border-radius: var(--radius);"
              >
                {i + 1}
              </div>
              <div class="flex-1 min-w-0">
                <div class="flex items-baseline gap-2 mb-0.5">
                  <span class="text-xs" style="color: var(--accent-dim); font-family: var(--font-mono);">{step.day}</span>
                  <span class="text-sm font-semibold" style="color: var(--text-primary);">{step.title}</span>
                </div>
                <p class="text-sm leading-relaxed" style="color: var(--text-tertiary);">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function KeyConcepts() {
  return (
    <div class="mb-6 overflow-hidden" style="background: var(--bg-surface-raised); border-radius: var(--radius);">
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider" style="color: var(--accent); letter-spacing: 0.06em;">
          Key Concepts
        </h2>
      </div>

      <div class="px-5 pb-4 grid grid-cols-1 sm:grid-cols-2 gap-3 stagger-children">
        {CONCEPTS.map((c, i) => (
          <div key={i} class="px-3 py-2.5" style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius);">
            <h3 class="text-sm font-bold mb-0.5" style="color: var(--accent);">{c.term}</h3>
            <p class="text-xs leading-relaxed" style="color: var(--text-tertiary);">{c.def}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PageGuide() {
  return (
    <div class="mb-6 overflow-hidden" style="background: var(--bg-surface-raised); border-radius: var(--radius);">
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider" style="color: var(--accent); letter-spacing: 0.06em;">
          Pages
        </h2>
      </div>

      <div class="stagger-children">
        {PAGE_GUIDE.map((page, i) => (
          <a
            key={i}
            href={`#${page.path}`}
            class="flex items-baseline gap-3 px-5 py-2.5 transition-colors"
            style="border-bottom: 1px solid var(--border-subtle);"
            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-surface)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
          >
            <span class="text-sm font-semibold shrink-0" style="color: var(--accent); min-width: 7rem;">{page.title}</span>
            <span class="text-sm" style="color: var(--text-tertiary);">{page.desc}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

function FaqSection() {
  return (
    <div class="mb-6 overflow-hidden" style="background: var(--bg-surface-raised); border-radius: var(--radius);">
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider" style="color: var(--accent); letter-spacing: 0.06em;">
          FAQ
        </h2>
      </div>

      <div class="px-5 pb-4 space-y-0 stagger-children">
        {FAQ.map((item, i) => (
          <div
            key={i}
            class="py-3"
            style={i < FAQ.length - 1 ? 'border-bottom: 1px solid var(--border-subtle);' : ''}
          >
            <h3 class="text-sm font-semibold mb-1" style="color: var(--text-primary);">{item.q}</h3>
            <p class="text-sm leading-relaxed" style="color: var(--text-tertiary);">{item.a}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Guide() {
  return (
    <div class="max-w-3xl mx-auto animate-page-enter">
      <HeroSection />
      <PrivacyNote />
      <JourneyTimeline />
      <KeyConcepts />
      <PageGuide />
      <FaqSection />

      <div class="text-center pb-6">
        <a
          href="#/"
          class="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold transition-colors"
          style="background: var(--accent); color: var(--bg-surface); border-radius: var(--radius);"
        >
          Go to Dashboard
        </a>
      </div>
    </div>
  );
}
