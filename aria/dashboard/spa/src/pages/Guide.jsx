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
// Shared styles
// ---------------------------------------------------------------------------

const DARK = '#111827';
const DARK_CARD = '#1a2332';
const DARK_BORDER = '#2a3545';
const CYAN = '#22d3ee';
const CYAN_DIM = '#0e7490';
const TEXT_PRIMARY = '#e5e7eb';
const TEXT_SECONDARY = '#9ca3af';
const TEXT_DIM = '#6b7280';

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function HeroSection() {
  return (
    <div class="relative rounded-md px-6 py-8 mb-6 overflow-hidden" style={`background: ${DARK}`}>
      <div class="absolute inset-0 pointer-events-none" style="opacity: 0.04">
        <div class="animate-scan-line" style={`width: 100%; height: 2px; background: ${CYAN}`} />
      </div>

      <div class="relative">
        <AriaLogo className="w-36 mb-3 animate-fade-in" color={CYAN} />
        <p class="text-xs font-medium mb-4 animate-fade-in delay-100" style={`color: #67e8f9; letter-spacing: 0.08em`}>
          ADAPTIVE RESIDENCE INTELLIGENCE ARCHITECTURE<span class="animate-blink" style={`color: ${CYAN}`}>_</span>
        </p>
        <p class="text-lg font-semibold leading-snug mb-3 animate-fade-in delay-200" style={`color: ${TEXT_PRIMARY}`}>
          Your home generates 22,000+ events every day.{' '}
          <span style={`color: ${CYAN}`}>ARIA learns what they mean.</span>
        </p>
        <p class="text-sm leading-relaxed animate-fade-in delay-300" style={`color: ${TEXT_DIM}`}>
          Local ML. No cloud. No subscriptions. Watches Home Assistant, learns patterns, generates automation suggestions.
        </p>
      </div>
    </div>
  );
}

function PrivacyNote() {
  return (
    <div
      class="rounded-md px-4 py-3 mb-6 animate-fade-in-up delay-400"
      style={`background: ${DARK_CARD}; border-left: 3px solid ${CYAN}`}
    >
      <p class="text-sm" style={`color: ${TEXT_SECONDARY}`}>
        <span class="font-semibold" style={`color: ${CYAN}`}>Fully local.</span>{' '}
        ML via scikit-learn, optional LLM via Ollama. No API calls, no telemetry, no cloud accounts.
      </p>
    </div>
  );
}

function JourneyTimeline() {
  return (
    <div class="rounded-md mb-6 overflow-hidden" style={`background: ${DARK_CARD}`}>
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider animate-fade-in-up" style={`color: ${CYAN}; letter-spacing: 0.06em`}>
          How ARIA Learns
        </h2>
      </div>

      <div class="px-5 pb-4">
        <div class="space-y-0 stagger-children">
          {JOURNEY_STEPS.map((step, i) => (
            <div
              key={i}
              class="flex gap-4 items-start py-3"
              style={i < JOURNEY_STEPS.length - 1 ? `border-bottom: 1px solid ${DARK_BORDER}` : ''}
            >
              <div
                class="shrink-0 w-6 h-6 flex items-center justify-center rounded-sm text-xs font-bold"
                style={`background: ${CYAN}; color: ${DARK}`}
              >
                {i + 1}
              </div>
              <div class="flex-1 min-w-0">
                <div class="flex items-baseline gap-2 mb-0.5">
                  <span class="text-xs font-mono" style={`color: ${CYAN_DIM}`}>{step.day}</span>
                  <span class="text-sm font-semibold" style={`color: ${TEXT_PRIMARY}`}>{step.title}</span>
                </div>
                <p class="text-sm leading-relaxed" style={`color: ${TEXT_DIM}`}>{step.desc}</p>
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
    <div class="rounded-md mb-6 overflow-hidden" style={`background: ${DARK_CARD}`}>
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider animate-fade-in-up" style={`color: ${CYAN}; letter-spacing: 0.06em`}>
          Key Concepts
        </h2>
      </div>

      <div class="px-5 pb-4 grid grid-cols-1 sm:grid-cols-2 gap-3 stagger-children">
        {CONCEPTS.map((c, i) => (
          <div key={i} class="rounded-sm px-3 py-2.5" style={`background: ${DARK}; border: 1px solid ${DARK_BORDER}`}>
            <h3 class="text-sm font-bold mb-0.5" style={`color: ${CYAN}`}>{c.term}</h3>
            <p class="text-xs leading-relaxed" style={`color: ${TEXT_DIM}`}>{c.def}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PageGuide() {
  return (
    <div class="rounded-md mb-6 overflow-hidden" style={`background: ${DARK_CARD}`}>
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider animate-fade-in-up" style={`color: ${CYAN}; letter-spacing: 0.06em`}>
          Pages
        </h2>
      </div>

      <div class="stagger-children">
        {PAGE_GUIDE.map((page, i) => (
          <a
            key={i}
            href={`#${page.path}`}
            class="flex items-baseline gap-3 px-5 py-2.5 transition-colors"
            style={`border-bottom: 1px solid ${DARK_BORDER}`}
            onMouseEnter={(e) => { e.currentTarget.style.background = DARK; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
          >
            <span class="text-sm font-semibold shrink-0" style={`color: ${CYAN}; min-width: 7rem`}>{page.title}</span>
            <span class="text-sm" style={`color: ${TEXT_DIM}`}>{page.desc}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

function FaqSection() {
  return (
    <div class="rounded-md mb-6 overflow-hidden" style={`background: ${DARK_CARD}`}>
      <div class="px-5 pt-4 pb-2">
        <h2 class="text-sm font-bold uppercase tracking-wider animate-fade-in-up" style={`color: ${CYAN}; letter-spacing: 0.06em`}>
          FAQ
        </h2>
      </div>

      <div class="px-5 pb-4 space-y-0 stagger-children">
        {FAQ.map((item, i) => (
          <div
            key={i}
            class="py-3"
            style={i < FAQ.length - 1 ? `border-bottom: 1px solid ${DARK_BORDER}` : ''}
          >
            <h3 class="text-sm font-semibold mb-1" style={`color: ${TEXT_PRIMARY}`}>{item.q}</h3>
            <p class="text-sm leading-relaxed" style={`color: ${TEXT_DIM}`}>{item.a}</p>
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
    <div class="max-w-3xl mx-auto">
      <HeroSection />
      <PrivacyNote />
      <JourneyTimeline />
      <KeyConcepts />
      <PageGuide />
      <FaqSection />

      <div class="text-center pb-6">
        <a
          href="#/"
          class="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold rounded-md transition-colors"
          style={`background: ${CYAN}; color: ${DARK}`}
        >
          Go to Dashboard
        </a>
      </div>
    </div>
  );
}
