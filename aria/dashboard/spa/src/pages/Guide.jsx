import AriaLogo from '../components/AriaLogo.jsx';

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const JOURNEY_STEPS = [
  {
    day: 'Day 1',
    title: 'Collecting',
    desc: "ARIA connects to Home Assistant and starts watching. Every light switch, door sensor, thermostat change, and motion event gets recorded. Zero configuration.",
  },
  {
    day: 'Days 2–3',
    title: 'Baselines',
    desc: 'After 24 hours, ARIA knows what "normal" looks like. It builds hourly averages for every metric — lights on at 8pm, power draw at noon, motion patterns.',
  },
  {
    day: 'Days 3–7',
    title: 'ML Training',
    desc: 'Six ML algorithms start training — predicting which room activates next, when devices turn on, energy patterns. Models improve with every snapshot.',
  },
  {
    day: 'Week 2+',
    title: 'Shadow Mode',
    desc: 'ARIA predicts silently and scores itself. "What I predicted" vs "what happened." Accuracy climbs daily. Nothing changes in your home yet.',
  },
  {
    day: 'When ready',
    title: 'Suggestions',
    desc: 'Once accuracy gates pass, ARIA generates automation YAML. You review and approve each one. Nothing runs without your sign-off.',
  },
];

const PAGE_GUIDE = [
  {
    title: 'Home',
    path: '/',
    desc: 'Live pipeline flowchart — data flow, module health, status chips. The "YOU" cards show if anything needs attention.',
  },
  {
    title: 'Discovery',
    path: '/discovery',
    desc: 'Every entity, device, and area HA knows about. Search and filter. Unavailable devices are flagged.',
  },
  {
    title: 'Capabilities',
    path: '/capabilities',
    desc: 'Detected capabilities — lighting control, climate, presence detection. Helps ARIA scope which automations fit your setup.',
  },
  {
    title: 'Data Curation',
    path: '/data-curation',
    desc: 'Include or exclude entities from analysis. Filter out noisy sensors that confuse the models.',
  },
  {
    title: 'Intelligence',
    path: '/intelligence',
    desc: 'Baselines, predictions vs actuals, 30-day trends, correlations, LLM daily insights. The core learning view.',
  },
  {
    title: 'Predictions',
    path: '/predictions',
    desc: 'ML model outputs with confidence scores. Green = high, yellow = moderate, red = low. Improves daily.',
  },
  {
    title: 'Patterns',
    path: '/patterns',
    desc: 'Recurring event sequences — "weekdays 6:30am: kitchen light, coffee maker, bathroom light." Basis for automation suggestions.',
  },
  {
    title: 'Shadow Mode',
    path: '/shadow',
    desc: 'Prediction accuracy, high-confidence disagreements, pipeline stage progression: backtest, shadow, suggest, autonomous.',
  },
  {
    title: 'Automations',
    path: '/automations',
    desc: 'Ready-to-use YAML generated from detected patterns. Review, approve, or reject. ARIA never acts without permission.',
  },
  {
    title: 'Settings',
    path: '/settings',
    desc: 'Engine parameters — confidence thresholds, retraining schedules, feature weights. Defaults work well.',
  },
];

const FAQ = [
  {
    q: 'Does ARIA send my data anywhere?',
    a: 'No. Everything runs locally — scikit-learn for ML, optionally Ollama for LLM. Zero data leaves your network.',
  },
  {
    q: 'Will ARIA change my Home Assistant setup?',
    a: 'Never without permission. ARIA only reads from HA. Automation suggestions require manual review and approval.',
  },
  {
    q: "How long until it's useful?",
    a: "Baselines within 24 hours. Patterns in 2\u20133 days. ML predictions after a week. Solid automation suggestions within 2\u20133 weeks.",
  },
  {
    q: 'Why does everything show "Blocked"?',
    a: 'Normal for the first few days. Each module waits for upstream data \u2014 ML Training waits for snapshots, Patterns waits for logbook history. The Home page shows exactly what\u2019s needed.',
  },
  {
    q: 'What does "Shadow Mode" mean?',
    a: 'ARIA makes predictions silently and compares them to reality. Like a co-pilot building a track record. It never acts until accuracy gates are met and you approve.',
  },
  {
    q: 'Do I need to configure anything?',
    a: 'No. Auto-discovers devices, classifies entities, builds models. Data Curation lets you fine-tune if you want, but defaults work.',
  },
];

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function HeroSection() {
  return (
    <div class="relative rounded-md px-6 py-8 mb-8 overflow-hidden" style="background: #111827">
      {/* Scan line effect */}
      <div class="absolute inset-0 pointer-events-none" style="opacity: 0.04">
        <div class="animate-scan-line" style="width: 100%; height: 2px; background: #22d3ee" />
      </div>

      <div class="relative">
        <AriaLogo className="w-36 mb-3 animate-fade-in" color="#22d3ee" />
        <p class="text-xs font-medium mb-4 animate-fade-in delay-100" style="color: #67e8f9; letter-spacing: 0.08em">
          ADAPTIVE RESIDENCE INTELLIGENCE ARCHITECTURE<span class="animate-blink" style="color: #22d3ee">_</span>
        </p>
        <p class="text-lg font-semibold leading-snug mb-3 animate-fade-in delay-200" style="color: #e5e7eb">
          Your home generates 22,000+ events every day.{' '}
          <span style="color: #22d3ee">ARIA learns what they mean.</span>
        </p>
        <p class="text-sm leading-relaxed animate-fade-in delay-300" style="color: #6b7280">
          Watches Home Assistant, learns household patterns, generates automation
          suggestions. Local ML. No cloud. No subscriptions.
        </p>
      </div>
    </div>
  );
}

function JourneyTimeline() {
  return (
    <div class="mb-10">
      <h2 class="text-base font-semibold text-gray-900 mb-4 animate-fade-in-up">How ARIA Learns</h2>

      <div class="relative">
        {/* Connecting line */}
        <div class="hidden sm:block absolute w-px animate-grow-width delay-200" style="left: 0.75rem; top: 1rem; bottom: 1rem; background: #d1d5db; width: 1px" />

        <div class="space-y-4 stagger-children">
          {JOURNEY_STEPS.map((step, i) => (
            <div key={i} class="relative flex gap-4 items-start">
              {/* Timeline dot */}
              <div
                class="hidden sm:flex shrink-0 w-6 h-6 items-center justify-center rounded-sm text-xs font-bold"
                style="background: #22d3ee; color: #111827; z-index: 10; margin-top: 2px"
              >
                {i + 1}
              </div>
              {/* Card */}
              <div class="flex-1 border-l-2 sm:border-l-0 pl-4 sm:pl-0" style="border-color: #22d3ee">
                <div class="flex items-baseline gap-2 mb-1">
                  <span class="text-xs font-mono font-medium" style="color: #22d3ee">{step.day}</span>
                  <h3 class="text-sm font-semibold text-gray-900">{step.title}</h3>
                </div>
                <p class="text-sm text-gray-600 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PageGuide() {
  return (
    <div class="mb-10">
      <h2 class="text-base font-semibold text-gray-900 mb-4 animate-fade-in-up">Pages</h2>

      <div class="border border-gray-200 rounded-md divide-y divide-gray-200 stagger-children">
        {PAGE_GUIDE.map((page, i) => (
          <a
            key={i}
            href={`#${page.path}`}
            class="flex items-baseline gap-3 px-4 py-3 hover:bg-gray-50 transition-colors"
          >
            <span class="text-sm font-semibold shrink-0" style="color: #22d3ee; min-width: 6rem">{page.title}</span>
            <span class="text-sm text-gray-600">{page.desc}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

function FaqSection() {
  return (
    <div class="mb-10">
      <h2 class="text-base font-semibold text-gray-900 mb-4 animate-fade-in-up">FAQ</h2>

      <div class="space-y-3 stagger-children">
        {FAQ.map((item, i) => (
          <div key={i} class="border border-gray-200 rounded-md px-4 py-3">
            <h3 class="text-sm font-semibold text-gray-900 mb-1">{item.q}</h3>
            <p class="text-sm text-gray-600 leading-relaxed">{item.a}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function KeyConcepts() {
  const concepts = [
    { term: 'Entity', def: 'Anything HA tracks \u2014 light, sensor, lock, motion detector.' },
    { term: 'Baseline', def: 'Hourly averages of "normal." Flags when something is unusual for the time of day.' },
    { term: 'Shadow Prediction', def: 'A silent prediction scored against reality. Builds a track record before any action.' },
    { term: 'Pipeline', def: 'Data path: collection \u2192 learning \u2192 action. Each stage gates the next.' },
    { term: 'Curation', def: 'Choosing which entities ARIA watches. Filters noise so models learn from clean signals.' },
    { term: 'Accuracy Gate', def: 'Quality checkpoint. No suggestions until predictions hit minimum accuracy.' },
  ];

  return (
    <div class="mb-10">
      <h2 class="text-base font-semibold text-gray-900 mb-4 animate-fade-in-up">Key Concepts</h2>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 stagger-children">
        {concepts.map((c, i) => (
          <div key={i} class="border border-gray-200 rounded-md px-4 py-3">
            <h3 class="text-sm font-bold text-gray-900 mb-0.5">{c.term}</h3>
            <p class="text-xs text-gray-600 leading-relaxed">{c.def}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PrivacyNote() {
  return (
    <div class="rounded-md px-4 py-3 mb-8 animate-fade-in-up delay-400" style="border-left: 3px solid #22d3ee; background: #f9fafb">
      <p class="text-sm text-gray-700">
        <span class="font-semibold">Fully local.</span>{' '}
        ML via scikit-learn, optional LLM via Ollama. No API calls, no telemetry, no cloud accounts. Your data stays on your hardware.
      </p>
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
          class="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold rounded-md text-white transition-colors"
          style="background: #0891b2"
        >
          Go to Dashboard
        </a>
      </div>
    </div>
  );
}
