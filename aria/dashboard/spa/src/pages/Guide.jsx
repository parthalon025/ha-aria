import AriaLogo from '../components/AriaLogo.jsx';

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const JOURNEY_STEPS = [
  {
    day: 'Day 1',
    title: 'Collecting',
    desc: "ARIA connects to Home Assistant and starts watching. Every light switch, door sensor, thermostat change, and motion event gets recorded. You don't need to configure anything.",
    bg: '#10b981', ringColor: '#a7f3d0',
    icon: '📡',
  },
  {
    day: 'Days 2–3',
    title: 'Baselines',
    desc: 'After 24 hours, ARIA knows what "normal" looks like for your home. It builds hourly averages for every metric — how many lights are typically on at 8pm, usual power draw at noon, typical motion patterns.',
    bg: '#3b82f6', ringColor: '#bfdbfe',
    icon: '📊',
  },
  {
    day: 'Days 3–7',
    title: 'ML Training',
    desc: 'With enough data, machine learning models start training. Six different algorithms learn to predict what your home will do next — which room activates, what time devices turn on, energy patterns.',
    bg: '#8b5cf6', ringColor: '#ddd6fe',
    icon: '🧠',
  },
  {
    day: 'Week 2+',
    title: 'Shadow Mode',
    desc: 'ARIA starts making predictions silently and scoring itself. It compares "what I predicted" vs "what actually happened." You can see its accuracy climb in real time. Nothing changes in your home yet.',
    bg: '#f59e0b', ringColor: '#fde68a',
    icon: '👻',
  },
  {
    day: 'When ready',
    title: 'Suggestions',
    desc: 'Once accuracy is high enough, ARIA generates automation suggestions as ready-to-use YAML. You review and approve each one before it touches your home. You are always in control.',
    bg: '#06b6d4', ringColor: '#a5f3fc',
    icon: '⚡',
  },
];

const PAGE_GUIDE = [
  {
    title: 'Home',
    path: '/',
    desc: 'The big picture. A live flowchart showing how data moves through ARIA — from collection to learning to action. Status chips show what\'s healthy, blocked, or waiting. The "YOU" cards at the bottom tell you if there\'s anything you need to do.',
    icon: '🏠',
  },
  {
    title: 'Discovery',
    path: '/discovery',
    desc: 'Everything Home Assistant knows about your home. Browse every entity (sensor, light, switch), device, and area. Search and filter to find anything. If a device shows "unavailable," it might need attention.',
    icon: '🔍',
  },
  {
    title: 'Capabilities',
    path: '/capabilities',
    desc: 'What your home can do. ARIA scans your devices and detects capabilities like "lighting control," "climate management," or "presence detection." This helps it understand which automations make sense for your setup.',
    icon: '⚡',
  },
  {
    title: 'Data Curation',
    path: '/data-curation',
    desc: 'Fine-tune what ARIA pays attention to. Some entities are noisy (like update sensors that flip constantly) and can confuse the models. This page lets you include or exclude specific entities from analysis.',
    icon: '🎯',
  },
  {
    title: 'Intelligence',
    path: '/intelligence',
    desc: 'The brain of ARIA. See baselines, predictions vs actuals, 30-day trends, cross-metric correlations, and LLM-generated daily insights. This is where you see ARIA actually learning your home.',
    icon: '🧠',
  },
  {
    title: 'Predictions',
    path: '/predictions',
    desc: 'What ARIA thinks will happen next. Each prediction has a confidence score. Green means high confidence, yellow is moderate, red is low. Predictions get better every day as more data comes in.',
    icon: '📋',
  },
  {
    title: 'Patterns',
    path: '/patterns',
    desc: 'Recurring sequences ARIA has detected. "Every weekday at 6:30am, the kitchen light turns on, then the coffee maker, then the bathroom light." These patterns become the basis for automation suggestions.',
    icon: '🔗',
  },
  {
    title: 'Shadow Mode',
    path: '/shadow',
    desc: "ARIA's training ground. It makes predictions silently and scores itself against reality. Watch accuracy improve over time. The pipeline advances through stages: backtest → shadow → suggest → autonomous.",
    icon: '👻',
  },
  {
    title: 'Automations',
    path: '/automations',
    desc: 'Ready-to-use automation suggestions generated from detected patterns. Each comes with YAML you can copy into Home Assistant. Review, approve, or reject each one. ARIA never changes your home without permission.',
    icon: '⚙️',
  },
  {
    title: 'Settings',
    path: '/settings',
    desc: "Tune ARIA's engine parameters. Adjust confidence thresholds, retraining schedules, and feature weights. Most users won't need to touch this — defaults work well. But if you want control, it's here.",
    icon: '🔧',
  },
];

const FAQ = [
  {
    q: 'Does ARIA send my data anywhere?',
    a: 'No. Everything runs locally on your machine. ARIA uses local ML models (scikit-learn) and optionally a local LLM (Ollama). Zero data leaves your network.',
  },
  {
    q: 'Will ARIA change my Home Assistant setup?',
    a: 'Never without your permission. ARIA only reads from Home Assistant — it watches events and states. When it generates automation suggestions, you must manually review and approve each one before anything changes.',
  },
  {
    q: "How long until it's useful?",
    a: "You'll see baselines within 24 hours. Patterns emerge in 2–3 days. ML predictions start after about a week. Shadow mode accuracy improves daily. Most homes see solid automation suggestions within 2–3 weeks.",
  },
  {
    q: 'Why does everything show "Blocked"?',
    a: "That's normal for the first few days. ARIA needs data before it can learn. \"Blocked\" means the module is waiting for upstream data — like ML Training waiting for enough snapshots. The Home page shows exactly what's needed.",
  },
  {
    q: 'What does "Shadow Mode" mean?',
    a: 'Think of it like a driving student with a co-pilot. ARIA makes predictions silently and compares them to reality, building a track record. It never acts on predictions until accuracy gates are met and you approve.',
  },
  {
    q: 'Do I need to configure anything?',
    a: 'Nope. ARIA auto-discovers your devices, classifies entities, and builds models with zero configuration. The Data Curation page lets you fine-tune if you want, but defaults work for most homes.',
  },
];

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function HeroSection() {
  return (
    <div
      class="relative overflow-hidden rounded-2xl px-8 py-12 mb-10"
      style="background: linear-gradient(135deg, #111827, #1f2937, #111827)"
    >
      {/* Background glow */}
      <div class="absolute inset-0" style="opacity: 0.12">
        <div class="absolute rounded-full" style="top: 1rem; left: 2rem; width: 16rem; height: 16rem; background: #06b6d4; filter: blur(60px)" />
        <div class="absolute rounded-full" style="bottom: 1rem; right: 3rem; width: 12rem; height: 12rem; background: #8b5cf6; filter: blur(60px)" />
      </div>

      <div class="relative" style="z-index: 10; max-width: 42rem">
        <AriaLogo className="w-40 mb-4" color="#22d3ee" />
        <p class="text-sm font-medium mb-3" style="color: #67e8f9; letter-spacing: 0.05em">Adaptive Residence Intelligence Architecture</p>
        <h2 class="text-2xl sm:text-3xl font-bold leading-tight mb-4" style="color: #ffffff">
          Your home generates 22,000+ events every day.
          <br />
          <span style="color: #22d3ee">ARIA learns what they mean.</span>
        </h2>
        <p class="text-base leading-relaxed" style="color: #9ca3af">
          ARIA watches your Home Assistant instance, learns your household patterns, and generates
          automation suggestions — all running locally on your hardware. No cloud. No subscriptions.
          No data leaves your network.
        </p>
      </div>
    </div>
  );
}

function JourneyTimeline() {
  return (
    <div class="mb-12">
      <h2 class="text-xl font-bold text-gray-900 mb-1">How ARIA Learns</h2>
      <p class="text-sm text-gray-500 mb-6">ARIA gets smarter every day. Here's what happens behind the scenes.</p>

      <div class="relative">
        {/* Connecting line */}
        <div class="hidden sm:block absolute w-0.5" style="left: 1.5rem; top: 2rem; bottom: 2rem; background: linear-gradient(to bottom, #6ee7b7, #c4b5fd, #67e8f9)" />

        <div class="space-y-6">
          {JOURNEY_STEPS.map((step, i) => (
            <div key={i} class="relative flex gap-5 items-start">
              {/* Timeline dot */}
              <div
                class="hidden sm:flex shrink-0 w-12 h-12 items-center justify-center rounded-full text-white text-xl shadow-lg"
                style={`background: ${step.bg}; box-shadow: 0 0 0 4px ${step.ringColor}; z-index: 10`}
              >
                {step.icon}
              </div>
              {/* Card */}
              <div class="flex-1 bg-white rounded-xl border border-gray-200 p-5 shadow-sm hover:shadow-md transition-shadow">
                <div class="flex items-center gap-3 mb-2">
                  <span class="sm:hidden text-2xl">{step.icon}</span>
                  <div>
                    <span
                      class="inline-block text-xs font-bold uppercase px-2 py-0.5 rounded-full text-white"
                      style={`background: ${step.bg}; letter-spacing: 0.05em`}
                    >
                      {step.day}
                    </span>
                    <h3 class="text-lg font-semibold text-gray-900 mt-1">{step.title}</h3>
                  </div>
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
    <div class="mb-12">
      <h2 class="text-xl font-bold text-gray-900 mb-1">What Each Page Does</h2>
      <p class="text-sm text-gray-500 mb-6">Click any card to jump to that page.</p>

      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {PAGE_GUIDE.map((page, i) => (
          <a
            key={i}
            href={`#${page.path}`}
            class="group bg-white rounded-xl border border-gray-200 p-5 shadow-sm hover:shadow-md hover:border-cyan-300 transition-all"
          >
            <div class="flex items-start gap-3">
              <span class="text-2xl shrink-0">{page.icon}</span>
              <div>
                <h3 class="text-base font-semibold text-gray-900 group-hover:text-cyan-600 transition-colors">
                  {page.title}
                </h3>
                <p class="text-sm text-gray-500 leading-relaxed mt-1">{page.desc}</p>
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function FaqSection() {
  return (
    <div class="mb-12">
      <h2 class="text-xl font-bold text-gray-900 mb-1">Common Questions</h2>
      <p class="text-sm text-gray-500 mb-6">Everything you need to know, no jargon.</p>

      <div class="space-y-4">
        {FAQ.map((item, i) => (
          <div key={i} class="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 class="text-base font-semibold text-gray-900 mb-2">{item.q}</h3>
            <p class="text-sm text-gray-600 leading-relaxed">{item.a}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function KeyConcepts() {
  const concepts = [
    {
      term: 'Entity',
      def: 'Anything Home Assistant tracks — a light switch, temperature sensor, door lock, motion detector. Your home has thousands of these.',
      icon: '💡',
    },
    {
      term: 'Baseline',
      def: 'What "normal" looks like. ARIA calculates hourly averages so it can tell you "lights are unusually high for 2am" or "power draw is normal for Tuesday evening."',
      icon: '📏',
    },
    {
      term: 'Shadow Prediction',
      def: 'A prediction ARIA makes silently. It predicts what will happen next, then waits to see if it was right. Like a weather forecast that only you can see.',
      icon: '👻',
    },
    {
      term: 'Pipeline',
      def: 'The path data takes through ARIA: collection → learning → action. Each stage has to complete before the next one can start. The Home page shows this visually.',
      icon: '🔄',
    },
    {
      term: 'Curation',
      def: 'Choosing which entities ARIA pays attention to. Some sensors are noisy and confuse the models. Curation filters them out so ARIA learns from clean signals.',
      icon: '🎯',
    },
    {
      term: 'Accuracy Gate',
      def: "A quality checkpoint. ARIA won't suggest automations until its predictions hit a minimum accuracy threshold. This prevents bad suggestions from reaching you.",
      icon: '🚦',
    },
  ];

  return (
    <div class="mb-12">
      <h2 class="text-xl font-bold text-gray-900 mb-1">Key Concepts</h2>
      <p class="text-sm text-gray-500 mb-6">Terms you'll see throughout the dashboard, explained simply.</p>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {concepts.map((c, i) => (
          <div key={i} class="rounded-xl border border-gray-200 p-5 shadow-sm" style="background: linear-gradient(135deg, #ffffff, #f9fafb)">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-xl">{c.icon}</span>
              <h3 class="text-base font-bold text-gray-900">{c.term}</h3>
            </div>
            <p class="text-sm text-gray-600 leading-relaxed">{c.def}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PrivacyBanner() {
  return (
    <div class="rounded-2xl border p-6 mb-10" style="background: linear-gradient(90deg, #ecfdf5, #ecfeff); border-color: #a7f3d0">
      <div class="flex items-start gap-4">
        <span class="text-3xl shrink-0">🔒</span>
        <div>
          <h3 class="text-lg font-bold mb-1" style="color: #064e3b">100% Local. 100% Private.</h3>
          <p class="text-sm leading-relaxed" style="color: #065f46">
            ARIA runs entirely on your hardware. Machine learning models train locally using scikit-learn.
            The optional LLM runs through Ollama on your own machine. No API calls to external services.
            No telemetry. No cloud accounts. Your home data stays in your home.
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Guide() {
  return (
    <div class="max-w-4xl mx-auto">
      <HeroSection />
      <PrivacyBanner />
      <JourneyTimeline />
      <KeyConcepts />
      <PageGuide />
      <FaqSection />

      {/* Footer CTA */}
      <div class="text-center pb-8">
        <a
          href="#/"
          class="inline-flex items-center gap-2 px-6 py-3 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all"
          style="background: linear-gradient(90deg, #06b6d4, #0891b2)"
        >
          {"Go to Dashboard →"}
        </a>
        <p class="text-xs text-gray-400 mt-3">ARIA is already learning. Check the Home page for current status.</p>
      </div>
    </div>
  );
}
