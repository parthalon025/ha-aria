// Shared utilities, constants, and layout components for Intelligence sub-sections

export function confidenceColor(conf) {
  if (conf === 'high') return 'background: var(--status-healthy-glow); color: var(--status-healthy);';
  if (conf === 'medium') return 'background: var(--status-warning-glow); color: var(--status-warning);';
  return 'background: var(--status-error-glow); color: var(--status-error);';
}

export function relativeTime(ts) {
  if (!ts) return '\u2014';
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function durationSince(ts) {
  if (!ts) return '';
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  const rmins = mins % 60;
  return rmins > 0 ? `${hrs}h ${rmins}m` : `${hrs}h`;
}

export function Section({ title, subtitle, children }) {
  return (
    <section class="space-y-3">
      <div class="t-section-header" style="padding-bottom: 8px;">
        <h2 class="text-lg font-bold" style="color: var(--text-primary)">{title}</h2>
        {subtitle && <p class="text-sm" style="color: var(--text-tertiary)">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}

export function Callout({ children }) {
  return (
    <div class="t-callout p-3 text-sm">
      {children}
    </div>
  );
}

export const DOMAIN_LABELS = {
  light: 'Lights',
  switch: 'Switches',
  binary_sensor: 'Sensors',
  lock: 'Locks',
  media_player: 'Media',
  cover: 'Covers',
  climate: 'Climate',
  vacuum: 'Vacuum',
  person: 'People',
  device_tracker: 'Trackers',
  fan: 'Fans',
  sensor: 'Power',
};

export const EVENT_ICONS = {
  arrive: '\u25B6', depart: '\u25C0', person: '\u25CF',
  unlock: '\u25CB', lock: '\u25CF',
  motion: '\u25B8', clear: '\u00B7',
  door: '\u25A0', window: '\u25A1',
  light_on: '\u25CF', light_off: '\u00B7',
  switch: '\u25C6', sensor: '\u25B8',
  media: '\u25B6', climate: '\u223F', cover: '\u25A1',
  default: '\u00B7',
};

export function describeEvent(evt) {
  const name = evt.friendly_name || evt.entity || '?';
  const domain = evt.domain || '';
  const dc = evt.device_class || '';
  const to = evt.to || '';
  const from = evt.from || '';

  // People
  if (domain === 'person' || domain === 'device_tracker') {
    if (to === 'home') return { text: `${name} arrived home`, icon: 'arrive' };
    if (from === 'home') return { text: `${name} left`, icon: 'depart' };
    return { text: `${name}: ${to}`, icon: 'person' };
  }

  // Locks
  if (domain === 'lock') {
    if (to === 'unlocked') return { text: `${name} unlocked`, icon: 'unlock' };
    if (to === 'locked') return { text: `${name} locked`, icon: 'lock' };
    return { text: `${name}: ${to}`, icon: 'lock' };
  }

  // Binary sensors by device class
  if (domain === 'binary_sensor') {
    if (dc === 'motion') {
      return to === 'on'
        ? { text: `Motion in ${name}`, icon: 'motion' }
        : { text: `${name} cleared`, icon: 'clear' };
    }
    if (dc === 'door') {
      return to === 'on'
        ? { text: `${name} opened`, icon: 'door' }
        : { text: `${name} closed`, icon: 'door' };
    }
    if (dc === 'window') {
      return to === 'on'
        ? { text: `${name} opened`, icon: 'window' }
        : { text: `${name} closed`, icon: 'window' };
    }
    if (dc === 'occupancy') {
      return to === 'on'
        ? { text: `${name} occupied`, icon: 'motion' }
        : { text: `${name} clear`, icon: 'clear' };
    }
    // Generic binary sensor
    return to === 'on'
      ? { text: `${name} active`, icon: 'sensor' }
      : { text: `${name} inactive`, icon: 'clear' };
  }

  // Lights
  if (domain === 'light') {
    return to === 'on'
      ? { text: `${name} on`, icon: 'light_on' }
      : { text: `${name} off`, icon: 'light_off' };
  }

  // Switches
  if (domain === 'switch') {
    return to === 'on'
      ? { text: `${name} on`, icon: 'switch' }
      : { text: `${name} off`, icon: 'switch' };
  }

  // Media
  if (domain === 'media_player') {
    if (to === 'playing') return { text: `${name} playing`, icon: 'media' };
    if (to === 'paused') return { text: `${name} paused`, icon: 'media' };
    if (to === 'idle' || to === 'off') return { text: `${name} stopped`, icon: 'media' };
    return { text: `${name}: ${to}`, icon: 'media' };
  }

  // Climate
  if (domain === 'climate') {
    return { text: `${name} set to ${to}`, icon: 'climate' };
  }

  // Cover
  if (domain === 'cover') {
    if (to === 'open') return { text: `${name} opened`, icon: 'cover' };
    if (to === 'closed') return { text: `${name} closed`, icon: 'cover' };
    return { text: `${name}: ${to}`, icon: 'cover' };
  }

  // Fallback
  return { text: `${name}: ${from} \u2192 ${to}`, icon: 'default' };
}
