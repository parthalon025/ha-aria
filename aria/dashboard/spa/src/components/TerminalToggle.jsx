import { useState } from 'preact/hooks';

/**
 * Terminal-style ON/OFF toggle.
 *
 * @param {Object} props
 * @param {boolean} props.enabled - Current state
 * @param {Function} props.onChange - Called with new boolean value on toggle
 * @param {string} [props.label] - Optional label shown after the toggle
 */
export default function TerminalToggle({ enabled, onChange, label }) {
  const [flash, setFlash] = useState(false);

  function handleClick() {
    setFlash(true);
    setTimeout(() => setFlash(false), 200);
    onChange(!enabled);
  }

  return (
    <span
      class={`terminal-toggle ${flash ? 't2-tick-flash' : ''}`}
      onClick={handleClick}
      style={{
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
        cursor: 'pointer',
        color: enabled ? 'var(--accent)' : 'var(--text-tertiary)',
        userSelect: 'none',
      }}
      role="switch"
      aria-checked={enabled}
      tabIndex={0}
      onKeyDown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); handleClick(); } }}
    >
      {enabled ? '[ON ]' : '[OFF]'}
      {label && <span style={{ marginLeft: '0.5rem', color: 'var(--text-secondary)' }}>{label}</span>}
    </span>
  );
}
