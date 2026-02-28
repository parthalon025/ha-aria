/**
 * Clickable summary card for each OODA destination on the Home page.
 * Shatters into fragments on click (SUPERHOT style), then navigates.
 */
import { useRef, useCallback, useState } from 'preact/hooks';

export default function OodaSummaryCard({ title, subtitle, metric, metricLabel, href, accentColor }) {
  const color = accentColor || 'var(--accent)';
  const cardRef = useRef(null);
  const [shattered, setShattered] = useState(false);

  const handleClick = useCallback((ev) => {
    ev.preventDefault();
    if (shattered) return;
    setShattered(true);

    // Navigate after shatter animation completes
    setTimeout(() => {
      window.location.hash = href.startsWith('#') ? href.slice(1) : href;
    }, 500);
  }, [href, shattered]);

  return (
    <a
      ref={cardRef}
      href={href}
      onClick={handleClick}
      class={`t-frame t-card-hover block${shattered ? ' sh-card-shatter' : ''}`}
      style="text-decoration: none; padding: 16px 20px; cursor: pointer; position: relative;"
    >
      <div class="flex items-center justify-between mb-2">
        <span
          class="text-xs font-semibold uppercase"
          style={`letter-spacing: 0.05em; color: ${color};`}
        >
          {title}
        </span>
        <span class="text-xs" style="color: var(--text-tertiary);">&rarr;</span>
      </div>
      {metric !== null && metric !== undefined && (
        <div class="flex items-baseline gap-2 mb-1">
          <span class="data-mono text-lg font-bold" style={`color: ${color};`}>
            {metric}
          </span>
          {metricLabel && (
            <span class="text-xs" style="color: var(--text-tertiary);">{metricLabel}</span>
          )}
        </div>
      )}
      {subtitle && (
        <p class="text-xs" style="color: var(--text-secondary);">{subtitle}</p>
      )}
    </a>
  );
}
