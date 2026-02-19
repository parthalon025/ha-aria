/**
 * Clickable summary card for each OODA destination on the Home page.
 * Shows a brief metric and links to the full page.
 */
export default function OodaSummaryCard({ title, subtitle, metric, metricLabel, href, accentColor }) {
  const color = accentColor || 'var(--accent)';
  return (
    <a
      href={href}
      class="t-frame t-card-hover block"
      style="text-decoration: none; padding: 16px 20px; cursor: pointer;"
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
      {metric != null && (
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
