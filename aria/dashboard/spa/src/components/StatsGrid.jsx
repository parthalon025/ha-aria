/**
 * Responsive stats grid with ASCII bracket labels.
 * Phone: 2-col. Tablet: 3-col. Desktop: 4-col.
 */
export default function StatsGrid({ items }) {
  if (!items || items.length === 0) return null;

  return (
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {items.map((item, i) => (
        <div
          key={i}
          class="t-frame"
          style={`padding: 12px 16px;${item.warning ? ' border-left: 3px solid var(--status-warning);' : ''}`}
        >
          <div
            class="data-mono"
            style={`font-size: var(--type-data); font-weight: 600; color: ${item.warning ? 'var(--status-warning)' : 'var(--accent)'};`}
          >
            {item.value}
          </div>
          <div
            class="t-bracket"
            style="margin-top: 4px;"
          >
            {item.label}
          </div>
        </div>
      ))}
    </div>
  );
}
