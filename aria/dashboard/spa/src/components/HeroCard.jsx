/**
 * Hero metric card — the single most important number on the page.
 * Large monospace value with cursor state indicator.
 */
export default function HeroCard({ value, label, unit, delta, warning, loading }) {
  const cursorClass = loading ? 'cursor-working' : 'cursor-active';

  return (
    <div
      class={`t-frame ${cursorClass}`}
      data-label={label}
      style={warning ? 'border-left: 3px solid var(--status-warning);' : ''}
    >
      <div class="flex items-baseline gap-2">
        <span
          class="data-mono"
          style={`font-size: var(--type-hero); font-weight: 600; color: ${warning ? 'var(--status-warning)' : 'var(--accent)'}; line-height: 1;`}
        >
          {value ?? '\u2014'}
        </span>
        {unit && (
          <span
            class="data-mono"
            style="font-size: var(--type-headline); color: var(--text-tertiary);"
          >
            {unit}
          </span>
        )}
      </div>
      {delta && (
        <div
          style="font-size: var(--type-label); color: var(--text-secondary); margin-top: 8px; font-family: var(--font-mono);"
        >
          {delta}
        </div>
      )}
    </div>
  );
}
