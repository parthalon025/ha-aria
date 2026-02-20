/**
 * Terminal-style breadcrumb navigation.
 * segments: [{ label: 'HOME', href: '#/' }, { label: 'OBSERVE', href: '#/observe' }, ...]
 * Last segment is current (accent color, not clickable).
 */
export default function Breadcrumb({ segments }) {
  return (
    <nav class="flex items-center gap-1 text-xs mb-2" style="font-family: var(--font-mono);">
      {segments.map((seg, i) => {
        const isLast = i === segments.length - 1;
        return (
          <span key={i} class="flex items-center gap-1">
            {i > 0 && <span style="color: var(--text-tertiary)">/</span>}
            {isLast ? (
              <span style="color: var(--accent)">{seg.label}</span>
            ) : (
              <a
                href={seg.href}
                class="clickable-data"
                style="color: var(--text-tertiary); text-decoration: none;"
              >
                {seg.label}
              </a>
            )}
          </span>
        );
      })}
    </nav>
  );
}
