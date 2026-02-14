import { Section, Callout } from './utils.jsx';

function stripDomain(entityId) {
  const dot = entityId.indexOf('.');
  return dot >= 0 ? entityId.slice(dot + 1) : entityId;
}

function buildMatrix(correlations) {
  // Normalize data shape — support both object and array formats
  const pairs = correlations.map(c => ({
    a: c.entity_a || c[0],
    b: c.entity_b || c[1],
    strength: parseFloat(c.strength != null ? c.strength : c[2]) || 0,
  }));

  // Build lookup: entity -> average |correlation|
  const entityStrengths = {};
  for (const p of pairs) {
    for (const e of [p.a, p.b]) {
      if (!entityStrengths[e]) entityStrengths[e] = { sum: 0, count: 0 };
      entityStrengths[e].sum += Math.abs(p.strength);
      entityStrengths[e].count += 1;
    }
  }

  // Sort entities by strongest average |correlation| descending
  const entities = Object.keys(entityStrengths).sort((a, b) => {
    const avgA = entityStrengths[a].sum / entityStrengths[a].count;
    const avgB = entityStrengths[b].sum / entityStrengths[b].count;
    return avgB - avgA;
  });

  // Build pair lookup for O(1) access
  const pairMap = {};
  for (const p of pairs) {
    pairMap[p.a + '|' + p.b] = p.strength;
    pairMap[p.b + '|' + p.a] = p.strength;
  }

  return { entities, pairMap };
}

function cellBackground(strength) {
  const abs = Math.abs(strength);

  // Diagonal (self-correlation)
  if (strength === 1 && abs === 1) {
    return 'var(--bg-surface-raised)';
  }

  // Weak correlations — gray out
  if (abs <= 0.3) {
    return 'var(--bg-surface)';
  }

  // Scale opacity from 0.3 (at threshold) to 1.0 (at max)
  const opacity = (0.3 + (abs - 0.3) * (0.7 / 0.7)).toFixed(2);

  if (strength > 0) {
    return `color-mix(in srgb, var(--accent) ${Math.round(opacity * 100)}%, transparent)`;
  }
  return `color-mix(in srgb, var(--accent-purple) ${Math.round(opacity * 100)}%, transparent)`;
}

export function Correlations({ correlations }) {
  const hasData = correlations && correlations.length > 0;

  return (
    <Section
      title="Correlations"
      subtitle={hasData
        ? 'Devices that change together. Strong correlations suggest automation opportunities or shared failure modes.'
        : 'Devices that tend to change together \u2014 useful for creating automations or finding shared failure points.'
      }
      summary={hasData ? correlations.length + " pairs" : null}
    >
      {!hasData ? (
        <Callout>No correlations yet. Needs enough data to detect statistically reliable relationships between devices.</Callout>
      ) : (
        <MatrixHeatmap correlations={correlations} />
      )}
    </Section>
  );
}

function MatrixHeatmap({ correlations }) {
  const { entities, pairMap } = buildMatrix(correlations);
  const n = entities.length;

  // Grid: N+1 columns (row header + data cells), N+1 rows (col header + data cells)
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `auto repeat(${n}, minmax(2.5rem, 1fr))`,
    gridTemplateRows: `auto repeat(${n}, minmax(2rem, 1fr))`,
    gap: '1px',
    overflowX: 'auto',
  };

  return (
    <div class="t-frame overflow-x-auto" data-label="correlation-matrix">
      <div style={gridStyle}>
        {/* Top-left empty corner */}
        <div style={{ gridRow: 1, gridColumn: 1 }} />

        {/* Column headers — rotated 45 degrees */}
        {entities.map((entity, ci) => (
          <div
            key={'ch-' + ci}
            style={{
              gridRow: 1,
              gridColumn: ci + 2,
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'flex-start',
              height: '5rem',
              overflow: 'visible',
              paddingBottom: '0.25rem',
            }}
          >
            <span
              class="data-mono"
              style={{
                fontSize: 'var(--type-micro)',
                color: 'var(--text-secondary)',
                transform: 'rotate(-45deg)',
                transformOrigin: 'bottom left',
                whiteSpace: 'nowrap',
                display: 'inline-block',
              }}
            >
              {stripDomain(entity)}
            </span>
          </div>
        ))}

        {/* Data rows */}
        {entities.map((rowEntity, ri) => (
          <>
            {/* Row header */}
            <div
              key={'rh-' + ri}
              class="data-mono"
              style={{
                gridRow: ri + 2,
                gridColumn: 1,
                display: 'flex',
                alignItems: 'center',
                paddingRight: '0.5rem',
                fontSize: 'var(--type-micro)',
                color: 'var(--text-secondary)',
                whiteSpace: 'nowrap',
              }}
            >
              {stripDomain(rowEntity)}
            </div>

            {/* Data cells */}
            {entities.map((colEntity, ci) => {
              const isDiag = ri === ci;
              const strength = isDiag ? 1.0 : (pairMap[rowEntity + '|' + colEntity] ?? null);
              const hasValue = strength !== null;
              const abs = hasValue ? Math.abs(strength) : 0;
              const isWeak = hasValue && !isDiag && abs <= 0.3;

              return (
                <div
                  key={ri + '-' + ci}
                  style={{
                    gridRow: ri + 2,
                    gridColumn: ci + 2,
                    background: hasValue ? cellBackground(strength) : 'transparent',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '2px',
                    minHeight: '2rem',
                  }}
                  title={hasValue
                    ? `${stripDomain(rowEntity)} \u2194 ${stripDomain(colEntity)}: ${strength.toFixed(2)}`
                    : `${stripDomain(rowEntity)} \u2194 ${stripDomain(colEntity)}: no data`
                  }
                >
                  {hasValue && (
                    <span
                      style={{
                        fontSize: 'var(--type-micro)',
                        color: isWeak ? 'var(--text-tertiary)' : (abs > 0.6 ? 'var(--text-primary)' : 'var(--text-secondary)'),
                        fontVariantNumeric: 'tabular-nums',
                      }}
                    >
                      {isDiag ? '' : strength.toFixed(2)}
                    </span>
                  )}
                </div>
              );
            })}
          </>
        ))}
      </div>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginTop: '0.75rem',
          paddingTop: '0.5rem',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: 'var(--type-micro)',
          color: 'var(--text-tertiary)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <span style={{
            display: 'inline-block', width: '12px', height: '12px', borderRadius: '2px',
            background: 'color-mix(in srgb, var(--accent-purple) 70%, transparent)',
          }} />
          <span>Negative</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <span style={{
            display: 'inline-block', width: '12px', height: '12px', borderRadius: '2px',
            background: 'var(--bg-surface)',
          }} />
          <span>Weak (|r| &le; 0.3)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <span style={{
            display: 'inline-block', width: '12px', height: '12px', borderRadius: '2px',
            background: 'color-mix(in srgb, var(--accent) 70%, transparent)',
          }} />
          <span>Positive</span>
        </div>
      </div>
    </div>
  );
}
