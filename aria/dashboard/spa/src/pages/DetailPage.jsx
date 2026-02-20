/**
 * DetailPage — Generic detail route dispatcher.
 * Reads :type and :id from URL params, lazy-loads the correct detail renderer.
 * Route pattern: #/detail/:type/:id/:rest* (rest* handles composite IDs like correlation/entity1/entity2)
 */
import { useState, useEffect } from 'preact/hooks';
import PageBanner from '../components/PageBanner.jsx';
import Breadcrumb from '../components/Breadcrumb.jsx';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Maps each detail type to its parent OODA page for breadcrumb context. */
const PARENT_MAP = {
  anomaly:     { label: 'UNDERSTAND', href: '#/understand' },
  room:        { label: 'OBSERVE',    href: '#/observe' },
  entity:      { label: 'OBSERVE',    href: '#/observe' },
  prediction:  { label: 'UNDERSTAND', href: '#/understand' },
  suggestion:  { label: 'DECIDE',     href: '#/decide' },
  capability:  { label: 'CAPABILITIES', href: '#/capabilities' },
  model:       { label: 'ML ENGINE',  href: '#/ml-engine' },
  drift:       { label: 'ML ENGINE',  href: '#/ml-engine' },
  module:      { label: 'HOME',       href: '#/' },
  config:      { label: 'SETTINGS',   href: '#/settings' },
  curation:    { label: 'DATA CURATION', href: '#/data-curation' },
  correlation: { label: 'UNDERSTAND', href: '#/understand' },
  baseline:    { label: 'UNDERSTAND', href: '#/understand' },
};

/** All valid detail types. */
const VALID_TYPES = new Set(Object.keys(PARENT_MAP));

export default function DetailPage({ type, id, rest }) {
  const [Renderer, setRenderer] = useState(null);
  const [loadError, setLoadError] = useState(null);

  // Build the full ID (handles composite IDs via rest param)
  const fullId = rest ? `${id}/${rest}` : id;

  useEffect(() => {
    setRenderer(null);
    setLoadError(null);

    if (!VALID_TYPES.has(type)) {
      setLoadError(new Error(`Unknown detail type: "${type}"`));
      return;
    }

    // Dynamic import for code splitting — each renderer only loads when navigated to
    const filename = type.charAt(0).toUpperCase() + type.slice(1) + 'Detail';
    import(`../components/${filename}.jsx`)
      .then((mod) => {
        setRenderer(() => mod.default);
      })
      .catch((err) => {
        // Renderer doesn't exist yet — show a placeholder instead of crashing
        console.warn(`Detail renderer not found: ${filename}.jsx`, err);
        setLoadError(new Error(`Detail renderer for "${type}" is not yet implemented.`));
      });
  }, [type]);

  // Unknown type — show error immediately
  if (!VALID_TYPES.has(type)) {
    return (
      <div class="animate-page-enter">
        <PageBanner page="DETAIL" subtitle={`Unknown type: ${type}`} />
        <ErrorState error={`Unknown detail type: "${type}". Valid types: ${[...VALID_TYPES].join(', ')}`} />
      </div>
    );
  }

  const parent = PARENT_MAP[type];
  const typeLabel = type.toUpperCase();
  const breadcrumbs = [
    { label: 'HOME', href: '#/' },
    parent,
    { label: `${typeLabel}: ${decodeURIComponent(fullId).toUpperCase()}` },
  ];

  return (
    <div class="animate-page-enter">
      <PageBanner page={`DETAIL + ${typeLabel}`} subtitle={decodeURIComponent(fullId)} />
      <Breadcrumb segments={breadcrumbs} />

      {/* Back link — 48px touch target */}
      <a
        href={parent.href}
        class="inline-flex items-center gap-1 mb-4"
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--type-label)',
          color: 'var(--accent)',
          textDecoration: 'none',
          minHeight: '48px',
          minWidth: '48px',
          padding: '8px 0',
        }}
      >
        ← BACK
      </a>

      {/* Renderer area */}
      {loadError ? (
        <ErrorState error={loadError} />
      ) : Renderer ? (
        <Renderer id={fullId} type={type} />
      ) : (
        <LoadingState type="cards" />
      )}
    </div>
  );
}
