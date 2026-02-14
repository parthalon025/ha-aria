import { useState, useEffect } from 'preact/hooks';
import { wsConnected, wsMessage, theme, toggleTheme } from '../store.js';
import AriaLogo from './AriaLogo.jsx';

const NAV_ITEMS = [
  { path: '/', label: 'Home', icon: GridIcon },
  // Data Collection
  { section: 'Data Collection' },
  { path: '/discovery', label: 'Discovery', icon: SearchIcon },
  { path: '/capabilities', label: 'Capabilities', icon: ZapIcon },
  { path: '/data-curation', label: 'Data Curation', icon: FilterIcon },
  // Learning
  { section: 'Learning' },
  { path: '/intelligence', label: 'Intelligence', icon: BrainIcon },
  { path: '/ml-engine', label: 'ML Engine', icon: CpuIcon },
  { path: '/predictions', label: 'Predictions', icon: TrendingUpIcon },
  { path: '/patterns', label: 'Patterns', icon: LayersIcon },
  // Actions
  { section: 'Actions' },
  { path: '/shadow', label: 'Shadow Mode', icon: EyeIcon },
  { path: '/automations', label: 'Automations', icon: SettingsIcon },
  { path: '/settings', label: 'Settings', icon: SlidersIcon },
];

// The 5 primary tabs for the phone bottom bar
const PHONE_TABS = [
  { path: '/', label: 'Home', icon: GridIcon },
  { path: '/intelligence', label: 'Intel', icon: BrainIcon },
  { path: '/predictions', label: 'Predict', icon: TrendingUpIcon },
  { path: '/shadow', label: 'Shadow', icon: EyeIcon },
  { key: 'more', label: 'More', icon: MoreIcon },
];

// Items shown in the "More" sheet (everything not in PHONE_TABS)
const MORE_ITEMS = NAV_ITEMS.filter(
  (item) => !item.section && !['/', '/intelligence', '/predictions', '/shadow'].includes(item.path)
);

// Simple inline SVG icons
function GridIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function ZapIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function TrendingUpIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
      <polyline points="17 6 23 6 23 12" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z" />
      <line x1="9" y1="21" x2="15" y2="21" />
      <line x1="10" y1="19" x2="14" y2="19" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function SlidersIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
      <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
      <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
      <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" />
      <line x1="17" y1="16" x2="23" y2="16" />
    </svg>
  );
}

function CpuIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="15" x2="23" y2="15" />
      <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="15" x2="4" y2="15" />
    </svg>
  );
}

function FilterIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

function MoreIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <circle cx="12" cy="5" r="1.5" fill="currentColor" /><circle cx="12" cy="12" r="1.5" fill="currentColor" /><circle cx="12" cy="19" r="1.5" fill="currentColor" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function BookIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  );
}

function getHashPath() {
  const hash = window.location.hash || '#/';
  const path = hash.replace(/^#/, '') || '/';
  const qIdx = path.indexOf('?');
  return qIdx >= 0 ? path.slice(0, qIdx) : path;
}

// ─── Phone Bottom Tab Bar (< 640px) ───────────────────────────────────────────

function PhoneNav({ currentPath }) {
  const [moreOpen, setMoreOpen] = useState(false);
  const isDark = theme.value === 'dark';

  // Close "More" on navigation
  useEffect(() => {
    setMoreOpen(false);
  }, [currentPath]);

  return (
    <nav
      class="sm:hidden fixed bottom-0 left-0 right-0 z-40"
      aria-label="Primary navigation"
    >
      {/* More slide-up sheet */}
      {moreOpen && (
        <div
          class="fixed inset-0 z-40"
          style="background: rgba(0,0,0,0.5);"
          onClick={() => setMoreOpen(false)}
        >
          <div
            class="absolute bottom-14 left-0 right-0"
            style="background: var(--bg-surface); border-top: 1px solid var(--border-subtle); border-top-left-radius: 12px; border-top-right-radius: 12px; box-shadow: 0 -4px 20px rgba(0,0,0,0.15); max-height: 60vh; overflow-y: auto;"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Sheet header */}
            <div class="flex items-center justify-between" style="padding: 12px 16px; border-bottom: 1px solid var(--border-subtle);">
              <span style="font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">More</span>
              <button
                onClick={() => setMoreOpen(false)}
                aria-label="Close menu"
                style="color: var(--text-tertiary); padding: 4px; cursor: pointer; background: none; border: none;"
              >
                <CloseIcon />
              </button>
            </div>
            {/* More nav items */}
            <div style="padding: 8px 0;">
              {MORE_ITEMS.map((item) => {
                const active = currentPath === item.path;
                return (
                  <a
                    key={item.path}
                    href={`#${item.path}`}
                    class="flex items-center gap-3"
                    style={`padding: 12px 20px; font-size: 14px; font-weight: 500; text-decoration: none; ${active ? 'color: var(--accent); background: var(--bg-surface-raised);' : 'color: var(--text-secondary);'}`}
                  >
                    <item.icon />
                    {item.label}
                  </a>
                );
              })}
              {/* Guide link */}
              <a
                href="#/guide"
                class="flex items-center gap-3"
                style={`padding: 12px 20px; font-size: 14px; font-weight: 500; text-decoration: none; ${currentPath === '/guide' ? 'color: var(--accent); background: var(--bg-surface-raised);' : 'color: var(--accent);'}`}
              >
                <BookIcon />
                How to Use ARIA
              </a>
              {/* Theme toggle */}
              <div style="padding: 12px 20px; border-top: 1px solid var(--border-subtle); margin-top: 4px;">
                <button
                  onClick={toggleTheme}
                  class="flex items-center gap-2"
                  style="background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 6px 12px; font-size: 13px; cursor: pointer;"
                  aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {isDark ? <SunIcon /> : <MoonIcon />}
                  <span>{isDark ? 'Light mode' : 'Dark mode'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Bottom tab bar */}
      <div
        style="background: var(--bg-surface); border-top: 1px solid var(--border-subtle); min-height: 56px; padding-bottom: env(safe-area-inset-bottom);"
        class="flex justify-around items-center"
      >
        {PHONE_TABS.map((tab) => {
          const isMore = tab.key === 'more';
          const active = isMore ? moreOpen : currentPath === tab.path;
          const tabStyle = active
            ? 'color: var(--accent); border-top: 2px solid var(--accent); margin-top: -2px;'
            : 'color: var(--text-tertiary); border-top: 2px solid transparent; margin-top: -2px;';

          if (isMore) {
            return (
              <button
                key="more"
                class="flex flex-col items-center justify-center"
                style={`${tabStyle} min-width: 48px; min-height: 48px; background: none; border-left: none; border-right: none; border-bottom: none; cursor: pointer; font-size: 10px; font-weight: 500; gap: 2px;`}
                onClick={() => setMoreOpen(!moreOpen)}
                aria-label="More navigation options"
                aria-expanded={moreOpen}
              >
                <tab.icon />
                <span>{tab.label}</span>
              </button>
            );
          }
          return (
            <a
              key={tab.path}
              href={`#${tab.path}`}
              class="flex flex-col items-center justify-center"
              style={`${tabStyle} min-width: 48px; min-height: 48px; text-decoration: none; font-size: 10px; font-weight: 500; gap: 2px;`}
              aria-label={tab.label}
            >
              <tab.icon />
              <span>{tab.label}</span>
            </a>
          );
        })}
      </div>
    </nav>
  );
}

// ─── Tablet Icon Rail (640px - 1023px) ────────────────────────────────────────

function TabletNav({ currentPath }) {
  const [expanded, setExpanded] = useState(false);
  const connected = wsConnected.value;
  const isDark = theme.value === 'dark';

  // Collapse on navigation
  useEffect(() => {
    setExpanded(false);
  }, [currentPath]);

  const navItems = NAV_ITEMS.filter((item) => !item.section);
  const railWidth = expanded ? '240px' : '56px';

  return (
    <>
      {/* Overlay to close expanded rail */}
      {expanded && (
        <div
          class="hidden sm:block lg:hidden fixed inset-0"
          style="z-index: 29;"
          onClick={() => setExpanded(false)}
        />
      )}

      <nav
        class="hidden sm:flex lg:hidden fixed left-0 top-0 bottom-0 flex-col z-30"
        style={`width: ${railWidth}; background: var(--bg-surface); border-right: 1px solid var(--border-subtle); box-shadow: 2px 0 8px rgba(0,0,0,0.06); transition: width 0.2s ease; overflow: hidden;`}
        aria-label="Primary navigation"
      >
        {/* Menu toggle */}
        <button
          class="flex items-center justify-center"
          style={`width: 56px; height: 56px; flex-shrink: 0; background: none; border: none; color: var(--text-secondary); cursor: pointer;${expanded ? ' background: var(--bg-surface-raised);' : ''}`}
          onClick={() => setExpanded(!expanded)}
          aria-label={expanded ? 'Collapse navigation' : 'Expand navigation'}
          aria-expanded={expanded}
        >
          {expanded ? <CloseIcon /> : <MenuIcon />}
        </button>

        {/* Nav items */}
        <div class="flex-1 overflow-y-auto" style="padding: 4px 0;">
          {expanded ? (
            // Expanded: show sections + labels
            NAV_ITEMS.map((item, i) => {
              if (item.section) {
                return (
                  <div
                    key={item.section}
                    style={`padding: 0 16px; padding-top: 14px; padding-bottom: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); white-space: nowrap;${i > 0 ? ' border-top: 1px solid var(--border-subtle); margin-top: 6px;' : ''}`}
                  >
                    {item.section}
                  </div>
                );
              }
              const active = currentPath === item.path;
              return (
                <a
                  key={item.path}
                  href={`#${item.path}`}
                  class="flex items-center gap-3"
                  style={`padding: 8px 16px; font-size: 14px; font-weight: 500; text-decoration: none; white-space: nowrap; transition: background 0.15s ease; ${active ? 'color: var(--text-primary); background: var(--bg-surface-raised); border-left: 2px solid var(--accent); padding-left: 14px;' : 'color: var(--text-tertiary); border-left: 2px solid transparent; padding-left: 14px;'}`}
                  aria-label={item.label}
                >
                  <item.icon />
                  {item.label}
                </a>
              );
            })
          ) : (
            // Collapsed: icon-only
            navItems.map((item) => {
              const active = currentPath === item.path;
              return (
                <a
                  key={item.path}
                  href={`#${item.path}`}
                  class="flex items-center justify-center"
                  style={`width: 56px; height: 44px; text-decoration: none; transition: background 0.15s ease; ${active ? 'color: var(--accent); border-left: 2px solid var(--accent);' : 'color: var(--text-tertiary); border-left: 2px solid transparent;'}`}
                  title={item.label}
                  aria-label={item.label}
                >
                  <item.icon />
                </a>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div style="border-top: 1px solid var(--border-subtle); padding: 4px 0; flex-shrink: 0;">
          {/* Guide link */}
          <a
            href="#/guide"
            class="flex items-center justify-center"
            style={`width: 56px; height: 44px; text-decoration: none; ${currentPath === '/guide' ? 'color: var(--accent);' : 'color: var(--accent); opacity: 0.7;'}`}
            title="How to Use ARIA"
            aria-label="How to Use ARIA"
          >
            <BookIcon />
          </a>
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            class="flex items-center justify-center"
            style="width: 56px; height: 44px; background: none; border: none; color: var(--text-secondary); cursor: pointer;"
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDark ? <SunIcon /> : <MoonIcon />}
          </button>
          {/* WS status */}
          <div
            class="flex items-center justify-center"
            style="width: 56px; height: 28px;"
            title={connected ? 'Connected' : 'Disconnected'}
            aria-label={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
          >
            <span
              style={`display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`}
            />
          </div>
        </div>
      </nav>
    </>
  );
}

// ─── Desktop Full Sidebar (1024px+) ──────────────────────────────────────────

function DesktopNav({ currentPath }) {
  const connected = wsConnected.value;
  const statusText = wsMessage.value;
  const isDark = theme.value === 'dark';

  return (
    <nav
      class="hidden lg:flex fixed left-0 top-0 bottom-0 w-60 flex-col z-30 t-terminal-bg"
      style="background: var(--bg-surface); border-right: 1px solid var(--border-subtle); box-shadow: 2px 0 8px rgba(0,0,0,0.06);"
      aria-label="Primary navigation"
    >
      {/* Brand */}
      <div class="px-5 py-5">
        <AriaLogo className="w-28" color="var(--accent)" />
        <p style="font-size: 10px; color: var(--text-tertiary); margin-top: 6px; letter-spacing: 0.05em;">Adaptive Residence Intelligence</p>
      </div>

      {/* Nav links */}
      <div class="flex-1 px-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item, i) => {
          if (item.section) {
            return (
              <div
                key={item.section}
                style={`padding: 0 12px; padding-top: 16px; padding-bottom: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary);${i > 0 ? ' border-top: 1px solid var(--border-subtle); margin-top: 8px;' : ''}`}
              >
                {item.section}
              </div>
            );
          }
          const active = currentPath === item.path;
          return (
            <a
              key={item.path}
              href={`#${item.path}`}
              class="flex items-center gap-3 text-sm font-medium"
              style={active
                ? 'background: var(--bg-surface-raised); color: var(--text-primary); border-left: 2px solid var(--accent); padding: 8px 12px; border-radius: var(--radius); transition: background 0.15s ease, color 0.15s ease;'
                : 'color: var(--text-tertiary); padding: 8px 12px; border-left: 2px solid transparent; border-radius: var(--radius); transition: background 0.15s ease, color 0.15s ease;'
              }
              onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = 'var(--bg-surface-raised)'; e.currentTarget.style.color = 'var(--text-primary)'; }}
              onMouseLeave={(e) => { if (!active) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-tertiary)'; } }}
              aria-label={item.label}
            >
              <item.icon />
              {item.label}
            </a>
          );
        })}
      </div>

      {/* Footer: Guide link + Theme toggle + About + WS status */}
      <div style="padding: 12px; border-top: 1px solid var(--border-subtle);" class="space-y-2">
        <a
          href="#/guide"
          class="flex items-center gap-3 text-sm font-medium"
          style={currentPath === '/guide'
            ? 'background: var(--bg-surface-raised); color: var(--text-primary); border-left: 2px solid var(--accent); padding: 8px 12px; border-radius: var(--radius);'
            : 'color: var(--accent); padding: 8px 12px; border-left: 2px solid transparent; border-radius: var(--radius); transition: background 0.15s ease, color 0.15s ease;'
          }
          onMouseEnter={(e) => { if (currentPath !== '/guide') { e.currentTarget.style.background = 'var(--bg-surface-raised)'; } }}
          onMouseLeave={(e) => { if (currentPath !== '/guide') { e.currentTarget.style.background = 'transparent'; } }}
          aria-label="How to Use ARIA"
        >
          <BookIcon />
          How to Use ARIA
        </a>
        <div style="padding: 0 12px; font-size: 10px; color: var(--text-tertiary); line-height: 1.6;">
          <p style="font-weight: 600; color: var(--text-secondary);">ARIA v1.0.0</p>
          <p style="margin-top: 2px;">Local ML + real-time monitoring for Home Assistant. No cloud required.</p>
        </div>
        <button
          onClick={toggleTheme}
          style="background: var(--bg-surface-raised); color: var(--text-secondary); border: 1px solid var(--border-subtle); border-radius: var(--radius); padding: 4px 8px; margin: 0 12px; display: flex; align-items: center; gap: 6px; font-size: 0.75rem; cursor: pointer; transition: border-color 0.2s ease;"
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--border-primary)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; }}
        >
          {isDark ? <SunIcon /> : <MoonIcon />}
          <span>{isDark ? 'Light' : 'Dark'}</span>
        </button>
        <div class="flex items-center gap-2" style="padding: 0 12px; font-size: 0.75rem; color: var(--text-tertiary);">
          <span
            style={`display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${connected ? 'var(--status-healthy)' : 'var(--status-error)'};`}
            aria-label={connected ? 'WebSocket connected' : 'WebSocket disconnected'}
          />
          <span class="truncate">{statusText || (connected ? 'Connected' : 'Disconnected')}</span>
        </div>
      </div>
    </nav>
  );
}

// ─── Main Export ──────────────────────────────────────────────────────────────

export default function Sidebar() {
  const [currentPath, setCurrentPath] = useState(getHashPath);

  useEffect(() => {
    function onHashChange() {
      setCurrentPath(getHashPath());
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  return (
    <>
      <PhoneNav currentPath={currentPath} />
      <TabletNav currentPath={currentPath} />
      <DesktopNav currentPath={currentPath} />
    </>
  );
}
