import { useState, useEffect } from 'preact/hooks';
import { wsConnected, wsMessage } from '../store.js';
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
  { path: '/predictions', label: 'Predictions', icon: TrendingUpIcon },
  { path: '/patterns', label: 'Patterns', icon: LayersIcon },
  // Actions
  { section: 'Actions' },
  { path: '/shadow', label: 'Shadow Mode', icon: EyeIcon },
  { path: '/automations', label: 'Automations', icon: SettingsIcon },
  { path: '/settings', label: 'Settings', icon: SlidersIcon },
];

// Simple inline SVG icons
function GridIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function ZapIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function TrendingUpIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
      <polyline points="17 6 23 6 23 12" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z" />
      <line x1="9" y1="21" x2="15" y2="21" />
      <line x1="10" y1="19" x2="14" y2="19" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function SlidersIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" />
      <line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" />
      <line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" />
      <line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" />
      <line x1="17" y1="16" x2="23" y2="16" />
    </svg>
  );
}

function FilterIcon() {
  return (
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}

function getHashPath() {
  const hash = window.location.hash || '#/';
  const path = hash.replace(/^#/, '') || '/';
  const qIdx = path.indexOf('?');
  return qIdx >= 0 ? path.slice(0, qIdx) : path;
}

export default function Sidebar() {
  const [currentPath, setCurrentPath] = useState(getHashPath);

  useEffect(() => {
    function onHashChange() {
      setCurrentPath(getHashPath());
    }
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const connected = wsConnected.value;
  const statusText = wsMessage.value;

  return (
    <>
      {/* Desktop sidebar */}
      <nav class="hidden md:flex fixed left-0 top-0 bottom-0 w-60 bg-gray-900 flex-col z-30">
        {/* Brand */}
        <div class="px-5 py-5">
          <AriaLogo className="w-28" color="#22d3ee" />
          <p class="text-[10px] text-gray-500 mt-1.5 tracking-wide">Adaptive Residence Intelligence</p>
        </div>

        {/* Nav links */}
        <div class="flex-1 px-3 space-y-0.5 overflow-y-auto">
          {NAV_ITEMS.map((item, i) => {
            if (item.section) {
              return (
                <div key={item.section} class={`px-3 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-wider text-gray-600 ${i > 0 ? 'border-t border-gray-800 mt-2' : ''}`}>
                  {item.section}
                </div>
              );
            }
            const active = currentPath === item.path;
            return (
              <a
                key={item.path}
                href={`#${item.path}`}
                class={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  active
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                }`}
              >
                <item.icon />
                {item.label}
              </a>
            );
          })}
        </div>

        {/* Footer: Guide link + About + WS status */}
        <div class="px-3 py-3 border-t border-gray-800 space-y-2">
          <a
            href="#/guide"
            class={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              currentPath === '/guide'
                ? 'bg-gray-800 text-white'
                : 'text-cyan-400 hover:bg-gray-800 hover:text-cyan-300'
            }`}
          >
            <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
            </svg>
            How to Use ARIA
          </a>
          <div class="px-3 text-[10px] text-gray-600 leading-relaxed">
            <p class="font-semibold text-gray-500">ARIA v1.0.0</p>
            <p class="mt-0.5">Local ML + real-time monitoring for Home Assistant. No cloud required.</p>
          </div>
          <div class="flex items-center gap-2 px-3 text-xs text-gray-500">
            <span
              class={`inline-block w-2 h-2 rounded-full ${
                connected ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span class="truncate">{statusText || (connected ? 'Connected' : 'Disconnected')}</span>
          </div>
        </div>
      </nav>

      {/* Mobile bottom tab bar */}
      <nav class="md:hidden fixed bottom-0 left-0 right-0 bg-gray-900 border-t border-gray-800 z-30">
        <div class="flex justify-around items-center h-14">
          {NAV_ITEMS.filter((item) => !item.section).map((item) => {
            const active = currentPath === item.path;
            return (
              <a
                key={item.path}
                href={`#${item.path}`}
                class={`flex flex-col items-center justify-center p-1 ${
                  active ? 'text-white' : 'text-gray-500'
                }`}
                title={item.label}
              >
                <item.icon />
              </a>
            );
          })}
        </div>
      </nav>
    </>
  );
}
