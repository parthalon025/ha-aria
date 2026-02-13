/**
 * Reactive store for HA Intelligence Hub SPA.
 *
 * Uses @preact/signals for fine-grained reactivity.
 * - cacheStore: Map of category name → signal with { data, loading, error, stale, lastFetched }
 * - fetchCategory(): fetch cache data with stale-while-revalidate semantics
 * - connectWebSocket(): persistent WS connection with exponential backoff + ping
 */

import { signal } from '@preact/signals';
import { fetchJson } from './api.js';

// ---------------------------------------------------------------------------
// Cache store
// ---------------------------------------------------------------------------

/** @type {Map<string, import('@preact/signals').Signal>} */
const cacheStore = new Map();

/**
 * Get or create the signal for a cache category.
 * @param {string} name
 * @returns {import('@preact/signals').Signal}
 */
function getCategory(name) {
  if (!cacheStore.has(name)) {
    cacheStore.set(
      name,
      signal({ data: null, loading: false, error: null, stale: false, lastFetched: 0 })
    );
  }
  return cacheStore.get(name);
}

/**
 * Fetch a cache category from the API and update its signal.
 *
 * Stale-while-revalidate: if data already exists and the entry is stale,
 * the fetch happens in the background without flipping `loading` to true.
 *
 * @param {string} name - Cache category name (e.g. "entities")
 * @returns {Promise<void>}
 */
async function fetchCategory(name) {
  const sig = getCategory(name);
  const current = sig.value;

  // Skip if already loading or fetched within the last 2 seconds (prevents
  // double-fetch when mount effect and WS onopen both trigger simultaneously)
  if (current.loading) return;
  if (current.lastFetched && Date.now() - current.lastFetched < 2000) return;

  const hasData = current.data !== null;
  const isStaleRefresh = hasData && current.stale;

  // Only show loading spinner on first fetch (no cached data yet)
  if (!isStaleRefresh) {
    sig.value = { ...current, loading: true, error: null };
  }

  try {
    const result = await fetchJson(`/api/cache/${name}`);
    sig.value = {
      data: result,
      loading: false,
      error: null,
      stale: false,
      lastFetched: Date.now(),
    };
  } catch (err) {
    // 404 means the category doesn't exist yet (e.g. ML predictions before
    // first training run). Treat as empty data so pages show their "no data"
    // state instead of a scary red error banner.
    const is404 = err.message && err.message.startsWith('HTTP 404');
    sig.value = {
      ...sig.value,
      data: is404 ? { data: {} } : sig.value.data,
      loading: false,
      error: is404 ? null : err.message,
      stale: false,
      lastFetched: Date.now(),
    };
  }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

/** Whether the WebSocket is currently connected. */
const wsConnected = signal(false);

/** Last WebSocket status/error message (for UI display). */
const wsMessage = signal('');

/** @type {WebSocket|null} */
let ws = null;

/** Current backoff delay in ms. */
let backoff = 1000;
const BACKOFF_MAX = 30000;

/** Ping interval handle. */
let pingInterval = null;

/** Reconnect timeout handle. */
let reconnectTimer = null;

/**
 * Open a WebSocket connection to the hub.
 * Reconnects automatically with exponential backoff (1 s → 30 s cap).
 * Sends a ping every 30 s to keep the connection alive.
 *
 * On `cache_updated` messages the corresponding category signal is marked stale
 * and a background re-fetch is kicked off.
 *
 * On reconnect, ALL categories are marked stale and re-fetched.
 */
function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return; // already connected / connecting
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    backoff = 1000; // reset on successful connect
    wsConnected.value = true;
    wsMessage.value = 'Connected';

    // Mark everything stale on reconnect (we may have missed updates)
    markAllStale();

    // Keep-alive ping every 30 s
    clearInterval(pingInterval);
    pingInterval = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === 'cache_updated' && msg.data && msg.data.category) {
        const category = msg.data.category;
        const sig = getCategory(category);
        sig.value = { ...sig.value, stale: true };
        // Background re-fetch (fire and forget)
        fetchCategory(category);
      }
      // pong messages and others are silently consumed
    } catch {
      // Non-JSON or malformed — ignore
    }
  };

  ws.onclose = () => {
    wsConnected.value = false;
    wsMessage.value = `Disconnected — reconnecting in ${(backoff / 1000).toFixed(0)}s`;
    clearInterval(pingInterval);
    scheduleReconnect();
  };

  ws.onerror = () => {
    // onerror is always followed by onclose, so reconnect logic lives there.
    wsMessage.value = 'Connection error';
  };
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, backoff);
  backoff = Math.min(backoff * 2, BACKOFF_MAX);
}

/**
 * Close the WebSocket connection and cancel all timers.
 * Prevents reconnect attempts. Call from useEffect cleanup.
 */
function disconnectWebSocket() {
  clearInterval(pingInterval);
  pingInterval = null;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (ws) {
    ws.onclose = null; // prevent reconnect on intentional close
    ws.close();
    ws = null;
  }
  wsConnected.value = false;
  wsMessage.value = '';
}

/**
 * Mark every category in the cache store as stale and trigger background re-fetches.
 */
function markAllStale() {
  for (const [name, sig] of cacheStore) {
    sig.value = { ...sig.value, stale: true };
    fetchCategory(name);
  }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export {
  cacheStore,
  getCategory,
  fetchCategory,
  wsConnected,
  wsMessage,
  connectWebSocket,
  disconnectWebSocket,
};
