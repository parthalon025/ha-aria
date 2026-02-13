/**
 * API client for HA Intelligence Hub.
 *
 * - baseUrl derived from window.location.origin
 * - fetchJson() wrapper with error handling
 * - Request deduplication: concurrent fetches to the same path share one promise
 */

const baseUrl = window.location.origin;

/** In-flight request map for deduplication. Key = path, value = pending Promise. */
const inflight = new Map();

/**
 * Fetch JSON from the hub API. Throws on non-200 responses.
 * Concurrent calls to the same path return the same promise (dedup).
 *
 * @param {string} path - API path (e.g. "/api/cache/entities")
 * @returns {Promise<any>} Parsed JSON response
 */
export function fetchJson(path) {
  if (inflight.has(path)) {
    return inflight.get(path);
  }

  const promise = fetch(`${baseUrl}${path}`)
    .then((res) => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return res.json();
    })
    .finally(() => {
      inflight.delete(path);
    });

  inflight.set(path, promise);
  return promise;
}

/**
 * PUT JSON to the hub API.
 * @param {string} path - API path
 * @param {any} body - Request body (will be JSON.stringify'd)
 * @returns {Promise<any>} Parsed JSON response
 */
export function putJson(path, body) {
  return fetch(`${baseUrl}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  });
}

/**
 * POST JSON to the hub API.
 * @param {string} path - API path
 * @param {any} body - Request body (will be JSON.stringify'd)
 * @returns {Promise<any>} Parsed JSON response
 */
export function postJson(path, body) {
  return fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  });
}

export { baseUrl };
