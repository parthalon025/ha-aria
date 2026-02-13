import { useEffect, useCallback } from 'preact/hooks';
import { getCategory, fetchCategory } from '../store.js';

/**
 * Hook to consume a cache category from the reactive store.
 *
 * On mount: fetches data if the category signal has no data yet.
 * On stale: auto-refetches when the signal's stale flag becomes true.
 *
 * Reading signal.value during render subscribes the component to updates
 * automatically via Preact signals integration.
 *
 * @param {string} name - Cache category name (e.g. "entities")
 * @returns {{ data: any, loading: boolean, error: string|null, refetch: Function }}
 */
export default function useCache(name) {
  const sig = getCategory(name);

  // Read .value to subscribe this component to signal changes
  const state = sig.value;

  const refetch = useCallback(() => fetchCategory(name), [name]);

  // Fetch on mount if no data, and auto-refetch when stale
  useEffect(() => {
    const current = sig.value;
    if (current.data === null && !current.loading) {
      fetchCategory(name);
    }
  }, [name]);

  useEffect(() => {
    if (state.stale && !state.loading) {
      fetchCategory(name);
    }
  }, [state.stale, state.loading, name]);

  return {
    data: state.data,
    loading: state.loading,
    error: state.error,
    refetch,
  };
}
