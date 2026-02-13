import { useMemo } from 'preact/hooks';

/**
 * Memoized computation that recomputes only when dependencies change.
 *
 * @param {Function} computeFn - Function that produces the computed value
 * @param {Array} deps - Dependency array (same semantics as useMemo deps)
 * @returns {*} The memoized computed result
 */
export default function useComputed(computeFn, deps) {
  return useMemo(computeFn, deps);
}
