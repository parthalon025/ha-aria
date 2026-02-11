import { useState, useEffect, useRef, useMemo } from 'preact/hooks';

/**
 * Debounced search across specified fields of a data array.
 *
 * @param {Array<Object>} data - Array of objects to search through
 * @param {Array<string>} fields - Field names to search (case-insensitive includes)
 * @param {number} [debounceMs=200] - Debounce delay in milliseconds
 * @returns {{ searchTerm: string, setSearchTerm: Function, filteredData: Array<Object> }}
 */
export default function useSearch(data, fields, debounceMs = 200) {
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedTerm, setDebouncedTerm] = useState('');
  const timerRef = useRef(null);

  // Debounce the search term
  useEffect(() => {
    if (timerRef.current != null) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      setDebouncedTerm(searchTerm);
    }, debounceMs);

    return () => {
      if (timerRef.current != null) {
        clearTimeout(timerRef.current);
      }
    };
  }, [searchTerm, debounceMs]);

  const filteredData = useMemo(() => {
    if (!data || !Array.isArray(data)) return [];
    if (!debouncedTerm || !fields || fields.length === 0) return data;

    const needle = debouncedTerm.toLowerCase();
    return data.filter((row) =>
      fields.some((field) => {
        const val = row[field];
        if (val == null) return false;
        return String(val).toLowerCase().includes(needle);
      })
    );
  }, [data, debouncedTerm, fields]);

  return { searchTerm, setSearchTerm, filteredData };
}
