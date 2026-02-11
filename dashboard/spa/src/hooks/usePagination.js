import { useState, useEffect, useMemo } from 'preact/hooks';

/**
 * Pagination over a data array.
 * Page is 0-indexed and resets to 0 when data length changes.
 *
 * @param {Array} data - Full data array to paginate
 * @param {number} [pageSize=50] - Items per page
 * @returns {{ page: number, pageData: Array, nextPage: Function, prevPage: Function, totalPages: number, total: number, startIndex: number, endIndex: number }}
 */
export default function usePagination(data, pageSize = 50) {
  const [page, setPage] = useState(0);
  const total = data ? data.length : 0;

  // Reset to page 0 when data length changes
  useEffect(() => {
    setPage(0);
  }, [total]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const pageData = useMemo(() => {
    if (!data || data.length === 0) return [];
    const start = page * pageSize;
    return data.slice(start, start + pageSize);
  }, [data, page, pageSize]);

  const startIndex = total === 0 ? 0 : page * pageSize + 1;
  const endIndex = Math.min((page + 1) * pageSize, total);

  function nextPage() {
    setPage((p) => Math.min(p + 1, totalPages - 1));
  }

  function prevPage() {
    setPage((p) => Math.max(p - 1, 0));
  }

  return { page, pageData, nextPage, prevPage, totalPages, total, startIndex, endIndex };
}
