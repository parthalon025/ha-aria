import { useState, useMemo } from 'preact/hooks';
import useSearch from '../hooks/useSearch.js';
import usePagination from '../hooks/usePagination.js';

/**
 * Reusable data table with search, sort, and pagination.
 *
 * @param {Object} props
 * @param {Array<{key: string, label: string, render?: Function, sortable?: boolean, className?: string}>} props.columns
 * @param {Array<Object>} props.data
 * @param {Array<string>} [props.searchFields] - Fields to search across
 * @param {number} [props.pageSize=50] - Rows per page
 * @param {string} [props.searchPlaceholder='Search...'] - Placeholder for search input
 * @param {*} [props.filterContent] - Optional JSX rendered between search and table
 */
export default function DataTable({
  columns,
  data,
  searchFields = [],
  pageSize = 50,
  searchPlaceholder = 'Search...',
  filterContent,
}) {
  const safeData = data || [];

  // Search
  const { searchTerm, setSearchTerm, filteredData } = useSearch(safeData, searchFields);

  // Sort state
  const [sort, setSort] = useState({ key: null, direction: 'asc' });

  function handleSort(colKey) {
    setSort((prev) => {
      if (prev.key === colKey) {
        return { key: colKey, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key: colKey, direction: 'asc' };
    });
  }

  // Sorted data
  const sortedData = useMemo(() => {
    if (!sort.key) return filteredData;

    return [...filteredData].sort((a, b) => {
      const aVal = a[sort.key];
      const bVal = b[sort.key];

      // Nulls sort to end
      if ((aVal === null || aVal === undefined) && (bVal === null || bVal === undefined)) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      let cmp;
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        cmp = aVal - bVal;
      } else {
        cmp = String(aVal).localeCompare(String(bVal), undefined, { numeric: true, sensitivity: 'base' });
      }

      return sort.direction === 'asc' ? cmp : -cmp;
    });
  }, [filteredData, sort.key, sort.direction]);

  // Pagination (operates on sorted+filtered data)
  const { pageData, nextPage, prevPage, totalPages, total, startIndex, endIndex, page } =
    usePagination(sortedData, pageSize);

  const hasFirst = page > 0;
  const hasLast = page < totalPages - 1;

  return (
    <div class="t-card" style="overflow: hidden;">
      {/* Toolbar: search + filters */}
      <div class="px-4 py-3" style="border-bottom: 1px solid var(--border-subtle);">
        <div class="flex flex-col sm:flex-row sm:items-center gap-3">
          {/* Search input */}
          <div class="relative flex-1 max-w-sm">
            <input
              type="text"
              value={searchTerm}
              onInput={(e) => setSearchTerm(e.target.value)}
              placeholder={searchPlaceholder}
              class="t-input w-full px-3 py-1.5 text-sm"
            />
            {/* Results count badge */}
            {searchTerm && (
              <span class="absolute right-2 top-1/2 -translate-y-1/2 text-xs" style="color: var(--text-tertiary);">
                {filteredData.length} / {safeData.length}
              </span>
            )}
          </div>
        </div>

        {/* Optional filter slots */}
        {filterContent && <div class="mt-3">{filterContent}</div>}
      </div>

      {/* Table */}
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr style="background: var(--bg-surface-raised);">
              {columns.map((col) => (
                <th
                  key={col.key}
                  class={`px-4 py-2 text-left ${
                    col.sortable ? 'cursor-pointer select-none' : ''
                  } ${col.className || ''}`}
                  style="color: var(--text-tertiary); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;"
                  onClick={col.sortable ? () => handleSort(col.key) : undefined}
                >
                  <span class="inline-flex items-center gap-1">
                    {col.label}
                    {col.sortable && <SortIndicator active={sort.key === col.key} direction={sort.direction} />}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} class="px-4 py-12 text-center text-sm" style="color: var(--text-tertiary);">
                  {safeData.length === 0 ? 'No data available' : 'No results match your filters'}
                </td>
              </tr>
            ) : (
              pageData.map((row, idx) => (
                <tr key={row.id || row.entity_id || idx} style="border-bottom: 1px solid var(--border-subtle);">
                  {columns.map((col) => (
                    <td key={col.key} class={`px-4 py-2 text-sm ${col.className || ''}`} style="color: var(--text-secondary);">
                      {col.render ? col.render(row[col.key], row) : (row[col.key] !== null && row[col.key] !== undefined ? row[col.key] : '\u2014')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination footer */}
      {total > 0 && (
        <div class="px-4 py-3 flex items-center justify-between" style="border-top: 1px solid var(--border-subtle);">
          <span class="text-sm" style="color: var(--text-tertiary);">
            Showing {startIndex}\u2013{endIndex} of {total}
          </span>
          <div class="flex gap-2">
            <button
              onClick={prevPage}
              disabled={!hasFirst}
              class="px-3 py-1 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              style="color: var(--text-secondary); background: var(--bg-surface-raised); border: 1px solid var(--border-primary); border-radius: var(--radius);"
            >
              Previous
            </button>
            <button
              onClick={nextPage}
              disabled={!hasLast}
              class="px-3 py-1 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              style="color: var(--text-secondary); background: var(--bg-surface-raised); border: 1px solid var(--border-primary); border-radius: var(--radius);"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Sort direction arrow indicator.
 */
function SortIndicator({ active, direction }) {
  if (!active) {
    return (
      <svg class="w-3 h-3" style="color: var(--text-tertiary);" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 2l3 4H3z" />
        <path d="M6 10l-3-4h6z" />
      </svg>
    );
  }

  if (direction === 'asc') {
    return (
      <svg class="w-3 h-3" style="color: var(--text-primary);" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 2l3 4H3z" />
      </svg>
    );
  }

  return (
    <svg class="w-3 h-3" style="color: var(--text-primary);" viewBox="0 0 12 12" fill="currentColor">
      <path d="M6 10l-3-4h6z" />
    </svg>
  );
}
