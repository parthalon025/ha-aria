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
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

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
    <div class="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Toolbar: search + filters */}
      <div class="px-4 py-3 border-b border-gray-100">
        <div class="flex flex-col sm:flex-row sm:items-center gap-3">
          {/* Search input */}
          <div class="relative flex-1 max-w-sm">
            <input
              type="text"
              value={searchTerm}
              onInput={(e) => setSearchTerm(e.target.value)}
              placeholder={searchPlaceholder}
              class="w-full bg-white border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {/* Results count badge */}
            {searchTerm && (
              <span class="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-gray-400">
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
            <tr class="bg-gray-50">
              {columns.map((col) => (
                <th
                  key={col.key}
                  class={`px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                    col.sortable ? 'cursor-pointer select-none hover:text-gray-700' : ''
                  } ${col.className || ''}`}
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
                <td colSpan={columns.length} class="px-4 py-12 text-center text-gray-400 text-sm">
                  {safeData.length === 0 ? 'No data available' : 'No results match your filters'}
                </td>
              </tr>
            ) : (
              pageData.map((row, idx) => (
                <tr key={row.id || row.entity_id || idx} class="hover:bg-gray-50 border-b border-gray-100">
                  {columns.map((col) => (
                    <td key={col.key} class={`px-4 py-2 text-sm text-gray-700 ${col.className || ''}`}>
                      {col.render ? col.render(row[col.key], row) : (row[col.key] != null ? row[col.key] : '\u2014')}
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
        <div class="px-4 py-3 border-t border-gray-100 flex items-center justify-between">
          <span class="text-sm text-gray-500">
            Showing {startIndex}\u2013{endIndex} of {total}
          </span>
          <div class="flex gap-2">
            <button
              onClick={prevPage}
              disabled={!hasFirst}
              class="px-3 py-1 text-sm bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              onClick={nextPage}
              disabled={!hasLast}
              class="px-3 py-1 text-sm bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
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
      <svg class="w-3 h-3 text-gray-300" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 2l3 4H3z" />
        <path d="M6 10l-3-4h6z" />
      </svg>
    );
  }

  if (direction === 'asc') {
    return (
      <svg class="w-3 h-3 text-gray-700" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 2l3 4H3z" />
      </svg>
    );
  }

  return (
    <svg class="w-3 h-3 text-gray-700" viewBox="0 0 12 12" fill="currentColor">
      <path d="M6 10l-3-4h6z" />
    </svg>
  );
}
