/**
 * Skeleton loading placeholders.
 * @param {{ type: 'stats' | 'table' | 'cards' | 'full' }} props
 */
export default function LoadingState({ type = 'full' }) {
  if (type === 'stats') {
    return (
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} class="bg-white rounded-lg shadow-sm p-4">
            <div class="h-8 w-16 bg-gray-200 rounded animate-pulse mb-2" />
            <div class="h-4 w-24 bg-gray-200 rounded animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  if (type === 'table') {
    return (
      <div class="bg-white rounded-lg shadow-sm overflow-hidden">
        {/* Header row */}
        <div class="flex gap-4 px-4 py-3 border-b border-gray-100">
          {[...Array(4)].map((_, i) => (
            <div key={i} class="h-4 bg-gray-200 rounded animate-pulse flex-1" />
          ))}
        </div>
        {/* Body rows */}
        {[...Array(6)].map((_, i) => (
          <div key={i} class="flex gap-4 px-4 py-3 border-b border-gray-50">
            {[...Array(4)].map((_, j) => (
              <div key={j} class="h-4 bg-gray-100 rounded animate-pulse flex-1" />
            ))}
          </div>
        ))}
      </div>
    );
  }

  if (type === 'cards') {
    return (
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} class="bg-white rounded-lg shadow-sm p-4">
            <div class="h-5 w-32 bg-gray-200 rounded animate-pulse mb-3" />
            <div class="h-4 w-full bg-gray-100 rounded animate-pulse mb-2" />
            <div class="h-4 w-3/4 bg-gray-100 rounded animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  // type === 'full'
  return (
    <div class="space-y-6">
      {/* Stats skeleton */}
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} class="bg-white rounded-lg shadow-sm p-4">
            <div class="h-8 w-16 bg-gray-200 rounded animate-pulse mb-2" />
            <div class="h-4 w-24 bg-gray-200 rounded animate-pulse" />
          </div>
        ))}
      </div>
      {/* Table skeleton */}
      <div class="bg-white rounded-lg shadow-sm overflow-hidden">
        {[...Array(4)].map((_, i) => (
          <div key={i} class="flex gap-4 px-4 py-3 border-b border-gray-50">
            {[...Array(4)].map((_, j) => (
              <div key={j} class="h-4 bg-gray-100 rounded animate-pulse flex-1" />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
