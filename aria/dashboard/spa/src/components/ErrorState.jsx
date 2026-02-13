/**
 * Error alert with retry button.
 * @param {{ error: string|Error, onRetry?: function }} props
 */
export default function ErrorState({ error, onRetry }) {
  const message = error instanceof Error ? error.message : String(error || 'Unknown error');

  return (
    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
      <div class="flex items-start gap-3">
        <svg class="w-5 h-5 text-red-500 mt-0.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <div class="flex-1">
          <p class="text-sm text-red-700">{message}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            class="px-3 py-1 text-sm font-medium text-red-700 bg-red-100 hover:bg-red-200 rounded-md transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
