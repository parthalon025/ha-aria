/**
 * Horizontal bar chart for domain breakdown, pure CSS via Tailwind.
 *
 * @param {{ data: Array<{ domain: string, count: number }>, total: number }} props
 */
export default function DomainChart({ data, total }) {
  if (!data || data.length === 0) return null;

  const maxCount = data[0]?.count || 1;

  return (
    <div class="space-y-2">
      {data.map((item) => {
        const pct = total > 0 ? ((item.count / total) * 100).toFixed(1) : 0;
        const barWidth = maxCount > 0 ? (item.count / maxCount) * 100 : 0;

        return (
          <div key={item.domain} class="flex items-center gap-2">
            <span class="w-28 text-sm text-gray-600 text-right truncate" title={item.domain}>
              {item.domain}
            </span>
            <div class="flex-1 h-6 bg-gray-100 rounded overflow-hidden">
              <div
                class="h-full bg-blue-500 rounded transition-all"
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <span class="w-20 text-sm text-gray-500 tabular-nums">
              {item.count} ({pct}%)
            </span>
          </div>
        );
      })}
    </div>
  );
}
