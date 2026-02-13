/**
 * Responsive stats grid.
 * @param {{ items: Array<{ label: string, value: string|number, warning?: boolean }> }} props
 */
export default function StatsGrid({ items }) {
  if (!items || items.length === 0) return null;

  return (
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      {items.map((item, i) => (
        <div
          key={i}
          class={`bg-white rounded-lg shadow-sm p-4 ${
            item.warning ? 'border-2 border-amber-500' : ''
          }`}
        >
          <div
            class={`text-2xl font-bold ${
              item.warning ? 'text-amber-500' : 'text-blue-500'
            }`}
          >
            {item.value}
          </div>
          <div class="text-sm text-gray-500 mt-1">{item.label}</div>
        </div>
      ))}
    </div>
  );
}
