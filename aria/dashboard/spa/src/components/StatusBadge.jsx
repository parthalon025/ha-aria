/**
 * Status pill badge.
 * @param {{ state: string }} props
 */
export default function StatusBadge({ state }) {
  const s = (state || '').toLowerCase();

  let colorClass;
  if (s === 'on' || s === 'home') {
    colorClass = 'bg-green-500';
  } else if (s === 'off' || s === 'not_home') {
    colorClass = 'bg-gray-400';
  } else if (s === 'unavailable' || s === 'unknown') {
    colorClass = 'bg-red-500';
  } else if (state != null && state !== '' && !isNaN(Number(state))) {
    colorClass = 'bg-blue-500';
  } else {
    colorClass = 'bg-gray-400';
  }

  return (
    <span
      class={`inline-block px-2 py-0.5 rounded-full text-xs font-medium text-white ${colorClass}`}
    >
      {state}
    </span>
  );
}
