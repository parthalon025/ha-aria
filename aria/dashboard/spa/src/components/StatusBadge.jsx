/**
 * Terminal-style status badge with theme tokens.
 * @param {{ state: string }} props
 */
export default function StatusBadge({ state }) {
  const s = (state || '').toLowerCase();
  let statusClass;
  if (s === 'on' || s === 'home') statusClass = 't-status-healthy';
  else if (s === 'unavailable' || s === 'unknown') statusClass = 't-status-error';
  else if (state !== null && state !== undefined && state !== '' && !isNaN(Number(state))) statusClass = 't-status-healthy';
  else statusClass = 't-status-waiting';

  return (
    <span class={`t-status ${statusClass}`}>
      <span style="display: inline-block; width: 5px; height: 5px; border-radius: 50%; background: currentColor;" />
      {state}
    </span>
  );
}
