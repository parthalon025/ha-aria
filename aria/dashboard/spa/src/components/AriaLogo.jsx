/**
 * ARIA pixel-art logo — SVG recreation of the README ASCII block letters.
 * Each filled cell becomes a small rounded rect on a grid.
 */

// Pixel positions [col, row] for each letter
const PIXELS = [
  // A
  [1,0],[2,0],[3,0],
  [0,1],[1,1],[3,1],[4,1],
  [0,2],[1,2],[2,2],[3,2],[4,2],
  [0,3],[1,3],[3,3],[4,3],
  [0,4],[1,4],[3,4],[4,4],
  // R (offset 7)
  [7,0],[8,0],[9,0],[10,0],[11,0],
  [7,1],[8,1],[10,1],[11,1],
  [7,2],[8,2],[9,2],[10,2],
  [7,3],[8,3],[10,3],
  [7,4],[8,4],[11,4],
  // I (offset 14)
  [14,0],[14,1],[14,2],[14,3],[14,4],
  // A (offset 17)
  [18,0],[19,0],[20,0],
  [17,1],[18,1],[20,1],[21,1],
  [17,2],[18,2],[19,2],[20,2],[21,2],
  [17,3],[18,3],[20,3],[21,3],
  [17,4],[18,4],[20,4],[21,4],
];

const UNIT = 10;
const GAP = 2;
const SIZE = UNIT - GAP;
const RX = 1;

export default function AriaLogo({ className = '', color = 'currentColor' }) {
  return (
    <svg viewBox="0 0 220 50" class={className} fill={color} role="img" aria-label="ARIA">
      {PIXELS.map(([c, r], i) => (
        <rect key={i} x={c * UNIT + GAP / 2} y={r * UNIT + GAP / 2} width={SIZE} height={SIZE} rx={RX} />
      ))}
    </svg>
  );
}
