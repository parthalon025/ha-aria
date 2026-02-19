/**
 * PageBanner — ASCII pixel-art header for every page.
 * Renders "ARIA ✦ PAGE_NAME" in the same SVG pixel-art style as AriaLogo.
 * Provides visual consistency and wayfinding across all 12 dashboard pages.
 */
import { useMemo } from 'preact/hooks';

// 5-row pixel font — each char as array of binary row strings (1 = filled, 0 = empty)
// Matches the AriaLogo pixel proportions exactly for A, R, I
const FONT = {
  A: ['01110','11011','11111','11011','11011'],
  B: ['11110','11011','11110','11011','11110'],
  C: ['0111','1100','1100','1100','0111'],
  D: ['11110','11011','11011','11011','11110'],
  E: ['1111','1100','1110','1100','1111'],
  F: ['1111','1100','1110','1100','1100'],
  G: ['01110','11000','11011','11011','01110'],
  H: ['11011','11011','11111','11011','11011'],
  I: ['1','1','1','1','1'],
  J: ['0111','0011','0011','1011','0110'],
  K: ['11001','11010','11100','11010','11001'],
  L: ['1100','1100','1100','1100','1111'],
  M: ['10001','11011','10101','10001','10001'],
  N: ['10001','11001','10101','10011','10001'],
  O: ['01110','11011','11011','11011','01110'],
  P: ['11110','11011','11110','11000','11000'],
  Q: ['01110','11011','11011','11010','01101'],
  R: ['11111','11011','11110','11010','11001'],
  S: ['0111','1100','0110','0011','1110'],
  T: ['11111','00100','00100','00100','00100'],
  U: ['11011','11011','11011','11011','01110'],
  V: ['10001','10001','01010','01010','00100'],
  W: ['10001','10001','10101','10101','01010'],
  X: ['10001','01010','00100','01010','10001'],
  Y: ['10001','01010','00100','00100','00100'],
  Z: ['11111','00010','00100','01000','11111'],
};

// Cross/plus separator (3 wide, centered vertically)
const SEPARATOR = ['000','010','111','010','000'];

const UNIT = 10;
const GAP = 2;
const SIZE = UNIT - GAP;
const RX = 1;
const LETTER_GAP = 2;
const WORD_GAP = 3;

/** Convert a string into pixel positions [{col, row}] and total width in grid units. */
function layoutText(text) {
  let cursor = 0;
  const pixels = [];

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const pattern = FONT[ch];
    if (!pattern) continue;

    const width = pattern[0].length;
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < width; col++) {
        if (pattern[row][col] === '1') {
          pixels.push([cursor + col, row]);
        }
      }
    }
    cursor += width + LETTER_GAP;
  }

  return { pixels, endX: cursor > 0 ? cursor - LETTER_GAP : 0 };
}

/** Lay out the separator at a given x offset. */
function layoutSeparator(offsetX) {
  const pixels = [];
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 3; col++) {
      if (SEPARATOR[row][col] === '1') {
        pixels.push([offsetX + col, row]);
      }
    }
  }
  return { pixels, endX: offsetX + 3 };
}

export default function PageBanner({ page, subtitle }) {
  const layout = useMemo(() => {
    const aria = layoutText('ARIA');
    const sepStart = aria.endX + WORD_GAP;
    const sep = layoutSeparator(sepStart);
    const pageStart = sep.endX + WORD_GAP;
    const pg = layoutText(page);
    const pagePixels = pg.pixels.map(([c, r]) => [c + pageStart, r]);
    const totalWidth = (pageStart + pg.endX) * UNIT + UNIT;

    return { ariaPixels: aria.pixels, sepPixels: sep.pixels, pagePixels, totalWidth };
  }, [page]);

  return (
    <div class="page-banner-sh" style="margin-bottom: 1.5rem">
      <svg
        viewBox={`0 0 ${layout.totalWidth} ${5 * UNIT}`}
        style="height: 2rem; max-width: 100%; width: auto; display: block;"
        role="img"
        aria-label={`ARIA ${page}`}
      >
        {layout.ariaPixels.map(([c, r], i) => (
          <rect key={`a${i}`} x={c * UNIT + GAP / 2} y={r * UNIT + GAP / 2}
            width={SIZE} height={SIZE} rx={RX} fill="var(--accent)" />
        ))}
        {layout.sepPixels.map(([c, r], i) => (
          <rect key={`s${i}`} x={c * UNIT + GAP / 2} y={r * UNIT + GAP / 2}
            width={SIZE} height={SIZE} rx={RX} fill="var(--text-tertiary)" />
        ))}
        {layout.pagePixels.map(([c, r], i) => (
          <rect key={`p${i}`} x={c * UNIT + GAP / 2} y={r * UNIT + GAP / 2}
            width={SIZE} height={SIZE} rx={RX} fill="var(--text-primary)" />
        ))}
      </svg>
      {subtitle && (
        <p style="margin-top: 0.5rem; font-size: var(--type-label); color: var(--text-secondary)">{subtitle}</p>
      )}
    </div>
  );
}
