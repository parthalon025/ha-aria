import { useRef, useEffect } from 'preact/hooks';
import uPlot from 'uplot';

/**
 * uPlot wrapper for ARIA time-series charts.
 * Uses CSS variables for theming.
 *
 * @param {Object} props
 * @param {Array} props.data - uPlot data format: [timestamps[], series1[], series2[], ...]
 * @param {Array<{label: string, color: string, width?: number}>} props.series - Series config
 * @param {number} [props.height=120] - Chart height in px
 * @param {string} [props.className] - Additional CSS classes
 * @param {boolean} [props.compact=false] - Compact mode: no axes, grid, cursor, legend. Height defaults to 32px.
 */
export default function TimeChart({ data, series, height, className, compact = false }) {
  if (height === undefined) {
    height = compact ? 32 : 120;
  }
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Get computed CSS variables for theme-aware colors
    const styles = getComputedStyle(document.documentElement);
    const textColor = styles.getPropertyValue('--text-tertiary').trim();
    const gridColor = styles.getPropertyValue('--border-subtle').trim();
    const fontMono = styles.getPropertyValue('--font-mono').trim() || 'monospace';

    // Resolve CSS variables to computed values for canvas rendering
    function resolveColor(color) {
      if (color.startsWith('var(')) {
        return styles.getPropertyValue(color.slice(4, -1)).trim();
      }
      return color;
    }

    const opts = {
      width: containerRef.current.clientWidth,
      height,
      cursor: compact ? { show: false } : { show: true, drag: { x: false, y: false } },
      legend: { show: false },
      padding: compact ? [0, 0, 0, 0] : undefined,
      axes: compact
        ? [{ show: false }, { show: false }]
        : [
            {
              stroke: textColor,
              grid: { stroke: gridColor, width: 1 },
              font: `10px ${fontMono}`,
              ticks: { stroke: gridColor, width: 1 },
            },
            {
              stroke: textColor,
              grid: { stroke: gridColor, width: 1 },
              font: `10px ${fontMono}`,
              ticks: { stroke: gridColor, width: 1 },
              size: 50,
            },
          ],
      series: [
        {}, // x-axis (timestamps)
        ...series.map((s) => {
          const resolved = resolveColor(s.color);
          return {
            label: s.label,
            stroke: resolved,
            width: s.width || 2,
            fill: compact ? undefined : resolved + '15', // 15 = ~8% opacity hex
          };
        }),
      ],
    };

    // Destroy previous chart if exists
    if (chartRef.current) {
      chartRef.current.destroy();
    }

    chartRef.current = new uPlot(opts, data, containerRef.current);

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [data, series, height, compact]);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(() => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.setSize({
          width: containerRef.current.clientWidth,
          height,
        });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [height]);

  if (compact) {
    return (
      <div ref={containerRef} class={className || ''} role="img"
        aria-label={data && data.length > 1
          ? `Sparkline: ${series.map(s => s.label).join(', ')}`
          : 'Sparkline loading'} />
    );
  }

  return (
    <figure>
      <div ref={containerRef} class={className || ''} role="img"
        aria-label={data && data.length > 1
          ? `Chart: ${series.map(s => s.label).join(', ')}`
          : 'Chart loading'} />
      {data && data.length > 1 && (
        <figcaption class="sr-only">
          {series.map(s => s.label).join(', ')} — {data[0].length} data points
        </figcaption>
      )}
    </figure>
  );
}
