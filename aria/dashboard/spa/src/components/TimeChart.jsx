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
 */
export default function TimeChart({ data, series, height = 120, className }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !data || data.length === 0) return;

    // Get computed CSS variables for theme-aware colors
    const styles = getComputedStyle(document.documentElement);
    const textColor = styles.getPropertyValue('--text-tertiary').trim();
    const gridColor = styles.getPropertyValue('--border-subtle').trim();
    const fontMono = styles.getPropertyValue('--font-mono').trim() || 'monospace';

    const opts = {
      width: containerRef.current.clientWidth,
      height,
      cursor: { show: true, drag: { x: false, y: false } },
      legend: { show: false },
      axes: [
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
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width || 2,
          fill: s.color + '15', // 15 = ~8% opacity hex
        })),
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
  }, [data, series, height]);

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

  return <div ref={containerRef} class={className || ''} />;
}
