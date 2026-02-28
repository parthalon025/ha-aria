/**
 * sankeyLayout.js — Pure layout engine for the ARIA pipeline Sankey diagram.
 *
 * Computes node positions, cubic Bezier flow ribbon paths, and bus bar
 * coordinates for SVG rendering. No JSX, no DOM — pure data to coordinates.
 *
 * Exports:
 *   computeLayout({ nodes, links, width, expandedColumn }) → layout object
 *   computeTraceback(outputNodeId, links, nodeDetail) → Set<string>
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Column labels when collapsed into group nodes. */
const COLUMN_LABELS = ['SOURCES', 'INTAKE', 'PROCESSING', 'ENRICHMENT', 'OUTPUTS'];

// Node dimensions
const NODE_W = 140;          // individual node width
const NODE_H = 40;           // individual node height
const NODE_GAP = 8;          // vertical gap between stacked nodes
const GROUP_W = 180;         // collapsed group node width
const GROUP_H = 48;          // collapsed group node height

// Bus bar
const BUS_BAR_H = 28;       // bus bar height
const BUS_BAR_GAP = 20;     // gap between tallest column and bus bar

// Link ribbon
const LINK_MIN_W = 2;       // minimum ribbon half-width
const LINK_MAX_W = 20;      // maximum ribbon half-width

// Layout padding
const PAD_X = 40;            // horizontal padding from SVG edges
const PAD_TOP = 16;          // top padding above nodes

// ---------------------------------------------------------------------------
// computeLayout
// ---------------------------------------------------------------------------

/**
 * Compute full layout for the Sankey diagram.
 *
 * @param {Object} params
 * @param {Array}  params.nodes          - [{id, column (0-4), label, metricKey}, ...]
 * @param {Array}  params.links          - [{source, target, value, type}, ...]
 * @param {number} params.width          - SVG container width in pixels
 * @param {number} params.expandedColumn - Which column is expanded (0-4), or -1 for all collapsed
 * @returns {{ nodes: Array, links: Array, busBar: Object, svgHeight: number }}
 */
export function computeLayout({ nodes, links, width, expandedColumn = -1 }) {
  const usableWidth = width - PAD_X * 2;

  // Group nodes by column
  const columns = [[], [], [], [], []];
  for (const node of nodes) {
    const col = node.column;
    if (col >= 0 && col < 5) columns[col].push(node);
  }

  // Compute column x centers (evenly distributed across usable width)
  const colCount = 5;
  const colSpacing = usableWidth / (colCount - 1);
  const colX = Array.from({ length: colCount }, (_, i) => PAD_X + i * colSpacing);

  // Position nodes in each column
  const positionedNodes = [];
  const nodeById = new Map();
  let maxColumnBottom = PAD_TOP; // track tallest column for bus bar placement

  for (let col = 0; col < colCount; col++) {
    const children = columns[col];
    const isExpanded = expandedColumn === col;

    if (!isExpanded) {
      // Collapsed: single group node
      const w = GROUP_W;
      const h = GROUP_H;
      const x = colX[col] - w / 2;
      const y = PAD_TOP;
      const groupNode = {
        id: `group_${col}`,
        column: col,
        label: COLUMN_LABELS[col],
        isGroup: true,
        childIds: children.map((n) => n.id),
        children: children,
        x,
        y,
        w,
        h,
        cx: x + w / 2,
        cy: y + h / 2,
        bottomY: y + h,
      };
      positionedNodes.push(groupNode);
      nodeById.set(groupNode.id, groupNode);
      // Also map children IDs to the group so links can resolve
      for (const child of children) {
        nodeById.set(child.id, groupNode);
      }
      maxColumnBottom = Math.max(maxColumnBottom, groupNode.bottomY);
    } else {
      // Expanded: stack individual nodes vertically
      const w = NODE_W;
      let yOffset = PAD_TOP;
      for (const child of children) {
        const x = colX[col] - w / 2;
        const y = yOffset;
        const positioned = {
          ...child,
          isGroup: false,
          x,
          y,
          w,
          h: NODE_H,
          cx: x + w / 2,
          cy: y + NODE_H / 2,
          bottomY: y + NODE_H,
        };
        positionedNodes.push(positioned);
        nodeById.set(positioned.id, positioned);
        yOffset += NODE_H + NODE_GAP;
      }
      if (children.length > 0) {
        // bottomY of last node (subtract trailing gap)
        const lastBottom = PAD_TOP + children.length * NODE_H + (children.length - 1) * NODE_GAP;
        maxColumnBottom = Math.max(maxColumnBottom, lastBottom);
      }
    }
  }

  // Bus bar: horizontal strip below all nodes
  const busBar = {
    x: PAD_X,
    y: maxColumnBottom + BUS_BAR_GAP,
    width: usableWidth,
    height: BUS_BAR_H,
  };

  // Compute max link value for normalization
  const maxValue = links.reduce((max, l) => Math.max(max, l.value || 0), 0) || 1;

  // Build positioned links with SVG path strings
  const positionedLinks = [];
  for (const link of links) {
    const sourceNode = nodeById.get(link.source);
    const targetNode = nodeById.get(link.target);
    if (!sourceNode || !targetNode) continue;

    // Source point: right edge center of source node
    const x0 = sourceNode.x + sourceNode.w;
    const y0 = sourceNode.cy;

    // Target point: left edge center of target node
    const x1 = targetNode.x;
    const y1 = targetNode.cy;

    // Ribbon half-width proportional to value
    const normalized = (link.value || 0) / maxValue;
    const hw = LINK_MIN_W + normalized * (LINK_MAX_W - LINK_MIN_W);

    // Cubic Bezier ribbon path
    const midX = (x0 + x1) / 2;
    const path = [
      `M ${x0},${y0 - hw}`,
      `C ${midX},${y0 - hw} ${midX},${y1 - hw} ${x1},${y1 - hw}`,
      `L ${x1},${y1 + hw}`,
      `C ${midX},${y1 + hw} ${midX},${y0 + hw} ${x0},${y0 + hw}`,
      'Z',
    ].join(' ');

    positionedLinks.push({
      ...link,
      // Resolve to actual rendered node IDs (may be group_N)
      sourceId: sourceNode.id,
      targetId: targetNode.id,
      path,
      halfWidth: hw,
      x0,
      y0,
      x1,
      y1,
    });
  }

  const svgHeight = busBar.y + busBar.height + PAD_TOP;

  return {
    nodes: positionedNodes,
    links: positionedLinks,
    busBar,
    svgHeight,
  };
}

// ---------------------------------------------------------------------------
// computeTraceback
// ---------------------------------------------------------------------------

/**
 * BFS backward through links from an output node to find all ancestor nodes.
 *
 * @param {string}      outputNodeId - ID of the output node to trace from
 * @param {Array}       links        - All links [{source, target, ...}, ...]
 * @param {Object}      nodeDetail   - Node detail map (reserved for future use)
 * @returns {Set<string>} Set of node IDs on the path from sources to outputNodeId
 */
export function computeTraceback(outputNodeId, links, _nodeDetail) {
  const visited = new Set();
  const queue = [outputNodeId];
  visited.add(outputNodeId);

  // Build reverse adjacency: target → [source, ...]
  const reverseAdj = new Map();
  for (const link of links) {
    if (!reverseAdj.has(link.target)) {
      reverseAdj.set(link.target, []);
    }
    reverseAdj.get(link.target).push(link.source);
  }

  while (queue.length > 0) {
    const current = queue.shift();
    const parents = reverseAdj.get(current) || [];
    for (const parent of parents) {
      if (!visited.has(parent)) {
        visited.add(parent);
        queue.push(parent);
      }
    }
  }

  return visited;
}
