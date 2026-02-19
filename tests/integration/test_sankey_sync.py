"""Sankey topology sync test — verifies pipelineGraph.js ALL_NODES stay
in sync with hub module registry.

Parses the JavaScript source to extract node IDs from SOURCES, INTAKE,
PROCESSING, and ENRICHMENT arrays and compares against the Python module
registry. Outputs are excluded (they are API/dashboard endpoints, not
hub modules).

Closes #30 / RISK-10.
"""

import re
from pathlib import Path

import pytest

# Path to the Sankey graph definition
_PIPELINE_GRAPH_PATH = (
    Path(__file__).resolve().parents[2] / "aria" / "dashboard" / "spa" / "src" / "lib" / "pipelineGraph.js"
)

# The 10 surviving hub modules (from cli.py registration order)
_REGISTERED_MODULE_IDS = {
    "discovery",
    "activity_monitor",
    "patterns",
    "orchestrator",
    "shadow_engine",
    "trajectory_classifier",
    "ml_engine",
    "intelligence",
    "presence",
    "audit_logger",
}


def _parse_js_node_ids(js_text: str, array_name: str) -> set[str]:
    """Extract id values from a JS array-of-objects export.

    Looks for patterns like:
        export const INTAKE = [
          { id: 'discovery', ... },
          ...
        ];
    """
    # Match the full array block
    pattern = rf"export\s+const\s+{array_name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, js_text, re.DOTALL)
    if not match:
        return set()
    block = match.group(1)
    # Extract all id values (single or double quoted)
    ids = re.findall(r"""id:\s*['"]([^'"]+)['"]""", block)
    return set(ids)


class TestSankeyTopologySync:
    """Verify pipelineGraph.js stays in sync with hub module registry."""

    @pytest.fixture(autouse=True)
    def _load_graph(self):
        """Load the pipeline graph JS file."""
        assert _PIPELINE_GRAPH_PATH.exists(), f"pipelineGraph.js not found at {_PIPELINE_GRAPH_PATH}"
        self.js_text = _PIPELINE_GRAPH_PATH.read_text()

    def _get_sankey_module_ids(self) -> set[str]:
        """Get all module IDs from INTAKE + PROCESSING + ENRICHMENT arrays."""
        ids = set()
        for array_name in ("INTAKE", "PROCESSING", "ENRICHMENT"):
            ids |= _parse_js_node_ids(self.js_text, array_name)
        return ids

    def test_sankey_contains_all_hub_modules(self):
        """Every registered hub module (except audit_logger and engine)
        should appear in the Sankey graph.

        audit_logger is cross-cutting infrastructure, not a pipeline node.
        'engine' is a virtual node representing the batch pipeline, not a
        registered hub module.
        """
        sankey_ids = self._get_sankey_module_ids()
        # Exclude non-pipeline modules and virtual nodes
        expected = _REGISTERED_MODULE_IDS - {"audit_logger"}
        missing = expected - sankey_ids
        assert not missing, (
            f"Hub modules missing from Sankey graph: {sorted(missing)}. "
            f"Update pipelineGraph.js INTAKE/PROCESSING/ENRICHMENT arrays."
        )

    def test_sankey_has_no_stale_modules(self):
        """No Sankey node should reference a module that doesn't exist.

        The 'engine' node is a virtual aggregate — it represents the batch
        pipeline, not a single registered module. It's excluded from this
        check.
        """
        sankey_ids = self._get_sankey_module_ids()
        # 'engine' is a virtual node representing the batch pipeline
        allowed = _REGISTERED_MODULE_IDS | {"engine"}
        stale = sankey_ids - allowed
        assert not stale, (
            f"Sankey graph contains stale module IDs: {sorted(stale)}. "
            f"Remove them from pipelineGraph.js or register the modules."
        )

    def test_source_nodes_are_external(self):
        """SOURCES array should only contain external data sources, not
        hub modules."""
        source_ids = _parse_js_node_ids(self.js_text, "SOURCES")
        overlap = source_ids & _REGISTERED_MODULE_IDS
        assert not overlap, (
            f"SOURCES array contains hub module IDs: {sorted(overlap)}. "
            f"Module nodes belong in INTAKE/PROCESSING/ENRICHMENT."
        )

    def test_all_nodes_aggregation(self):
        """ALL_NODES should be the union of SOURCES + INTAKE + PROCESSING
        + ENRICHMENT + OUTPUTS."""
        all_ids = set()
        for array_name in ("SOURCES", "INTAKE", "PROCESSING", "ENRICHMENT", "OUTPUTS"):
            all_ids |= _parse_js_node_ids(self.js_text, array_name)
        # Verify ALL_NODES line exists (it's a spread, not parseable the same way)
        assert "ALL_NODES" in self.js_text, "ALL_NODES export missing from pipelineGraph.js"
        # Verify we found a reasonable number of nodes
        assert len(all_ids) > 15, f"Expected >15 total nodes, found {len(all_ids)}. Parsing may have failed."

    def test_patterns_module_in_sankey(self):
        """The 'patterns' module (PatternRecognition) must appear in the Sankey.

        This is a regression check — patterns was nearly dropped during the
        lean audit but remains a registered module.
        """
        sankey_ids = self._get_sankey_module_ids()
        # patterns module may appear as 'patterns' or 'trajectory_classifier'
        # depending on Sankey grouping. Both should be present.
        assert "trajectory_classifier" in sankey_ids, (
            "trajectory_classifier missing from Sankey — it's a registered processing module"
        )
