"""Capability dataclass and registry for ARIA.

Defines the Capability frozen dataclass that each hub module and engine command
uses to declare what it provides, and a CapabilityRegistry that collects,
validates, and queries those declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


VALID_LAYERS = frozenset({"hub", "engine", "dashboard", "cross-cutting"})
VALID_STATUSES = frozenset({"stable", "experimental", "planned"})
VALID_PIPELINE_STAGES = frozenset({"backtest", "shadow", "suggest", "autonomous"})


@dataclass(frozen=True)
class Capability:
    """Declares a single ARIA capability.

    Frozen — instances are immutable after creation. Validation runs in
    __post_init__ to reject invalid layer/status/pipeline_stage values.
    """

    id: str
    name: str
    description: str
    module: str
    layer: str
    config_keys: List[str] = field(default_factory=list)
    test_paths: List[str] = field(default_factory=list)
    test_markers: List[str] = field(default_factory=list)
    runtime_deps: List[str] = field(default_factory=list)
    optional_deps: List[str] = field(default_factory=list)
    data_paths: List[str] = field(default_factory=list)
    systemd_units: List[str] = field(default_factory=list)
    pipeline_stage: Optional[str] = None
    status: str = "stable"
    added_version: str = "1.0.0"
    depends_on: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if self.layer not in VALID_LAYERS:
            raise ValueError(
                f"layer must be one of {sorted(VALID_LAYERS)}, got {self.layer!r}"
            )
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}"
            )
        if self.pipeline_stage is not None and self.pipeline_stage not in VALID_PIPELINE_STAGES:
            raise ValueError(
                f"pipeline_stage must be one of {sorted(VALID_PIPELINE_STAGES)} or None, "
                f"got {self.pipeline_stage!r}"
            )


class CapabilityRegistry:
    """Collects Capability declarations and provides query/validation methods."""

    def __init__(self) -> None:
        self._caps: Dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        """Register a capability. Raises ValueError on duplicate id."""
        if cap.id in self._caps:
            raise ValueError(f"Capability {cap.id!r} already registered")
        self._caps[cap.id] = cap

    def get(self, cap_id: str) -> Optional[Capability]:
        """Return a capability by id, or None if not found."""
        return self._caps.get(cap_id)

    def list_ids(self) -> List[str]:
        """Return all registered capability ids."""
        return list(self._caps.keys())

    def list_all(self) -> List[Capability]:
        """Return all registered capabilities."""
        return list(self._caps.values())

    def list_by_layer(self, layer: str) -> List[Capability]:
        """Return capabilities matching the given layer."""
        return [c for c in self._caps.values() if c.layer == layer]

    def list_by_status(self, status: str) -> List[Capability]:
        """Return capabilities matching the given status."""
        return [c for c in self._caps.values() if c.status == status]

    def dependency_graph(self) -> Dict[str, List[str]]:
        """Return {cap_id: [dependency_ids]} for all registered capabilities."""
        return {cap_id: list(cap.depends_on) for cap_id, cap in self._caps.items()}

    def validate_deps(self) -> List[str]:
        """Validate dependency integrity. Returns a list of error strings.

        Checks for:
        1. Missing dependencies (depends on an id not in the registry)
        2. Cycles (circular dependency chains)
        """
        errors: List[str] = []
        graph = self.dependency_graph()

        # Check for missing deps
        for cap_id, deps in graph.items():
            for dep in deps:
                if dep not in self._caps:
                    errors.append(
                        f"Capability {cap_id!r} depends on {dep!r}, "
                        f"which is not registered"
                    )

        # Cycle detection via DFS with coloring
        # WHITE=unvisited, GRAY=in current path, BLACK=finished
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {cid: WHITE for cid in graph}

        def dfs(node: str, path: List[str]) -> None:
            color[node] = GRAY
            for dep in graph.get(node, []):
                if dep not in color:
                    continue  # missing dep, already reported above
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    errors.append(
                        f"Cycle detected: {' -> '.join(cycle)}"
                    )
                elif color[dep] == WHITE:
                    dfs(dep, path + [dep])
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node, [node])

        return errors
