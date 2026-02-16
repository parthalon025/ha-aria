"""Capability dataclass and registry for ARIA.

Defines the Capability frozen dataclass that each hub module and engine command
uses to declare what it provides, and a CapabilityRegistry that collects,
validates, and queries those declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VALID_LAYERS = frozenset({"hub", "engine", "dashboard", "cross-cutting"})
VALID_STATUSES = frozenset({"stable", "experimental", "planned"})
VALID_PIPELINE_STAGES = frozenset({"backtest", "shadow", "suggest", "autonomous"})


@dataclass(frozen=True)
class DemandSignal:
    """Declares what entity groupings a module needs from discovery."""

    entity_domains: list[str] = field(default_factory=list)
    device_classes: list[str] = field(default_factory=list)
    min_entities: int = 5
    description: str = ""


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
    config_keys: list[str] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)
    test_markers: list[str] = field(default_factory=list)
    runtime_deps: list[str] = field(default_factory=list)
    optional_deps: list[str] = field(default_factory=list)
    data_paths: list[str] = field(default_factory=list)
    demand_signals: list[DemandSignal] = field(default_factory=list)
    systemd_units: list[str] = field(default_factory=list)
    pipeline_stage: str | None = None
    status: str = "stable"
    added_version: str = "1.0.0"
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must not be empty")
        if not self.name:
            raise ValueError("name must not be empty")
        if self.layer not in VALID_LAYERS:
            raise ValueError(f"layer must be one of {sorted(VALID_LAYERS)}, got {self.layer!r}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {self.status!r}")
        if self.pipeline_stage is not None and self.pipeline_stage not in VALID_PIPELINE_STAGES:
            raise ValueError(
                f"pipeline_stage must be one of {sorted(VALID_PIPELINE_STAGES)} or None, got {self.pipeline_stage!r}"
            )


class CapabilityRegistry:
    """Collects Capability declarations and provides query/validation methods."""

    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        """Register a capability. Raises ValueError on duplicate id."""
        if cap.id in self._caps:
            raise ValueError(f"Capability {cap.id!r} already registered")
        self._caps[cap.id] = cap

    def get(self, cap_id: str) -> Capability | None:
        """Return a capability by id, or None if not found."""
        return self._caps.get(cap_id)

    def list_ids(self) -> list[str]:
        """Return all registered capability ids."""
        return list(self._caps.keys())

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._caps.values())

    def list_by_layer(self, layer: str) -> list[Capability]:
        """Return capabilities matching the given layer."""
        return [c for c in self._caps.values() if c.layer == layer]

    def list_by_status(self, status: str) -> list[Capability]:
        """Return capabilities matching the given status."""
        return [c for c in self._caps.values() if c.status == status]

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return {cap_id: [dependency_ids]} for all registered capabilities."""
        return {cap_id: list(cap.depends_on) for cap_id, cap in self._caps.items()}

    def validate_deps(self) -> list[str]:
        """Validate dependency integrity. Returns a list of error strings.

        Checks for:
        1. Missing dependencies (depends on an id not in the registry)
        2. Cycles (circular dependency chains)
        """
        errors: list[str] = []
        graph = self.dependency_graph()

        # Check for missing deps
        for cap_id, deps in graph.items():
            for dep in deps:
                if dep not in self._caps:
                    errors.append(f"Capability {cap_id!r} depends on {dep!r}, which is not registered")

        # Cycle detection via DFS with coloring
        errors.extend(self._detect_cycles(graph))

        return errors

    @staticmethod
    def _detect_cycles(graph: dict[str, list[str]]) -> list[str]:
        """Detect cycles in a dependency graph using DFS coloring.

        Args:
            graph: {cap_id: [dependency_ids]} mapping.

        Returns:
            List of cycle error strings.
        """
        errors: list[str] = []
        # WHITE=unvisited, GRAY=in current path, BLACK=finished
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {cid: WHITE for cid in graph}

        def dfs(node: str, path: list[str]) -> None:
            color[node] = GRAY
            for dep in graph.get(node, []):
                if dep not in color:
                    continue  # missing dep, already reported above
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    errors.append(f"Cycle detected: {' -> '.join(cycle)}")
                elif color[dep] == WHITE:
                    dfs(dep, path + [dep])
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node, [node])

        return errors

    def validate_config_keys(self) -> list[str]:
        """Validate that all declared config_keys exist in CONFIG_DEFAULTS.

        Returns a list of issue strings for any config key not found.
        """
        from aria.hub.config_defaults import CONFIG_DEFAULTS

        valid_keys = {entry["key"] for entry in CONFIG_DEFAULTS}
        issues: list[str] = []
        for cap in self._caps.values():
            for key in cap.config_keys:
                if key not in valid_keys:
                    issues.append(f"Capability {cap.id!r} declares config_key {key!r}, which is not in CONFIG_DEFAULTS")
        return issues

    def validate_test_paths(self) -> list[str]:
        """Validate that all declared test_paths exist on disk.

        Paths are resolved relative to the project root (parent of the aria/
        package directory).

        Returns a list of issue strings for any path not found.
        """
        project_root = Path(__file__).resolve().parent.parent
        issues: list[str] = []
        for cap in self._caps.values():
            for test_path in cap.test_paths:
                full_path = project_root / test_path
                if not full_path.exists():
                    issues.append(f"Capability {cap.id!r} declares test_path {test_path!r}, which does not exist")
        return issues

    def validate_all(self) -> list[str]:
        """Run all validation checks and return combined issues.

        Combines: validate_deps, validate_config_keys, validate_test_paths.
        """
        issues: list[str] = []
        issues.extend(self.validate_deps())
        issues.extend(self.validate_config_keys())
        issues.extend(self.validate_test_paths())
        return issues

    def health(self, module_status: dict[str, str]) -> dict[str, dict]:
        """Check runtime health of each capability against hub module status.

        Args:
            module_status: {module_id: "running"|"failed"|"registered"} from hub

        Returns:
            {capability_id: {"module_loaded": bool|None, "module_status": str}}
        """
        result = {}
        for cap in self._caps.values():
            if cap.layer == "engine":
                result[cap.id] = {"module_loaded": None, "module_status": "batch"}
            elif cap.module in module_status:
                running = module_status[cap.module] == "running"
                result[cap.id] = {
                    "module_loaded": running,
                    "module_status": module_status[cap.module],
                }
            else:
                result[cap.id] = {"module_loaded": None, "module_status": "unknown"}
        return result

    def collect_from_modules(self) -> None:
        """Discover and register all capabilities from hub modules and engine."""
        # Hub modules
        from aria.modules.activity_monitor import ActivityMonitor
        from aria.modules.data_quality import DataQualityModule
        from aria.modules.discovery import DiscoveryModule
        from aria.modules.intelligence import IntelligenceModule
        from aria.modules.ml_engine import MLEngine
        from aria.modules.orchestrator import OrchestratorModule
        from aria.modules.organic_discovery.module import OrganicDiscoveryModule
        from aria.modules.patterns import PatternRecognition
        from aria.modules.presence import PresenceModule
        from aria.modules.shadow_engine import ShadowEngine

        hub_modules = [
            DiscoveryModule,
            MLEngine,
            PatternRecognition,
            OrchestratorModule,
            ShadowEngine,
            DataQualityModule,
            OrganicDiscoveryModule,
            IntelligenceModule,
            ActivityMonitor,
            PresenceModule,
        ]
        for module_cls in hub_modules:
            for cap in getattr(module_cls, "CAPABILITIES", []):
                self.register(cap)

        # Engine capabilities
        from aria.engine.capabilities import ENGINE_CAPABILITIES

        for cap in ENGINE_CAPABILITIES:
            self.register(cap)
