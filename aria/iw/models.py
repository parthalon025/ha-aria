"""Data models for the I&W (Indicators & Warnings) behavioral state framework.

Defines frozen and mutable dataclasses for representing behavioral state
definitions, trackers, active states, and their constituent indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_ROLES = {"trigger", "confirming", "deviation"}
_VALID_MODES = {"state_change", "quiet_period", "threshold"}
_VALID_LIFECYCLES = {"seed", "emerging", "confirmed", "mature", "dormant", "retired"}
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "seed": {"emerging", "retired"},
    "emerging": {"confirmed", "dormant", "seed", "retired"},
    "confirmed": {"mature", "dormant", "seed", "retired"},
    "mature": {"dormant", "retired"},
    "dormant": {"emerging", "retired"},
    "retired": set(),
}


@dataclass(frozen=True)
class Indicator:
    """A single observable signal used in a behavioral state definition.

    Frozen — immutable after creation. Validation runs in __post_init__.
    """

    entity_id: str
    role: str  # trigger | confirming | deviation
    mode: str  # state_change | quiet_period | threshold
    expected_state: str | None = None
    quiet_seconds: int | None = None
    threshold_value: float | None = None
    threshold_direction: str | None = None  # above | below
    max_delay_seconds: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {self.role!r}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}")
        # Cross-validate mode-specific required fields (#146)
        if self.mode == "threshold":
            if self.threshold_value is None:
                raise ValueError("threshold_value required when mode='threshold'")
            if self.threshold_direction not in ("above", "below"):
                raise ValueError("threshold_direction must be 'above' or 'below' when mode='threshold'")
        if self.mode == "quiet_period" and not self.quiet_seconds:
            raise ValueError("quiet_seconds required when mode='quiet_period'")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "mode": self.mode,
            "expected_state": self.expected_state,
            "quiet_seconds": self.quiet_seconds,
            "threshold_value": self.threshold_value,
            "threshold_direction": self.threshold_direction,
            "max_delay_seconds": self.max_delay_seconds,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Indicator:
        """Deserialize from a dict produced by to_dict()."""
        return cls(
            entity_id=data["entity_id"],
            role=data["role"],
            mode=data["mode"],
            expected_state=data.get("expected_state"),
            quiet_seconds=data.get("quiet_seconds"),
            threshold_value=data.get("threshold_value"),
            threshold_direction=data.get("threshold_direction"),
            max_delay_seconds=data.get("max_delay_seconds", 0),
            confidence=data.get("confidence", 0.0),
        )


@dataclass(frozen=True)
class BehavioralStateDefinition:
    """Describes a recurring behavioral pattern observed in the home.

    Frozen — immutable after creation. Contains a trigger indicator plus
    optional preconditions, confirming signals, and deviation signals.
    """

    id: str
    name: str
    trigger: Indicator
    trigger_preconditions: list[Indicator]
    confirming: list[Indicator]
    deviations: list[Indicator]
    areas: frozenset[str]
    day_types: frozenset[str]
    person_attribution: str | None
    typical_duration_minutes: float
    expected_outcomes: tuple[dict, ...]
    composite_of: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must be non-empty")
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.trigger.role != "trigger":
            raise ValueError(f"trigger must have role='trigger', got {self.trigger.role!r}")
        for ind in self.confirming:
            if ind.role != "confirming":
                raise ValueError(f"confirming indicator must have role='confirming', got {ind.role!r}")
        for ind in self.deviations:
            if ind.role != "deviation":
                raise ValueError(f"deviation indicator must have role='deviation', got {ind.role!r}")

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "trigger": self.trigger.to_dict(),
            "trigger_preconditions": [i.to_dict() for i in self.trigger_preconditions],
            "confirming": [i.to_dict() for i in self.confirming],
            "deviations": [i.to_dict() for i in self.deviations],
            "areas": sorted(self.areas),
            "day_types": sorted(self.day_types),
            "person_attribution": self.person_attribution,
            "typical_duration_minutes": self.typical_duration_minutes,
            "expected_outcomes": list(self.expected_outcomes),
            "composite_of": list(self.composite_of),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BehavioralStateDefinition:
        """Deserialize from a dict produced by to_dict()."""
        return cls(
            id=data["id"],
            name=data["name"],
            trigger=Indicator.from_dict(data["trigger"]),
            trigger_preconditions=[Indicator.from_dict(i) for i in data.get("trigger_preconditions", [])],
            confirming=[Indicator.from_dict(i) for i in data.get("confirming", [])],
            deviations=[Indicator.from_dict(i) for i in data.get("deviations", [])],
            areas=frozenset(data.get("areas", [])),
            day_types=frozenset(data.get("day_types", [])),
            person_attribution=data.get("person_attribution"),
            typical_duration_minutes=data.get("typical_duration_minutes", 0.0),
            expected_outcomes=tuple(data.get("expected_outcomes", [])),
            composite_of=tuple(data.get("composite_of", [])),
        )


@dataclass
class BehavioralStateTracker:
    """Tracks lifecycle and observation history for a behavioral state definition.

    Mutable — fields are updated as new observations are recorded.
    """

    definition_id: str
    lifecycle: str = "seed"
    observation_count: int = 0
    consistency: float = 0.0
    first_seen: str = ""
    last_seen: str = ""
    lifecycle_history: list[dict] = field(default_factory=list)
    backtest_result: dict | None = None
    user_feedback: str | None = None
    automation_suggestion_id: str | None = None
    automation_status: str | None = None

    def __post_init__(self) -> None:
        if self.lifecycle not in _VALID_LIFECYCLES:
            raise ValueError(f"lifecycle must be one of {sorted(_VALID_LIFECYCLES)}, got {self.lifecycle!r}")

    def transition_lifecycle(self, new_lifecycle: str, timestamp: str) -> None:
        """Transition to a new lifecycle stage with state machine validation.

        Raises ValueError for invalid transitions (e.g. mature -> seed).
        Records the transition in lifecycle_history.
        """
        if new_lifecycle not in _VALID_LIFECYCLES:
            raise ValueError(f"lifecycle must be one of {sorted(_VALID_LIFECYCLES)}, got {new_lifecycle!r}")
        valid = _VALID_TRANSITIONS.get(self.lifecycle, set())
        if new_lifecycle not in valid:
            raise ValueError(
                f"Invalid lifecycle transition: {self.lifecycle!r} -> {new_lifecycle!r}. "
                f"Valid transitions from {self.lifecycle!r}: {sorted(valid)}"
            )
        old = self.lifecycle
        self.lifecycle = new_lifecycle
        self.lifecycle_history.append(
            {
                "from": old,
                "to": new_lifecycle,
                "timestamp": timestamp,
            }
        )

    def record_observation(self, timestamp: str, match_ratio: float) -> None:
        """Record a new observation, updating count, timestamps, and consistency.

        Consistency is a running average of match_ratio values over all observations.
        """
        self.observation_count += 1
        self.last_seen = timestamp
        if not self.first_seen:
            self.first_seen = timestamp
        # Running average: new_avg = old_avg + (new_val - old_avg) / n
        n = self.observation_count
        self.consistency = self.consistency + (match_ratio - self.consistency) / n

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "definition_id": self.definition_id,
            "lifecycle": self.lifecycle,
            "observation_count": self.observation_count,
            "consistency": self.consistency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "lifecycle_history": list(self.lifecycle_history),
            "backtest_result": self.backtest_result,
            "user_feedback": self.user_feedback,
            "automation_suggestion_id": self.automation_suggestion_id,
            "automation_status": self.automation_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BehavioralStateTracker:
        """Deserialize from a dict produced by to_dict()."""
        tracker = cls(
            definition_id=data["definition_id"],
            lifecycle=data.get("lifecycle", "seed"),
            observation_count=data.get("observation_count", 0),
            consistency=data.get("consistency", 0.0),
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            lifecycle_history=list(data.get("lifecycle_history", [])),
            backtest_result=data.get("backtest_result"),
            user_feedback=data.get("user_feedback"),
            automation_suggestion_id=data.get("automation_suggestion_id"),
            automation_status=data.get("automation_status"),
        )
        return tracker


@dataclass
class ActiveState:
    """Represents a behavioral state that has been triggered and is being evaluated.

    Mutable — matched/pending confirming lists are updated as signals arrive.
    """

    definition_id: str
    trigger_time: str
    matched_confirming: list[str]
    pending_confirming: list[str]
    window_expires: str

    def __post_init__(self) -> None:
        overlap = set(self.matched_confirming) & set(self.pending_confirming)
        if overlap:
            raise ValueError(f"matched_confirming and pending_confirming must be disjoint, overlap: {overlap}")

    def confirm_indicator(self, entity_id: str) -> None:
        """Move an indicator from pending to matched. Raises ValueError if not pending."""
        if entity_id not in self.pending_confirming:
            raise ValueError(f"{entity_id!r} is not in pending_confirming")
        self.pending_confirming.remove(entity_id)
        self.matched_confirming.append(entity_id)

    @property
    def match_ratio(self) -> float:
        """Fraction of confirming indicators that have matched.

        Returns 0.0 if both matched and pending lists are empty.
        """
        total = len(self.matched_confirming) + len(self.pending_confirming)
        if total == 0:
            return 0.0
        return len(self.matched_confirming) / total
