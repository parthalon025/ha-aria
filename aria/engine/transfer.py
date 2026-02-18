"""Cross-domain pattern transfer data model.

TransferCandidates represent hypotheses that a pattern observed in one
context might apply in another. The shadow engine tests these hypotheses
and promotes them after sufficient evidence accumulates.

Tier 3+ only — no heavy dependencies, pure data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class TransferType(StrEnum):
    """Types of cross-domain pattern transfer."""

    ROOM_TO_ROOM = "room_to_room"
    ROUTINE_TO_ROUTINE = "routine_to_routine"


MIN_SIMILARITY = 0.4


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A,B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 for empty sets.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class TransferCandidate:
    """A hypothesis that a pattern in one context applies to another.

    Lifecycle: hypothesis → testing → promoted/rejected

    Args:
        source_capability: Name of the capability being transferred from.
        target_context: Target context identifier (room name, routine label).
        transfer_type: Room-to-room or routine-to-routine.
        similarity_score: Jaccard similarity between source and target (0-1).
        source_entities: Entity IDs in the source context.
        target_entities: Entity IDs in the target context.
        timing_offset_minutes: For routine-to-routine, time shift in minutes.
    """

    source_capability: str
    target_context: str
    transfer_type: TransferType
    similarity_score: float
    source_entities: list[str]
    target_entities: list[str]
    timing_offset_minutes: int = 0

    # Lifecycle state
    state: str = field(default="hypothesis", init=False)
    created_at: datetime = field(default_factory=datetime.now, init=False)
    testing_since: datetime | None = field(default=None, init=False)
    promoted_at: datetime | None = field(default=None, init=False)

    # Shadow test tracking
    shadow_tests: int = field(default=0, init=False)
    shadow_hits: int = field(default=0, init=False)

    def __post_init__(self):
        if self.similarity_score < MIN_SIMILARITY:
            raise ValueError(
                f"similarity score {self.similarity_score:.2f} below minimum "
                f"{MIN_SIMILARITY}. Transfer candidates require structural "
                f"similarity >= {MIN_SIMILARITY}."
            )

    @property
    def hit_rate(self) -> float:
        """Current shadow test hit rate (0-1)."""
        if self.shadow_tests == 0:
            return 0.0
        return self.shadow_hits / self.shadow_tests

    def record_shadow_result(self, hit: bool) -> None:
        """Record a shadow engine test result.

        Transitions from hypothesis to testing on first result.
        """
        if self.state == "hypothesis":
            self.state = "testing"
            self.testing_since = datetime.now()

        self.shadow_tests += 1
        if hit:
            self.shadow_hits += 1

    def check_promotion(
        self,
        min_days: int = 7,
        min_hit_rate: float = 0.6,
        reject_below: float = 0.3,
    ) -> None:
        """Check if this candidate should be promoted or rejected.

        Args:
            min_days: Minimum days of testing before promotion decision.
            min_hit_rate: Minimum hit rate for promotion.
            reject_below: Hit rate below which candidate is rejected.
        """
        if self.state != "testing" or self.testing_since is None:
            return

        days_testing = (datetime.now() - self.testing_since).days
        if days_testing < min_days:
            return

        if self.shadow_tests < 5:
            return  # Not enough data

        if self.hit_rate >= min_hit_rate:
            self.state = "promoted"
            self.promoted_at = datetime.now()
        elif self.hit_rate < reject_below:
            self.state = "rejected"

    def to_dict(self) -> dict:
        """Serialize to dict for cache storage."""
        return {
            "source_capability": self.source_capability,
            "target_context": self.target_context,
            "transfer_type": self.transfer_type.value,
            "similarity_score": self.similarity_score,
            "source_entities": self.source_entities,
            "target_entities": self.target_entities,
            "timing_offset_minutes": self.timing_offset_minutes,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "testing_since": self.testing_since.isoformat() if self.testing_since else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "shadow_tests": self.shadow_tests,
            "shadow_hits": self.shadow_hits,
            "hit_rate": round(self.hit_rate, 4),
        }
