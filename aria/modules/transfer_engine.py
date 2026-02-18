"""Transfer engine hub module — cross-domain pattern transfer orchestration.

Generates transfer candidates from organic discovery capabilities,
tests them via shadow engine results, and promotes/rejects after
sufficient evidence. Self-gates on Tier 3+ hardware.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any

from aria.engine.hardware import recommend_tier, scan_hardware
from aria.engine.transfer import TransferCandidate, TransferType
from aria.engine.transfer_generator import generate_transfer_candidates
from aria.hub.core import Module

logger = logging.getLogger(__name__)

MIN_TIER = 3


class TransferEngineModule(Module):
    """Hub module for cross-domain pattern transfer."""

    def __init__(self, hub):
        super().__init__("transfer_engine", hub)
        self.active = False
        self.candidates: list[TransferCandidate] = []
        self._generation_count = 0

    async def initialize(self):
        """Check hardware tier and activate if sufficient."""
        profile = scan_hardware()
        tier = recommend_tier(profile)

        if tier < MIN_TIER:
            logger.info(
                f"Transfer engine disabled: tier {tier} < {MIN_TIER} "
                f"({profile.ram_gb:.1f}GB RAM, {profile.cpu_cores} cores)"
            )
            self.active = False
            return

        self.active = True
        self.hub.subscribe("organic_discovery_complete", self._on_discovery_complete)
        self.hub.subscribe("shadow_resolved", self._on_shadow_resolved)

        # Load persisted candidates from cache
        cached = await self.hub.get_cache("transfer_candidates")
        if cached and cached.get("data"):
            self._load_candidates(cached["data"])

        logger.info(f"Transfer engine active at tier {tier}, {len(self.candidates)} cached candidates loaded")

    async def shutdown(self):
        """Unsubscribe and persist state."""
        if self.active:
            self.hub.unsubscribe("organic_discovery_complete", self._on_discovery_complete)
            self.hub.unsubscribe("shadow_resolved", self._on_shadow_resolved)
            await self._persist_candidates()

    async def _on_discovery_complete(self, event: dict[str, Any]):
        """Regenerate transfer candidates when organic discovery runs."""
        if not self.active:
            return

        caps_entry = await self.hub.get_cache("capabilities")
        entities_entry = await self.hub.get_cache("entities")

        if not caps_entry or not caps_entry.get("data"):
            return
        if not entities_entry or not entities_entry.get("data"):
            return

        capabilities = caps_entry["data"]
        entities_cache = entities_entry["data"]

        # Generate new candidates
        new_candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.6)

        # Merge with existing — preserve active/testing candidates
        existing_keys = {
            (c.source_capability, c.target_context) for c in self.candidates if c.state in ("testing", "promoted")
        }
        for nc in new_candidates:
            key = (nc.source_capability, nc.target_context)
            if key not in existing_keys:
                self.candidates.append(nc)

        # Prune rejected candidates older than 30 days
        self.candidates = [
            c for c in self.candidates if c.state != "rejected" or (datetime.now() - c.created_at).days < 30
        ]

        self._generation_count += 1
        await self._persist_candidates()

        logger.info(
            f"Transfer generation #{self._generation_count}: {len(new_candidates)} new, {len(self.candidates)} total"
        )

    async def _on_shadow_resolved(self, event: dict[str, Any]):
        """Test active transfer candidates against shadow results."""
        if not self.active:
            return

        outcome = event.get("outcome", "")
        actual_data = event.get("actual_data", {})
        if not outcome or not actual_data:
            return

        # Check if any candidate's target entities are involved
        entity_id = actual_data.get("entity_id", "")
        if not entity_id:
            return

        for candidate in self.candidates:
            if candidate.state not in ("hypothesis", "testing"):
                continue

            if entity_id in candidate.target_entities:
                hit = outcome == "correct"
                candidate.record_shadow_result(hit=hit)

        # Periodic promotion check (every shadow event)
        self._check_promotions()

    def _check_promotions(self):
        """Check all testing candidates for promotion/rejection."""
        for candidate in self.candidates:
            if candidate.state == "testing":
                candidate.check_promotion(min_days=7, min_hit_rate=0.6, reject_below=0.3)

    def _load_candidates(self, data: list[dict]) -> None:
        """Load candidates from cached dicts (best-effort)."""
        for d in data:
            try:
                tc = TransferCandidate(
                    source_capability=d["source_capability"],
                    target_context=d["target_context"],
                    transfer_type=TransferType(d["transfer_type"]),
                    similarity_score=d["similarity_score"],
                    source_entities=d.get("source_entities", []),
                    target_entities=d.get("target_entities", []),
                    timing_offset_minutes=d.get("timing_offset_minutes", 0),
                )
                tc.state = d.get("state", "hypothesis")
                tc.shadow_tests = d.get("shadow_tests", 0)
                tc.shadow_hits = d.get("shadow_hits", 0)
                if d.get("testing_since"):
                    tc.testing_since = datetime.fromisoformat(d["testing_since"])
                self.candidates.append(tc)
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping invalid cached candidate: {e}")

    async def _persist_candidates(self):
        """Save candidates to cache."""
        data = [c.to_dict() for c in self.candidates]
        await self.hub.set_cache(
            "transfer_candidates",
            data,
            {"count": len(data), "source": "transfer_engine"},
        )

    def get_current_state(self) -> dict[str, Any]:
        """Return current transfer engine state."""
        state_counts = Counter(c.state for c in self.candidates)
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "summary": {
                "total": len(self.candidates),
                "by_state": dict(state_counts),
                "generation_runs": self._generation_count,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Return module statistics."""
        state_counts = Counter(c.state for c in self.candidates)
        return {
            "active": self.active,
            "candidates_total": len(self.candidates),
            "candidates_by_state": dict(state_counts),
            "generation_runs": self._generation_count,
        }
