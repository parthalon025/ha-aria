"""AutomationGenerator Hub Module — coordinator for the suggestion pipeline.

Reads pattern and gap detection caches, applies combined scoring,
selects top-N candidates, runs them through the template engine →
LLM refiner → validator → shadow comparison pipeline, and stores
the surviving suggestions in the automation_suggestions cache.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

from aria.automation.llm_refiner import refine_automation
from aria.automation.models import ChainLink, DetectionResult
from aria.automation.template_engine import AutomationTemplate
from aria.automation.validator import validate_automation
from aria.capabilities import Capability
from aria.hub.core import IntelligenceHub, Module
from aria.shared.shadow_comparison import compare_candidate

logger = logging.getLogger(__name__)

# Scoring weights: pattern × 0.5 + gap × 0.3 + recency × 0.2
PATTERN_WEIGHT = 0.5
GAP_WEIGHT = 0.3
RECENCY_WEIGHT = 0.2

# Shadow statuses that pass through to suggestions
ALLOWED_SHADOW_STATUSES = frozenset({"new", "gap_fill", "conflict"})


def compute_combined_score(detection: DetectionResult) -> float:
    """Score a detection: pattern × 0.5 + gap × 0.3 + recency × 0.2.

    For pattern-source detections, confidence feeds the pattern weight.
    For gap-source detections, confidence feeds the gap weight.
    Recency weight always applies.

    Args:
        detection: The detection result to score.

    Returns:
        Combined score (0.0–1.0), also stored on detection.combined_score.
    """
    if detection.source == "pattern":
        score = detection.confidence * PATTERN_WEIGHT + 0.0 * GAP_WEIGHT + detection.recency_weight * RECENCY_WEIGHT
    else:
        score = 0.0 * PATTERN_WEIGHT + detection.confidence * GAP_WEIGHT + detection.recency_weight * RECENCY_WEIGHT

    detection.combined_score = score
    return score


class AutomationGeneratorModule(Module):
    """Coordinates the full automation suggestion pipeline."""

    CAPABILITIES = [
        Capability(
            id="automation_generator",
            name="Automation Generator",
            description="Generates HA automation suggestions from pattern and gap detections.",
            module="automation_generator",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_automation_generator.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["orchestrator"],
        ),
    ]

    def __init__(
        self,
        hub: IntelligenceHub,
        top_n: int = 10,
        min_confidence: float = 0.7,
    ):
        """Initialize automation generator.

        Args:
            hub: IntelligenceHub instance.
            top_n: Maximum number of suggestions to produce per cycle.
            min_confidence: Minimum detection confidence to consider.
        """
        super().__init__("automation_generator", hub)
        self.top_n = top_n
        self.min_confidence = min_confidence

    async def initialize(self):
        """Schedule periodic suggestion generation."""
        self.logger.info("AutomationGenerator initializing...")

        # Initial generation
        try:
            suggestions = await self.generate_suggestions()
            self.logger.info("Initial generation: %d suggestions", len(suggestions))
        except Exception as e:
            self.logger.error("Initial generation failed: %s", e)

        # Schedule periodic generation (every 6 hours)
        await self.hub.schedule_task(
            task_id="automation_generator_cycle",
            coro=self.generate_suggestions,
            interval=timedelta(hours=6),
            run_immediately=False,
        )

    async def shutdown(self):
        """No resources to clean up."""
        pass

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Regenerate suggestions when patterns or gaps are updated."""
        if event_type == "cache_updated" and data.get("category") in ("patterns", "gaps"):
            self.logger.info("Cache '%s' updated, regenerating suggestions", data["category"])
            try:
                await self.generate_suggestions()
            except Exception as e:
                self.logger.error("Failed to regenerate suggestions: %s", e)

    async def generate_suggestions(self) -> list[dict[str, Any]]:
        """Run the full pipeline: score → top-N → template → LLM → validate → shadow → store.

        Returns:
            List of suggestion dicts that passed all gates.
        """
        # 1. Load detections from both caches
        detections = await self._load_detections()
        if not detections:
            self.logger.info("No detections found in pattern/gap caches")
            await self._update_health_cache([])
            return []

        # 2. Score and sort
        for det in detections:
            compute_combined_score(det)

        detections.sort(key=lambda d: d.combined_score, reverse=True)

        # 3. Filter by min_confidence and take top-N
        eligible = [d for d in detections if d.confidence >= self.min_confidence]
        candidates = eligible[: self.top_n]

        self.logger.info(
            "Pipeline: %d detections, %d eligible, %d candidates",
            len(detections),
            len(eligible),
            len(candidates),
        )

        # 4. Load HA automations for shadow comparison
        ha_automations = await self._load_ha_automations()
        existing_ids = {a.get("id", "") for a in ha_automations}

        # 5. Template → LLM → Validate → Shadow
        template_engine = AutomationTemplate(self.hub.entity_graph)
        suggestions: list[dict[str, Any]] = []

        for det in candidates:
            suggestion = await self._process_candidate(
                det,
                template_engine,
                ha_automations,
                existing_ids,
            )
            if suggestion is not None:
                suggestions.append(suggestion)

        # 6. Load existing suggestions to preserve status
        existing_suggestions = await self._load_existing_suggestions()

        # 7. Merge with existing (preserve approval status)
        for s in suggestions:
            sid = s["suggestion_id"]
            if sid in existing_suggestions:
                existing = existing_suggestions[sid]
                s["status"] = existing.get("status", "pending")
                s["created_at"] = existing.get("created_at", s.get("created_at"))

        # 8. Store in cache
        await self.hub.set_cache(
            "automation_suggestions",
            {
                "suggestions": suggestions,
                "count": len(suggestions),
            },
            {"source": "automation_generator"},
        )

        self.logger.info("Generated %d automation suggestions", len(suggestions))

        # 9. Update system health cache
        await self._update_health_cache(suggestions)

        return suggestions

    # ------------------------------------------------------------------
    # Health reporting
    # ------------------------------------------------------------------

    async def _update_health_cache(self, suggestions: list[dict[str, Any]]) -> None:
        """Update the automation_system_health cache category.

        Provides a single-read health snapshot for the /api/automations/health
        endpoint and observability dashboards.
        """
        try:
            ha_cache = await self.hub.get_cache("ha_automations")
            pipeline = await self.hub.get_cache("pipeline_state")
            feedback_cache = await self.hub.get_cache("automation_feedback")

            ha_data = ha_cache.get("data", {}) if ha_cache else {}
            fb_data = feedback_cache.get("data", {}) if feedback_cache else {}

            pending = sum(1 for s in suggestions if s.get("status") == "pending")
            approved = sum(1 for s in suggestions if s.get("status") == "approved")
            rejected = sum(1 for s in suggestions if s.get("status") == "rejected")

            health = {
                "suggestions_total": len(suggestions),
                "suggestions_pending": pending,
                "suggestions_approved": approved,
                "suggestions_rejected": rejected,
                "ha_automations_count": len(ha_data.get("automations", {})),
                "ha_automations_last_synced": ha_cache.get("last_updated") if ha_cache else None,
                "pipeline_stage": pipeline.get("current_stage", "shadow") if pipeline else "shadow",
                "feedback_count": len(fb_data.get("suggestions", {})),
                "generator_loaded": True,
                "orchestrator_loaded": self.hub.get_module("orchestrator") is not None,
                "last_generation": datetime.now().isoformat(),
            }

            await self.hub.set_cache(
                "automation_system_health",
                health,
                {"source": "automation_generator"},
            )
        except Exception as e:
            self.logger.warning("Failed to update health cache: %s", e)

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    async def _load_detections(self) -> list[DetectionResult]:
        """Load and parse detections from patterns and gaps caches."""
        detections: list[DetectionResult] = []

        patterns_cache = await self.hub.get_cache("patterns")
        if patterns_cache and "data" in patterns_cache:
            raw = patterns_cache["data"].get("patterns") or patterns_cache["data"].get("detections", [])
            detections.extend(self._parse_detections(raw))

        gaps_cache = await self.hub.get_cache("gaps")
        if gaps_cache and "data" in gaps_cache:
            raw = gaps_cache["data"].get("gaps") or gaps_cache["data"].get("detections", [])
            detections.extend(self._parse_detections(raw))

        return detections

    @staticmethod
    def _parse_detections(raw_list: list[dict]) -> list[DetectionResult]:
        """Parse raw dicts into DetectionResult dataclasses."""
        results: list[DetectionResult] = []
        for raw in raw_list:
            try:
                chain = [
                    ChainLink(
                        entity_id=c["entity_id"],
                        state=c["state"],
                        offset_seconds=c["offset_seconds"],
                    )
                    for c in raw.get("entity_chain", [])
                ]
                det = DetectionResult(
                    source=raw["source"],
                    trigger_entity=raw["trigger_entity"],
                    action_entities=raw.get("action_entities", []),
                    entity_chain=chain,
                    area_id=raw.get("area_id"),
                    confidence=raw.get("confidence", 0.0),
                    recency_weight=raw.get("recency_weight", 0.0),
                    observation_count=raw.get("observation_count", 0),
                    first_seen=raw.get("first_seen", ""),
                    last_seen=raw.get("last_seen", ""),
                    day_type=raw.get("day_type", "all"),
                )
                results.append(det)
            except (KeyError, TypeError) as e:
                logger.warning("Skipping malformed detection: %s", e)
        return results

    async def _load_ha_automations(self) -> list[dict[str, Any]]:
        """Load existing HA automations from cache for shadow comparison."""
        cached = await self.hub.get_cache("ha_automations")
        if cached and "data" in cached:
            return cached["data"].get("automations", [])
        return []

    async def _load_existing_suggestions(self) -> dict[str, dict[str, Any]]:
        """Load existing suggestions keyed by suggestion_id."""
        cached = await self.hub.get_cache("automation_suggestions")
        if cached and "data" in cached:
            return {s["suggestion_id"]: s for s in cached["data"].get("suggestions", [])}
        return {}

    async def _process_candidate(
        self,
        detection: DetectionResult,
        template_engine: AutomationTemplate,
        ha_automations: list[dict[str, Any]],
        existing_ids: set[str],
    ) -> dict[str, Any] | None:
        """Process one detection through template → LLM → validate → shadow.

        Returns a suggestion dict or None if the candidate was rejected.
        """
        # Template composition
        try:
            automation = template_engine.build(detection)
        except Exception as e:
            self.logger.error("Template build failed for %s: %s", detection.trigger_entity, e)
            return None

        # LLM refinement (best-effort)
        try:
            automation = await refine_automation(automation)
        except Exception:
            self.logger.debug("LLM refiner failed for %s, using template output", detection.trigger_entity)

        # Validation
        valid, errors = validate_automation(automation, self.hub.entity_graph, existing_ids)
        if not valid:
            self.logger.info(
                "Validation rejected %s: %s",
                detection.trigger_entity,
                "; ".join(errors),
            )
            return None

        # Shadow comparison
        shadow_result = compare_candidate(automation, ha_automations, self.hub.entity_graph)
        if shadow_result.status not in ALLOWED_SHADOW_STATUSES:
            self.logger.info(
                "Shadow rejected %s: %s (%s)",
                detection.trigger_entity,
                shadow_result.status,
                shadow_result.reason,
            )
            return None

        # Track the ID so subsequent candidates don't collide
        auto_id = automation.get("id", "")
        if auto_id:
            existing_ids.add(auto_id)

        # Build suggestion
        suggestion_id = _generate_suggestion_id(detection)
        return {
            "suggestion_id": suggestion_id,
            "automation_yaml": automation,
            "combined_score": detection.combined_score,
            "source": detection.source,
            "shadow_status": shadow_result.status,
            "shadow_reason": shadow_result.reason,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "trigger_entity": detection.trigger_entity,
                "action_entities": detection.action_entities,
                "area_id": detection.area_id,
                "confidence": detection.confidence,
                "recency_weight": detection.recency_weight,
                "observation_count": detection.observation_count,
                "day_type": detection.day_type,
            },
        }


def _generate_suggestion_id(detection: DetectionResult) -> str:
    """Generate a deterministic suggestion ID from detection fields."""
    parts = [
        detection.source,
        detection.trigger_entity,
        "|".join(sorted(detection.action_entities)),
        detection.day_type,
    ]
    raw = "::".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
