"""Orchestrator Module - Generate and execute automation suggestions from patterns.

Converts detected behavioral patterns into Home Assistant automations,
manages approval flow, and creates virtual sensors for pattern detection events.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from aria.capabilities import Capability
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)


# Safety guardrail domains - require explicit approval
RESTRICTED_DOMAINS = {"lock", "cover", "alarm_control_panel"}


class OrchestratorModule(Module):
    """Generates automation suggestions and executes approved automations."""

    CAPABILITIES = [
        Capability(
            id="orchestrator",
            name="Automation Orchestrator",
            description="Generates HA automation suggestions from detected behavioral patterns.",
            module="orchestrator",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_orchestrator.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["trajectory_classifier"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub, ha_url: str, ha_token: str, min_confidence: float = 0.7):
        """Initialize orchestrator module.

        Args:
            hub: IntelligenceHub instance
            ha_url: Home Assistant URL (e.g., http://<ha-host>:8123)
            ha_token: Long-lived access token
            min_confidence: Minimum pattern confidence for suggestions (0-1)
        """
        super().__init__("orchestrator", hub)
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self.min_confidence = min_confidence
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self):
        """Initialize HTTP session and generate initial suggestions."""
        self.logger.info("Orchestrator module initializing...")

        # Create HTTP session
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.ha_token}", "Content-Type": "application/json"}
        )

        # Generate initial automation suggestions
        try:
            suggestions = await self.generate_suggestions()
            self.logger.info(f"Initial suggestion generation: {len(suggestions)} suggestions")
        except Exception as e:
            self.logger.error(f"Initial suggestion generation failed: {e}")

        # Schedule periodic suggestion generation (every 6 hours)
        await self.hub.schedule_task(
            task_id="orchestrator_suggestions",
            coro=self.generate_suggestions,
            interval=timedelta(hours=6),
            run_immediately=False,  # Already ran above
        )

    async def shutdown(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events - generate suggestions when patterns updated.

        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type == "cache_updated" and data.get("category") == "patterns":
            self.logger.info("Patterns updated, regenerating automation suggestions")
            try:
                await self.generate_suggestions()
            except Exception as e:
                self.logger.error(f"Failed to regenerate suggestions: {e}")

    async def generate_suggestions(self) -> list[dict[str, Any]]:
        """Generate automation suggestions by delegating to AutomationGeneratorModule.

        If the automation_generator module is registered, delegates to it.
        Otherwise falls back to reading the automation_suggestions cache directly.

        Returns:
            List of generated suggestions
        """
        self.logger.info("Generating automation suggestions...")

        # Delegate to AutomationGeneratorModule if available
        generator = self.hub.modules.get("automation_generator")
        if generator is not None:
            self.logger.info("Delegating suggestion generation to AutomationGeneratorModule")
            return await generator.generate_suggestions()

        # Fallback: read existing suggestions from cache
        self.logger.warning("AutomationGeneratorModule not registered, reading cache directly")
        cached = await self.hub.get_cache("automation_suggestions")
        if cached and "data" in cached:
            return cached["data"].get("suggestions", [])
        return []

    def _check_safety_guardrails(self, actions: list[dict[str, Any]]) -> bool:
        """Check if actions contain restricted domains.

        Args:
            actions: List of HA action dictionaries

        Returns:
            True if explicit approval required
        """
        for action in actions:
            service = action.get("service", "")
            domain = service.split(".")[0] if "." in service else ""

            if domain in RESTRICTED_DOMAINS:
                return True

        return False

    async def approve_suggestion(self, suggestion_id: str) -> dict[str, Any]:
        """Approve an automation suggestion and create it in Home Assistant.

        Args:
            suggestion_id: Suggestion identifier

        Returns:
            Result dictionary with success status and details
        """
        self.logger.info(f"Approving suggestion: {suggestion_id}")

        # 1. Load suggestion from cache
        suggestions_cache = await self.hub.get_cache("automation_suggestions")
        if not suggestions_cache or "data" not in suggestions_cache:
            return {"success": False, "error": "No suggestions found in cache"}

        suggestions = suggestions_cache["data"].get("suggestions", [])
        suggestion = None
        for s in suggestions:
            if s["suggestion_id"] == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            return {"success": False, "error": f"Suggestion {suggestion_id} not found"}

        # 2. Check if already approved
        if suggestion["status"] == "approved":
            return {
                "success": False,
                "error": "Suggestion already approved",
                "automation_id": suggestion.get("automation_id"),
            }

        # 3. Create automation in HA
        try:
            automation_id = f"pattern_{suggestion_id}"
            automation_yaml = suggestion["automation_yaml"]

            result = await self._create_automation(automation_id, automation_yaml)

            if not result["success"]:
                return result

            # 4. Update suggestion status
            suggestion["status"] = "approved"
            suggestion["approved_at"] = datetime.now().isoformat()
            suggestion["automation_id"] = automation_id

            # 5. Save updated suggestions
            await self.hub.set_cache("automation_suggestions", {"suggestions": suggestions, "count": len(suggestions)})

            # 6. Track created automation
            await self._track_created_automation(automation_id, suggestion_id)

            # 6b. Immediately add to ha_automations cache (prevents re-suggestion)
            await self._update_ha_automations_cache(automation_id, automation_yaml)

            # 7. Publish approval event
            await self.hub.publish(
                "automation_approved",
                {
                    "suggestion_id": suggestion_id,
                    "automation_id": automation_id,
                    "pattern_id": suggestion.get("metadata", {}).get("pattern_id", suggestion.get("source", "")),
                },
            )

            self.logger.info(f"Suggestion {suggestion_id} approved, automation {automation_id} created")

            return {"success": True, "automation_id": automation_id, "suggestion_id": suggestion_id}

        except Exception as e:
            self.logger.error(f"Failed to approve suggestion {suggestion_id}: {e}")
            return {"success": False, "error": str(e)}

    async def reject_suggestion(self, suggestion_id: str) -> dict[str, Any]:
        """Reject an automation suggestion.

        Args:
            suggestion_id: Suggestion identifier

        Returns:
            Result dictionary with success status
        """
        self.logger.info(f"Rejecting suggestion: {suggestion_id}")

        # 1. Load suggestion from cache
        suggestions_cache = await self.hub.get_cache("automation_suggestions")
        if not suggestions_cache or "data" not in suggestions_cache:
            return {"success": False, "error": "No suggestions found in cache"}

        suggestions = suggestions_cache["data"].get("suggestions", [])
        suggestion = None
        for s in suggestions:
            if s["suggestion_id"] == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            return {"success": False, "error": f"Suggestion {suggestion_id} not found"}

        # 2. Update status
        suggestion["status"] = "rejected"
        suggestion["rejected_at"] = datetime.now().isoformat()

        # 3. Save updated suggestions
        await self.hub.set_cache("automation_suggestions", {"suggestions": suggestions, "count": len(suggestions)})

        # 4. Publish rejection event
        await self.hub.publish(
            "automation_rejected",
            {
                "suggestion_id": suggestion_id,
                "pattern_id": suggestion.get("metadata", {}).get("pattern_id", suggestion.get("source", "")),
            },
        )

        self.logger.info(f"Suggestion {suggestion_id} rejected")

        return {"success": True, "suggestion_id": suggestion_id}

    async def _create_automation(self, automation_id: str, automation_yaml: dict[str, Any]) -> dict[str, Any]:
        """Create automation in Home Assistant via REST API.

        NOTE: The /api/config/automation/config/{id} endpoint requires admin
        privileges and may return 401 Unauthorized with standard long-lived tokens.

        Alternative approaches:
        1. Use Home Assistant REST API service call: homeassistant.reload_config_entry
        2. Write to automations.yaml file directly (requires file access)
        3. Use HA WebSocket API for automation management

        For MVP, we store the automation YAML in cache and log it for manual creation.

        Args:
            automation_id: Unique automation identifier
            automation_yaml: Automation configuration

        Returns:
            Result dictionary with success status
        """
        url = f"{self.ha_url}/api/config/automation/config/{automation_id}"

        try:
            async with self._session.post(url, json=automation_yaml) as response:
                if response.status in (200, 201):
                    self.logger.info(f"Created automation: {automation_id}")
                    return {"success": True, "automation_id": automation_id}
                elif response.status == 401:
                    # Expected for standard tokens - log automation for manual creation
                    self.logger.warning(
                        "Automation creation requires admin token (HTTP 401). "
                        "Automation YAML logged for manual creation."
                    )
                    self.logger.info(
                        f"Manual automation YAML for {automation_id}:\n{json.dumps(automation_yaml, indent=2)}"
                    )

                    # Store automation YAML in cache for dashboard to display
                    await self._store_pending_automation(automation_id, automation_yaml)

                    return {
                        "success": True,  # Success = stored for manual creation
                        "automation_id": automation_id,
                        "manual_creation_required": True,
                        "note": "Automation stored for manual creation (admin token required)",
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(
                        f"Failed to create automation {automation_id}: HTTP {response.status} - {error_text}"
                    )
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP request failed for automation {automation_id}: {e}")
            return {"success": False, "error": f"Network error: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Unexpected error creating automation {automation_id}: {e}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def _store_pending_automation(self, automation_id: str, automation_yaml: dict[str, Any]):
        """Store automation YAML in cache for manual creation.

        Args:
            automation_id: Automation identifier
            automation_yaml: Automation configuration
        """
        pending_cache = await self.hub.get_cache("pending_automations")
        pending_data = {}

        if pending_cache and "data" in pending_cache:
            pending_data = pending_cache["data"]

        if "automations" not in pending_data:
            pending_data["automations"] = {}

        pending_data["automations"][automation_id] = {"yaml": automation_yaml, "created_at": datetime.now().isoformat()}

        await self.hub.set_cache("pending_automations", pending_data)

    async def _track_created_automation(self, automation_id: str, suggestion_id: str):
        """Track created automation in cache.

        Args:
            automation_id: HA automation ID
            suggestion_id: Suggestion that generated this automation
        """
        # Load existing tracking data
        tracking_cache = await self.hub.get_cache("created_automations")
        tracking_data = {}

        if tracking_cache and "data" in tracking_cache:
            tracking_data = tracking_cache["data"]

        # Add new automation
        if "automations" not in tracking_data:
            tracking_data["automations"] = {}

        tracking_data["automations"][automation_id] = {
            "suggestion_id": suggestion_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # Save tracking data
        await self.hub.set_cache("created_automations", tracking_data)

    async def _update_ha_automations_cache(self, automation_id: str, automation_yaml: dict[str, Any]):
        """Immediately add approved automation to ha_automations cache.

        Prevents re-suggestion before the next sync cycle by ensuring
        the shadow comparison engine sees the new automation.

        Args:
            automation_id: Unique automation identifier
            automation_yaml: Automation configuration
        """
        # Load existing ha_automations cache
        ha_cache = await self.hub.get_cache("ha_automations")
        automations = list(ha_cache["data"].get("automations", [])) if ha_cache and "data" in ha_cache else []

        # Build the automation entry
        automation_entry = {
            "id": automation_id,
            **automation_yaml,
        }

        # Replace existing or append
        replaced = False
        for i, existing in enumerate(automations):
            if existing.get("id") == automation_id:
                automations[i] = automation_entry
                replaced = True
                break

        if not replaced:
            automations.append(automation_entry)

        # Update cache
        await self.hub.set_cache(
            "ha_automations",
            {
                "automations": automations,
                "count": len(automations),
                "last_sync": ha_cache["data"].get("last_sync", "") if ha_cache and "data" in ha_cache else "",
                "changes_since_last": 1,
            },
            {"source": "orchestrator_approval"},
        )

        self.logger.info(f"Added automation {automation_id} to ha_automations cache")

    async def update_pattern_detection_sensor(self, pattern_name: str, pattern_id: str, confidence: float):
        """Update HA virtual sensor for pattern detection events.

        Args:
            pattern_name: Human-readable pattern name
            pattern_id: Pattern identifier
            confidence: Pattern confidence (0-1)
        """
        sensor_entity_id = "sensor.ha_hub_pattern_detected"
        url = f"{self.ha_url}/api/states/{sensor_entity_id}"

        sensor_state = {
            "state": pattern_name,
            "attributes": {
                "pattern_id": pattern_id,
                "confidence": confidence,
                "last_triggered": datetime.now().isoformat(),
                "friendly_name": "HA Hub Pattern Detected",
                "icon": "mdi:brain",
            },
        }

        try:
            async with self._session.post(url, json=sensor_state) as response:
                if response.status == 200:
                    self.logger.debug(f"Updated pattern sensor: {pattern_name}")
                else:
                    error_text = await response.text()
                    self.logger.warning(f"Failed to update pattern sensor: HTTP {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to update pattern sensor: {e}")

    async def get_suggestions(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """Get automation suggestions from cache.

        Args:
            status_filter: Optional status filter (pending|approved|rejected)

        Returns:
            List of suggestions
        """
        suggestions_cache = await self.hub.get_cache("automation_suggestions")
        if not suggestions_cache or "data" not in suggestions_cache:
            return []

        suggestions = suggestions_cache["data"].get("suggestions", [])

        if status_filter:
            suggestions = [s for s in suggestions if s.get("status") == status_filter]

        return suggestions

    async def get_created_automations(self) -> dict[str, Any]:
        """Get tracking data for created automations.

        Returns:
            Dictionary mapping automation ID to metadata
        """
        tracking_cache = await self.hub.get_cache("created_automations")
        if not tracking_cache or "data" not in tracking_cache:
            return {}

        return tracking_cache["data"].get("automations", {})
