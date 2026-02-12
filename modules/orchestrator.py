"""Orchestrator Module - Generate and execute automation suggestions from patterns.

Converts detected behavioral patterns into Home Assistant automations,
manages approval flow, and creates virtual sensors for pattern detection events.
"""

import json
import logging
import aiohttp
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from hub.core import Module, IntelligenceHub


logger = logging.getLogger(__name__)


# Safety guardrail domains - require explicit approval
RESTRICTED_DOMAINS = {"lock", "cover", "alarm_control_panel"}


class OrchestratorModule(Module):
    """Generates automation suggestions and executes approved automations."""

    def __init__(
        self,
        hub: IntelligenceHub,
        ha_url: str,
        ha_token: str,
        min_confidence: float = 0.7
    ):
        """Initialize orchestrator module.

        Args:
            hub: IntelligenceHub instance
            ha_url: Home Assistant URL (e.g., http://192.168.1.35:8123)
            ha_token: Long-lived access token
            min_confidence: Minimum pattern confidence for suggestions (0-1)
        """
        super().__init__("orchestrator", hub)
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self.min_confidence = min_confidence
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session and generate initial suggestions."""
        self.logger.info("Orchestrator module initializing...")

        # Create HTTP session
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json"
            }
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
            run_immediately=False  # Already ran above
        )

    async def shutdown(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def on_event(self, event_type: str, data: Dict[str, Any]):
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

    async def generate_suggestions(self) -> List[Dict[str, Any]]:
        """Generate automation suggestions from detected patterns.

        Reads patterns from hub cache, filters by confidence threshold,
        generates HA automation YAML structures, and stores suggestions in cache.

        Returns:
            List of generated suggestions
        """
        self.logger.info("Generating automation suggestions from patterns...")

        # 1. Load patterns from cache (warn if older than 24 hours)
        patterns_cache = await self.hub.get_cache_fresh(
            "patterns", timedelta(hours=24), caller="orchestrator"
        )
        if not patterns_cache or "data" not in patterns_cache:
            self.logger.warning("No patterns found in cache")
            return []

        patterns_data = patterns_cache["data"]
        patterns = patterns_data.get("patterns", [])

        if not patterns:
            self.logger.warning("Patterns cache is empty")
            return []

        self.logger.info(f"Found {len(patterns)} patterns in cache")

        # 2. Filter patterns by confidence
        eligible_patterns = [
            p for p in patterns
            if p.get("confidence", 0) >= self.min_confidence
        ]

        self.logger.info(
            f"{len(eligible_patterns)} patterns meet confidence threshold "
            f"(≥{self.min_confidence:.0%})"
        )

        # 3. Generate suggestions
        suggestions = []
        for pattern in eligible_patterns:
            try:
                suggestion = await self._pattern_to_suggestion(pattern)
                suggestions.append(suggestion)
            except Exception as e:
                self.logger.error(
                    f"Failed to generate suggestion for pattern {pattern.get('pattern_id')}: {e}"
                )

        # 4. Load existing suggestions to preserve status
        existing_suggestions_cache = await self.hub.get_cache("automation_suggestions")
        existing_suggestions = {}
        if existing_suggestions_cache and "data" in existing_suggestions_cache:
            for s in existing_suggestions_cache["data"].get("suggestions", []):
                existing_suggestions[s["suggestion_id"]] = s

        # 5. Merge with existing suggestions (preserve approval status)
        final_suggestions = []
        for suggestion in suggestions:
            suggestion_id = suggestion["suggestion_id"]
            if suggestion_id in existing_suggestions:
                # Preserve status from existing suggestion
                existing = existing_suggestions[suggestion_id]
                suggestion["status"] = existing.get("status", "pending")
                suggestion["created_at"] = existing.get("created_at", suggestion["created_at"])
                if existing.get("automation_id"):
                    suggestion["automation_id"] = existing["automation_id"]
            final_suggestions.append(suggestion)

        # 6. Store in cache
        await self.hub.set_cache("automation_suggestions", {
            "suggestions": final_suggestions,
            "count": len(final_suggestions),
            "eligible_patterns": len(eligible_patterns),
            "total_patterns": len(patterns)
        }, {
            "source": "orchestrator",
            "min_confidence": self.min_confidence
        })

        self.logger.info(f"Generated {len(final_suggestions)} automation suggestions")
        return final_suggestions

    async def _pattern_to_suggestion(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a pattern into an automation suggestion.

        Args:
            pattern: Pattern dictionary from pattern recognition module

        Returns:
            Automation suggestion dictionary
        """
        pattern_id = pattern["pattern_id"]
        area = pattern.get("area", "general")
        typical_time = pattern.get("typical_time", "00:00")  # HH:MM format
        variance_minutes = pattern.get("variance_minutes", 0)
        frequency = pattern.get("frequency", 0)
        total_days = pattern.get("total_days", 1)
        associated_signals = pattern.get("associated_signals", [])
        llm_description = pattern.get("llm_description", "No description available")

        # Parse typical time
        time_parts = typical_time.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])

        # Generate automation YAML structure
        # Trigger: Time pattern (typical time ± variance)
        automation_yaml = {
            "alias": f"Pattern: {pattern.get('name', pattern_id)}",
            "description": f"Auto-generated from detected pattern. {llm_description}",
            "trigger": [
                {
                    "platform": "time",
                    "at": f"{hour:02d}:{minute:02d}:00"
                }
            ],
            "condition": [],
            "action": []
        }

        # Extract actions from associated signals
        actions = self._signals_to_actions(area, associated_signals)
        automation_yaml["action"] = actions

        # Check for restricted domains
        requires_explicit_approval = self._check_safety_guardrails(actions)

        # Generate unique suggestion ID (hash of pattern_id + timestamp)
        suggestion_id = hashlib.sha256(
            f"{pattern_id}_{typical_time}".encode()
        ).hexdigest()[:16]

        # Calculate confidence (based on pattern frequency)
        confidence = frequency / total_days if total_days > 0 else 0

        suggestion = {
            "suggestion_id": suggestion_id,
            "pattern_id": pattern_id,
            "automation_yaml": automation_yaml,
            "confidence": confidence,
            "status": "pending",
            "requires_explicit_approval": requires_explicit_approval,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "area": area,
                "typical_time": typical_time,
                "variance_minutes": variance_minutes,
                "frequency": frequency,
                "total_days": total_days,
                "llm_description": llm_description
            }
        }

        if requires_explicit_approval:
            self.logger.warning(
                f"Suggestion {suggestion_id} requires explicit approval "
                f"(restricted domains detected)"
            )

        return suggestion

    def _signals_to_actions(self, area: str, signals: List[str]) -> List[Dict[str, Any]]:
        """Convert associated signals into Home Assistant actions.

        Args:
            area: Area name
            signals: List of signal strings (e.g., "bedroom_light_on_h7")

        Returns:
            List of HA action dictionaries
        """
        actions = []

        # Parse signals to extract entities and states
        for signal in signals:
            # Signal format: "{area}_{entity_type}_{state}_h{hour}"
            parts = signal.split("_")

            if len(parts) < 3:
                continue

            signal_area = parts[0]
            entity_type = parts[1]
            state = parts[2]

            # Only process signals for this area
            if signal_area != area:
                continue

            # Map to HA service calls
            if entity_type == "light":
                if state == "on":
                    actions.append({
                        "service": "light.turn_on",
                        "target": {
                            "area_id": area
                        }
                    })
                elif state == "off":
                    actions.append({
                        "service": "light.turn_off",
                        "target": {
                            "area_id": area
                        }
                    })

        # Default action if no signals parsed
        if not actions:
            actions.append({
                "service": "notify.persistent_notification",
                "data": {
                    "message": f"Pattern detected in {area} at typical time",
                    "title": f"{area.title()} Pattern"
                }
            })

        return actions

    def _check_safety_guardrails(self, actions: List[Dict[str, Any]]) -> bool:
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

    async def approve_suggestion(self, suggestion_id: str) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": "No suggestions found in cache"
            }

        suggestions = suggestions_cache["data"].get("suggestions", [])
        suggestion = None
        for s in suggestions:
            if s["suggestion_id"] == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            return {
                "success": False,
                "error": f"Suggestion {suggestion_id} not found"
            }

        # 2. Check if already approved
        if suggestion["status"] == "approved":
            return {
                "success": False,
                "error": "Suggestion already approved",
                "automation_id": suggestion.get("automation_id")
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
            await self.hub.set_cache("automation_suggestions", {
                "suggestions": suggestions,
                "count": len(suggestions)
            })

            # 6. Track created automation
            await self._track_created_automation(automation_id, suggestion_id)

            # 7. Publish approval event
            await self.hub.publish("automation_approved", {
                "suggestion_id": suggestion_id,
                "automation_id": automation_id,
                "pattern_id": suggestion["pattern_id"]
            })

            self.logger.info(f"Suggestion {suggestion_id} approved, automation {automation_id} created")

            return {
                "success": True,
                "automation_id": automation_id,
                "suggestion_id": suggestion_id
            }

        except Exception as e:
            self.logger.error(f"Failed to approve suggestion {suggestion_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def reject_suggestion(self, suggestion_id: str) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": "No suggestions found in cache"
            }

        suggestions = suggestions_cache["data"].get("suggestions", [])
        suggestion = None
        for s in suggestions:
            if s["suggestion_id"] == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            return {
                "success": False,
                "error": f"Suggestion {suggestion_id} not found"
            }

        # 2. Update status
        suggestion["status"] = "rejected"
        suggestion["rejected_at"] = datetime.now().isoformat()

        # 3. Save updated suggestions
        await self.hub.set_cache("automation_suggestions", {
            "suggestions": suggestions,
            "count": len(suggestions)
        })

        # 4. Publish rejection event
        await self.hub.publish("automation_rejected", {
            "suggestion_id": suggestion_id,
            "pattern_id": suggestion["pattern_id"]
        })

        self.logger.info(f"Suggestion {suggestion_id} rejected")

        return {
            "success": True,
            "suggestion_id": suggestion_id
        }

    async def _create_automation(
        self,
        automation_id: str,
        automation_yaml: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    return {
                        "success": True,
                        "automation_id": automation_id
                    }
                elif response.status == 401:
                    # Expected for standard tokens - log automation for manual creation
                    self.logger.warning(
                        f"Automation creation requires admin token (HTTP 401). "
                        f"Automation YAML logged for manual creation."
                    )
                    self.logger.info(
                        f"Manual automation YAML for {automation_id}:\n"
                        f"{json.dumps(automation_yaml, indent=2)}"
                    )

                    # Store automation YAML in cache for dashboard to display
                    await self._store_pending_automation(automation_id, automation_yaml)

                    return {
                        "success": True,  # Success = stored for manual creation
                        "automation_id": automation_id,
                        "manual_creation_required": True,
                        "note": "Automation stored for manual creation (admin token required)"
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(
                        f"Failed to create automation {automation_id}: "
                        f"HTTP {response.status} - {error_text}"
                    )
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP request failed for automation {automation_id}: {e}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error creating automation {automation_id}: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    async def _store_pending_automation(
        self,
        automation_id: str,
        automation_yaml: Dict[str, Any]
    ):
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

        pending_data["automations"][automation_id] = {
            "yaml": automation_yaml,
            "created_at": datetime.now().isoformat()
        }

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
            "status": "active"
        }

        # Save tracking data
        await self.hub.set_cache("created_automations", tracking_data)

    async def update_pattern_detection_sensor(
        self,
        pattern_name: str,
        pattern_id: str,
        confidence: float
    ):
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
                "icon": "mdi:brain"
            }
        }

        try:
            async with self._session.post(url, json=sensor_state) as response:
                if response.status == 200:
                    self.logger.debug(f"Updated pattern sensor: {pattern_name}")
                else:
                    error_text = await response.text()
                    self.logger.warning(
                        f"Failed to update pattern sensor: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to update pattern sensor: {e}")

    async def get_suggestions(
        self,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
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

    async def get_created_automations(self) -> Dict[str, Any]:
        """Get tracking data for created automations.

        Returns:
            Dictionary mapping automation ID to metadata
        """
        tracking_cache = await self.hub.get_cache("created_automations")
        if not tracking_cache or "data" not in tracking_cache:
            return {}

        return tracking_cache["data"].get("automations", {})
