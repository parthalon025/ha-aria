"""Real-time detector — subscribes to state_changed events and evaluates behavioral states.

Builds an entity-indexed lookup from BehavioralStateDefinitions loaded from the store.
When a trigger fires, creates an ActiveState. Confirming signals update it.
A periodic timer expires windows and records observations in trackers.

Lessons applied:
  - #37: Store callback ref on self, unsubscribe in shutdown()
  - #39: Domain filter before any async work
  - #28: Resource acquisition in initialize(), not __init__()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from aria.capabilities import Capability
from aria.hub.core import IntelligenceHub, Module
from aria.iw.models import ActiveState, BehavioralStateDefinition, BehavioralStateTracker
from aria.iw.store import BehavioralStateStore

logger = logging.getLogger(__name__)

# Default minimum match ratio to record an observation
_DEFAULT_MIN_MATCH_RATIO = 0.5

# Default expiry check interval in seconds
_DEFAULT_EXPIRY_INTERVAL_S = 30


class IWDetector(Module):
    """Real-time behavioral state detector.

    Subscribes to state_changed events on the hub event bus. Uses an
    entity-indexed O(1) lookup to match events against loaded definitions.
    """

    module_id = "iw_detector"

    CAPABILITIES = [
        Capability(
            id="iw_detector",
            name="I&W Real-time Detector",
            description="Monitors state_changed events and detects behavioral state activations in real time",
            module="iw_detector",
            layer="hub",
            status="experimental",
            pipeline_stage="shadow",
        ),
    ]

    def __init__(self, module_id: str, hub: IntelligenceHub, store: BehavioralStateStore) -> None:
        super().__init__(module_id, hub)
        self._store = store

        # Built during initialize / refresh
        self._definitions: list[BehavioralStateDefinition] = []
        self._entity_index: dict[str, list[tuple[str, str]]] = {}  # entity_id -> [(defn_id, role)]
        self._definition_map: dict[str, BehavioralStateDefinition] = {}  # defn_id -> defn
        self._domain_set: set[str] = set()

        # Runtime state
        self._active_states: list[ActiveState] = []
        self._expiry_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._state_changed_callback = self._on_state_changed  # lesson #37: store ref

    async def initialize(self) -> None:
        """Load definitions, build indexes, replay cold-start events, subscribe, start timer."""
        await self._load_definitions()
        await self._cold_start_replay()
        self.hub.subscribe("state_changed", self._state_changed_callback)
        self._expiry_task = asyncio.create_task(self._expiry_loop())
        self.logger.info(
            "IWDetector initialized: %d definitions, %d indexed entities, %d replayed ActiveStates, domains=%s",
            len(self._definitions),
            len(self._entity_index),
            len(self._active_states),
            sorted(self._domain_set),
        )

    async def shutdown(self) -> None:
        """Unsubscribe from events and cancel expiry timer."""
        self.hub.unsubscribe("state_changed", self._state_changed_callback)  # lesson #37
        if self._expiry_task and not self._expiry_task.done():
            self._expiry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._expiry_task
        self._active_states.clear()
        self.logger.info("IWDetector shut down")

    # ── Event handling ─────────────────────────────────────────────────

    async def _on_state_changed(self, data: dict[str, Any]) -> None:
        """Handle a state_changed event from the hub event bus.

        Flow:
        1. Domain filter (lesson #39)
        2. Person departure check
        3. Entity index lookup (O(1))
        4. Dispatch to trigger/confirming handlers
        """
        entity_id = data.get("entity_id", "")
        domain = data.get("domain") or entity_id.split(".")[0] if entity_id else ""

        # Domain filter — drop events outside our interest set
        if domain not in self._domain_set:
            return

        new_state = data.get("new_state", "")

        # Person departure handling
        if domain == "person" and new_state in ("away", "not_home"):
            self._handle_person_departure(entity_id)
            return

        # Entity index lookup
        entries = self._entity_index.get(entity_id)
        if not entries:
            return

        now = datetime.now(UTC)
        for defn_id, role in entries:
            defn = self._definition_map.get(defn_id)
            if not defn:
                continue
            if role == "trigger":
                self._handle_trigger(defn, now, new_state)
            elif role == "confirming":
                self._handle_confirming(defn, entity_id, new_state)

    def _handle_trigger(self, defn: BehavioralStateDefinition, now: datetime, new_state: str) -> None:
        """Create an ActiveState if the trigger's expected_state matches."""
        if defn.trigger.expected_state and new_state != defn.trigger.expected_state:
            return
        window_end = now + timedelta(minutes=defn.typical_duration_minutes)
        pending = [ind.entity_id for ind in defn.confirming]
        active = ActiveState(
            definition_id=defn.id,
            trigger_time=now.isoformat(),
            matched_confirming=[],
            pending_confirming=pending,
            window_expires=window_end.isoformat(),
        )
        self._active_states.append(active)
        self.logger.debug("ActiveState created for %s", defn.id)

    def _handle_confirming(self, defn: BehavioralStateDefinition, entity_id: str, new_state: str) -> None:
        """Update ActiveStates that have this entity pending confirmation."""
        for active in self._active_states:
            if active.definition_id != defn.id:
                continue
            if entity_id not in active.pending_confirming:
                continue
            conf_ind = next((ind for ind in defn.confirming if ind.entity_id == entity_id), None)
            if conf_ind and conf_ind.expected_state and new_state != conf_ind.expected_state:
                continue
            active.confirm_indicator(entity_id)
            self.logger.debug("Confirmed %s for %s", entity_id, defn.id)

    def _handle_person_departure(self, person_entity: str) -> None:
        """Terminate ActiveStates attributed to a departing person."""
        before = len(self._active_states)
        self._active_states = [
            s
            for s in self._active_states
            if self._definition_map.get(s.definition_id, _SENTINEL_DEFN).person_attribution != person_entity
        ]
        removed = before - len(self._active_states)
        if removed:
            self.logger.info("Terminated %d ActiveState(s) for departing %s", removed, person_entity)

    # ── Expiry ─────────────────────────────────────────────────────────

    async def _expiry_loop(self) -> None:
        """Periodic loop that checks for expired ActiveStates."""
        interval = await self.hub.cache.get_config_value("iw.expiry_check_interval_seconds", _DEFAULT_EXPIRY_INTERVAL_S)
        while True:
            await asyncio.sleep(interval)
            await self._check_expiry()

    async def _check_expiry(self) -> None:
        """Evaluate expired ActiveStates: record observation or discard."""
        now = datetime.now(UTC)
        min_ratio = await self.hub.cache.get_config_value("iw.min_match_ratio", _DEFAULT_MIN_MATCH_RATIO)

        still_active: list[ActiveState] = []
        for state in self._active_states:
            expires = datetime.fromisoformat(state.window_expires)
            if now < expires:
                still_active.append(state)
                continue

            # Window expired — evaluate
            ratio = state.match_ratio
            if ratio >= min_ratio:
                await self._record_observation(state, ratio)
                self.logger.info("Observation recorded for %s (ratio=%.2f)", state.definition_id, ratio)
            else:
                self.logger.debug(
                    "Discarded ActiveState for %s (ratio=%.2f < %.2f)",
                    state.definition_id,
                    ratio,
                    min_ratio,
                )

        self._active_states = still_active

    async def _record_observation(self, state: ActiveState, match_ratio: float) -> None:
        """Record an observation in the tracker for this definition."""
        tracker = await self._store.get_tracker(state.definition_id)
        if tracker is None:
            tracker = BehavioralStateTracker(definition_id=state.definition_id)
        tracker.record_observation(state.trigger_time, match_ratio)
        await self._store.save_tracker(tracker)

    # ── Definition management ──────────────────────────────────────────

    async def refresh_definitions(self) -> None:
        """Reload definitions from store and rebuild indexes."""
        await self._load_definitions()
        self.logger.info(
            "Definitions refreshed: %d definitions, %d indexed entities",
            len(self._definitions),
            len(self._entity_index),
        )

    async def _cold_start_replay(self) -> None:
        """Replay recent events from EventStore to reconstruct ActiveStates after restart.

        Queries EventStore for events in the last `iw.cold_start_replay_minutes`,
        then feeds each through `_on_state_changed()` in timestamp order.
        This addresses lesson #5: event-driven state machines must seed current state on startup.
        """
        if not hasattr(self.hub, "event_store") or self.hub.event_store is None:
            return

        replay_minutes = await self.hub.cache.get_config_value("iw.cold_start_replay_minutes", 15)
        now = datetime.now(UTC)
        start = (now - timedelta(minutes=replay_minutes)).isoformat()
        end = now.isoformat()

        events = await self.hub.event_store.query_events(start, end)
        if not events:
            self.logger.debug("Cold-start replay: no events in last %d minutes", replay_minutes)
            return

        for event in events:
            await self._on_state_changed(event)

        self.logger.info(
            "Cold-start replay: processed %d events, created %d ActiveState(s)",
            len(events),
            len(self._active_states),
        )

    async def _load_definitions(self) -> None:
        """Load all definitions from store and build entity index + domain set."""
        self._definitions = await self._store.list_definitions()
        self._definition_map = {d.id: d for d in self._definitions}
        self._entity_index = {}
        self._domain_set = {"person"}  # always include person for departure detection

        for defn in self._definitions:
            # Index trigger entity
            trigger_eid = defn.trigger.entity_id
            self._entity_index.setdefault(trigger_eid, []).append((defn.id, "trigger"))
            self._domain_set.add(trigger_eid.split(".")[0])

            # Index confirming entities
            for ind in defn.confirming:
                self._entity_index.setdefault(ind.entity_id, []).append((defn.id, "confirming"))
                self._domain_set.add(ind.entity_id.split(".")[0])


# Sentinel for person_attribution check when definition not found
class _SentinelDefn:
    person_attribution = None


_SENTINEL_DEFN = _SentinelDefn()
