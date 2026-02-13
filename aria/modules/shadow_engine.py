"""Shadow Engine Module - Predict-compare-score loop for shadow mode.

Captures context snapshots when significant events arrive, generates
predictions using ML models and frequency heuristics, tracks open
predictions within time windows, and scores outcomes when windows expire.

Exploration strategy (configurable via shadow.explore_strategy):
- "epsilon" (default): Fixed 80/20 epsilon-greedy explore/exploit
- "thompson": Thompson Sampling with Beta posterior per context bucket
  (Cavenaghi et al., Entropy 2021 — validated for non-stationary bandits)
"""

import asyncio
import logging
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from aria.hub.core import Module, IntelligenceHub
from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY

logger = logging.getLogger(__name__)

# Minimum confidence to store a prediction (below this, skip)
MIN_CONFIDENCE = 0.3

# Default evaluation window in seconds (10 minutes)
DEFAULT_WINDOW_SECONDS = 600

# How often to resolve expired prediction windows (seconds)
RESOLUTION_INTERVAL_S = 60

# Minimum seconds between prediction attempts (debounce rapid events)
PREDICTION_COOLDOWN_S = 30

# Domains worth tracking for predictions
PREDICTABLE_DOMAINS = {
    "light", "switch", "media_player", "cover", "climate",
    "vacuum", "fan", "lock",
}

# Domains used for room detection
ROOM_INDICATOR_DOMAINS = {
    "light", "switch", "binary_sensor", "media_player", "fan",
}


class ThompsonSampler:
    """Thompson Sampling for explore/exploit decisions using Beta posteriors.

    Maintains per-context-bucket success/failure counts. On each decision,
    samples from Beta(alpha, beta) for each bucket and picks the best.

    For the shadow engine, "explore" means making predictions in contexts
    where we have less data, while "exploit" uses highest-confidence methods.

    Reference: Cavenaghi et al., "f-dsw Thompson Sampling" (Entropy 2021).
    This is the basic version; f-dsw adaptation can be layered later.
    """

    def __init__(self):
        # Maps bucket_key -> {"alpha": successes+1, "beta": failures+1}
        self._buckets: Dict[str, Dict[str, float]] = {}

    def get_bucket_key(self, context: Dict[str, Any]) -> str:
        """Derive a bucket key from context features.

        Uses hour-of-day quantized to 4 time bands + presence.
        """
        time_features = context.get("time_features", {})
        hour_sin = time_features.get("hour_sin", 0)
        # Quantize to 4 periods: night(0), morning(1), afternoon(2), evening(3)
        # Using hour_sin: >0.5 = morning, <-0.5 = evening, etc.
        if hour_sin > 0.5:
            period = "morning"
        elif hour_sin > -0.5:
            period = "afternoon" if time_features.get("hour_cos", 0) < 0 else "night"
        else:
            period = "evening"

        presence = context.get("presence", {})
        home = "home" if presence.get("home") else "away"
        return f"{period}_{home}"

    def should_explore(self, context: Dict[str, Any]) -> bool:
        """Decide whether to explore (try less-tested methods) or exploit.

        Samples from Beta posteriors for this context bucket.
        Returns True if the explore arm wins the sample.
        """
        key = self.get_bucket_key(context)
        bucket = self._buckets.get(key, {"alpha": 1.0, "beta": 1.0})

        # Sample from Beta posterior
        exploit_sample = random.betavariate(bucket["alpha"], bucket["beta"])
        # Explore arm has a flat prior (less informed = more uncertain)
        explore_sample = random.betavariate(1.0, 1.0)

        return explore_sample > exploit_sample

    def record_outcome(self, context: Dict[str, Any], success: bool):
        """Update the posterior for this context bucket.

        Args:
            context: The prediction context.
            success: Whether the prediction was correct.
        """
        key = self.get_bucket_key(context)
        if key not in self._buckets:
            self._buckets[key] = {"alpha": 1.0, "beta": 1.0}

        if success:
            self._buckets[key]["alpha"] += 1.0
        else:
            self._buckets[key]["beta"] += 1.0

    def get_stats(self) -> Dict[str, Any]:
        """Return current bucket statistics for observability."""
        return {
            key: {
                "alpha": round(b["alpha"], 1),
                "beta": round(b["beta"], 1),
                "mean": round(b["alpha"] / (b["alpha"] + b["beta"]), 3),
                "trials": int(b["alpha"] + b["beta"] - 2),
            }
            for key, b in self._buckets.items()
        }


class ShadowEngine(Module):
    """Shadow mode prediction engine: predict-compare-score loop."""

    def __init__(self, hub: IntelligenceHub):
        super().__init__("shadow_engine", hub)

        # Recent events buffer for context capture (last 5 minutes)
        self._recent_events: List[Dict[str, Any]] = []
        self._recent_events_max_age_s = 300  # 5 minutes

        # Cooldown tracking
        self._last_prediction_time: Optional[datetime] = None

        # Resolution task handle
        self._resolution_task: Optional[asyncio.Task] = None

        # Track events that occurred during open prediction windows
        # Maps prediction_id -> list of events that happened in that window
        self._window_events: Dict[str, List[Dict[str, Any]]] = {}

        # Thompson Sampling for explore/exploit
        self._thompson = ThompsonSampler()

    async def initialize(self):
        """Subscribe to state_changed events and start periodic resolution."""
        self.logger.info("Shadow engine initializing...")

        # Subscribe to state_changed events on the hub event bus
        self.hub.subscribe("state_changed", self._on_state_changed)

        # Start periodic resolution task
        self._resolution_task = asyncio.create_task(
            self._resolution_loop()
        )

        self.logger.info("Shadow engine initialized")

    async def shutdown(self):
        """Cancel the resolution task and unsubscribe from events."""
        self.hub.unsubscribe("state_changed", self._on_state_changed)

        if self._resolution_task and not self._resolution_task.done():
            self._resolution_task.cancel()
            try:
                await self._resolution_task
            except asyncio.CancelledError:
                pass
            self._resolution_task = None

        self.logger.info("Shadow engine shut down")

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        """Not used — shadow engine listens via hub.subscribe() instead.

        Using on_event AND subscribe would cause double-handling since
        hub.publish() invokes both subscriber callbacks and on_event
        for all registered modules.
        """
        pass

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_state_changed(self, data: Dict[str, Any]):
        """Handle a state_changed event from the activity monitor.

        Buffers the event, records it against open prediction windows,
        and generates new predictions if cooldown has elapsed.

        Args:
            data: Event data with entity_id, domain, from, to, timestamp, etc.
        """
        now = datetime.now()

        # Build normalized event record
        entity_id = data.get("entity_id", "")
        domain = entity_id.split(".")[0] if "." in entity_id else ""

        # Phase 2: Check entity curation — skip excluded entities
        included_ids = await self.hub.cache.get_included_entity_ids()
        if included_ids and entity_id not in included_ids:
            return

        # Extract state values from either flat or nested format
        if "new_state" in data:
            # Nested format from HA WebSocket
            new_state = data.get("new_state", {})
            old_state = data.get("old_state", {})
            to_state = new_state.get("state", "")
            from_state = old_state.get("state", "")
            friendly_name = new_state.get("attributes", {}).get(
                "friendly_name", entity_id
            )
        else:
            # Flat format from activity_monitor buffer
            to_state = data.get("to", "")
            from_state = data.get("from", "")
            friendly_name = data.get("friendly_name", entity_id)

        event = {
            "entity_id": entity_id,
            "domain": domain,
            "from": from_state,
            "to": to_state,
            "timestamp": now.isoformat(),
            "seconds_ago": 0,
            "friendly_name": friendly_name,
        }

        # Add to recent events buffer
        self._recent_events.append(event)
        self._prune_recent_events(now)

        # Record event against all open prediction windows
        for pred_id in list(self._window_events.keys()):
            self._window_events[pred_id].append(event)

        # Check cooldown before generating predictions (config with constant fallback)
        cooldown = await self.hub.cache.get_config_value(
            "shadow.prediction_cooldown_s", PREDICTION_COOLDOWN_S
        ) or PREDICTION_COOLDOWN_S
        if self._last_prediction_time:
            elapsed = (now - self._last_prediction_time).total_seconds()
            if elapsed < cooldown:
                return

        # Only predict on actionable domains
        if domain not in PREDICTABLE_DOMAINS:
            return

        # Generate predictions
        try:
            context = await self._capture_context(data)
            predictions = await self._generate_predictions(context)

            if predictions:
                # Determine explore/exploit via configured strategy
                explore_strategy = await self.hub.cache.get_config_value(
                    "shadow.explore_strategy", "epsilon"
                ) or "epsilon"

                if explore_strategy == "thompson":
                    is_exploration = self._thompson.should_explore(context)
                else:
                    # Default epsilon-greedy (80% exploit, 20% explore)
                    is_exploration = random.random() < 0.2

                await self._store_predictions(
                    context, predictions, is_exploration=is_exploration
                )
                self._last_prediction_time = now
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")

    def _prune_recent_events(self, now: datetime):
        """Remove events older than the max age from the buffer."""
        cutoff = now - timedelta(seconds=self._recent_events_max_age_s)
        cutoff_iso = cutoff.isoformat()
        self._recent_events = [
            e for e in self._recent_events
            if e.get("timestamp", "") >= cutoff_iso
        ]

    # ------------------------------------------------------------------
    # Context capture
    # ------------------------------------------------------------------

    async def _capture_context(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Build a context snapshot from current state.

        Args:
            trigger_event: The event that triggered this context capture.

        Returns:
            Context snapshot dictionary.
        """
        now = datetime.now()

        # Time features (sin/cos encoding for cyclical patterns)
        time_features = self._compute_time_features(now)

        # Presence from activity_summary cache
        presence = await self._get_presence()

        # Recent events from buffer (with seconds_ago computed)
        recent_events = self._get_recent_events_snapshot(now)

        # Current states of key tracked entities
        current_states = await self._get_current_states()

        # Rolling stats from activity log
        rolling_stats = await self._get_rolling_stats()

        return {
            "timestamp": now.isoformat(),
            "time_features": time_features,
            "presence": presence,
            "recent_events": recent_events,
            "current_states": current_states,
            "rolling_stats": rolling_stats,
            "trigger_event": {
                "entity_id": trigger_event.get("entity_id", ""),
                "domain": trigger_event.get("entity_id", "").split(".")[0]
                if "." in trigger_event.get("entity_id", "")
                else "",
            },
        }

    def _compute_time_features(self, dt: datetime) -> Dict[str, float]:
        """Compute sin/cos time features for cyclical encoding.

        Args:
            dt: Current datetime.

        Returns:
            Dict with hour_sin, hour_cos, dow_sin, dow_cos.
        """
        hour_angle = 2 * math.pi * dt.hour / 24
        dow_angle = 2 * math.pi * dt.weekday() / 7

        return {
            "hour_sin": round(math.sin(hour_angle), 6),
            "hour_cos": round(math.cos(hour_angle), 6),
            "dow_sin": round(math.sin(dow_angle), 6),
            "dow_cos": round(math.cos(dow_angle), 6),
        }

    async def _get_presence(self) -> Dict[str, Any]:
        """Get presence info from activity_summary cache.

        Returns:
            Dict with home (bool) and rooms (list of active room names).
        """
        summary = await self.hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        if not summary or not summary.get("data"):
            return {"home": False, "rooms": []}

        data = summary["data"]
        occupancy = data.get("occupancy", {})
        anyone_home = occupancy.get("anyone_home", False)

        # Derive active rooms from recent activity domains
        rooms = []
        recent = data.get("recent_activity", [])
        for evt in recent:
            entity = evt.get("entity", "")
            name = evt.get("friendly_name", "")
            room = self._extract_room(entity, name)
            if room and room not in rooms:
                rooms.append(room)

        return {"home": anyone_home, "rooms": rooms[:5]}

    def _extract_room(self, entity_id: str, friendly_name: str) -> Optional[str]:
        """Extract room name from entity ID or friendly name.

        Args:
            entity_id: HA entity ID.
            friendly_name: Human-readable name.

        Returns:
            Room name or None.
        """
        room_keywords = [
            "bedroom", "kitchen", "living", "bathroom", "closet",
            "office", "garage", "hallway", "dining", "basement",
            "porch", "patio", "laundry",
        ]

        text = f"{entity_id} {friendly_name}".lower()
        for room in room_keywords:
            if room in text:
                return room

        return None

    def _get_recent_events_snapshot(self, now: datetime) -> List[Dict[str, Any]]:
        """Get recent events with seconds_ago computed.

        Args:
            now: Current time.

        Returns:
            List of recent event dicts with seconds_ago field.
        """
        result = []
        for evt in self._recent_events[-20:]:  # Last 20 events max
            ts_str = evt.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
                seconds_ago = (now - ts).total_seconds()
            except (ValueError, TypeError):
                seconds_ago = 0

            result.append({
                "domain": evt.get("domain", ""),
                "entity": evt.get("entity_id", ""),
                "state": evt.get("to", ""),
                "seconds_ago": round(seconds_ago),
            })

        return result

    async def _get_current_states(self) -> Dict[str, str]:
        """Get current states of key entities from cache.

        Returns:
            Dict mapping entity_id to current state string.
        """
        summary = await self.hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        if not summary or not summary.get("data"):
            return {}

        states = {}
        recent = summary["data"].get("recent_activity", [])
        for evt in recent:
            entity = evt.get("entity", "")
            if entity:
                states[entity] = evt.get("to", "")

        return states

    async def _get_rolling_stats(self) -> Dict[str, float]:
        """Compute rolling statistics from activity log.

        Returns:
            Dict with 1h_event_count, 1h_domain_entropy, 1h_dominant_domain_pct.
        """
        activity_log = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        if not activity_log or not activity_log.get("data"):
            return {
                "1h_event_count": 0,
                "1h_domain_entropy": 0.0,
                "1h_dominant_domain_pct": 0.0,
            }

        windows = activity_log["data"].get("windows", [])
        if not windows:
            return {
                "1h_event_count": 0,
                "1h_domain_entropy": 0.0,
                "1h_dominant_domain_pct": 0.0,
            }

        # Filter to last 1 hour
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
        recent = [w for w in windows if w.get("window_start", "") >= cutoff]

        if not recent:
            return {
                "1h_event_count": 0,
                "1h_domain_entropy": 0.0,
                "1h_dominant_domain_pct": 0.0,
            }

        total_events = sum(w.get("event_count", 0) for w in recent)

        # Aggregate domain counts
        domain_counts: Dict[str, int] = {}
        for w in recent:
            for domain, count in w.get("by_domain", {}).items():
                domain_counts[domain] = domain_counts.get(domain, 0) + count

        # Domain entropy
        entropy = 0.0
        dominant_pct = 0.0
        if total_events > 0 and domain_counts:
            for count in domain_counts.values():
                p = count / total_events
                if p > 0:
                    entropy -= p * math.log2(p)
            dominant_pct = max(domain_counts.values()) / total_events

        return {
            "1h_event_count": total_events,
            "1h_domain_entropy": round(entropy, 4),
            "1h_dominant_domain_pct": round(dominant_pct, 4),
        }

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    async def _generate_predictions(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate predictions using ML models and frequency heuristics.

        Produces up to 3 prediction types:
        1. next_domain_action - which domain acts next
        2. room_activation - which room becomes active
        3. routine_trigger - is a known routine about to start

        Args:
            context: Context snapshot from _capture_context().

        Returns:
            List of prediction dicts (may be empty if all below threshold).
        """
        predictions = []

        # Phase 2: read min confidence from config store (constant as fallback)
        min_conf = await self.hub.cache.get_config_value(
            "shadow.min_confidence", MIN_CONFIDENCE
        ) or MIN_CONFIDENCE

        # 1. Next domain action prediction
        domain_pred = await self._predict_next_domain(context)
        if domain_pred and domain_pred.get("confidence", 0) >= min_conf:
            predictions.append(domain_pred)

        # 2. Room activation prediction
        room_pred = await self._predict_room_activation(context)
        if room_pred and room_pred.get("confidence", 0) >= min_conf:
            predictions.append(room_pred)

        # 3. Routine trigger prediction
        routine_pred = await self._predict_routine_trigger(context)
        if routine_pred and routine_pred.get("confidence", 0) >= min_conf:
            predictions.append(routine_pred)

        return predictions

    async def _predict_next_domain(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Predict which domain will produce the next action.

        Uses ML engine models if available, falls back to frequency analysis
        from the activity monitor's event sequence prediction.

        Args:
            context: Context snapshot.

        Returns:
            Prediction dict or None.
        """
        # Phase 2: read window from config store (constant as fallback)
        window_s = await self.hub.cache.get_config_value(
            "shadow.default_window_seconds", DEFAULT_WINDOW_SECONDS
        ) or DEFAULT_WINDOW_SECONDS

        # Try activity_summary event_predictions (frequency-based)
        summary = await self.hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        if summary and summary.get("data"):
            event_preds = summary["data"].get("event_predictions", {})
            if event_preds.get("predicted_next_domain"):
                probability = event_preds.get("probability", 0)
                return {
                    "type": "next_domain_action",
                    "predicted": event_preds["predicted_next_domain"],
                    "confidence": probability,
                    "method": event_preds.get("method", "frequency"),
                    "window_seconds": window_s,
                }

        # Fall back to simple frequency from recent events
        # Phase 2: filter recent events by included entity set
        included_ids = await self.hub.cache.get_included_entity_ids()
        recent = context.get("recent_events", [])
        if included_ids:
            recent = [e for e in recent if e.get("entity", "") in included_ids or not included_ids]

        if not recent:
            return None

        domain_counts: Dict[str, int] = defaultdict(int)
        for evt in recent:
            domain = evt.get("domain", "")
            if domain in PREDICTABLE_DOMAINS:
                domain_counts[domain] += 1

        if not domain_counts:
            return None

        top_domain = max(domain_counts, key=domain_counts.get)
        total = sum(domain_counts.values())
        confidence = domain_counts[top_domain] / total if total > 0 else 0

        return {
            "type": "next_domain_action",
            "predicted": top_domain,
            "confidence": round(confidence, 3),
            "method": "recent_frequency",
            "window_seconds": window_s,
        }

    async def _predict_room_activation(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Predict which room will become active next.

        Based on recent event rooms and presence data.

        Args:
            context: Context snapshot.

        Returns:
            Prediction dict or None.
        """
        recent = context.get("recent_events", [])
        presence = context.get("presence", {})
        current_rooms = presence.get("rooms", [])

        # Count room mentions in recent events
        room_counts: Dict[str, int] = defaultdict(int)
        for evt in recent:
            entity = evt.get("entity", "")
            room = self._extract_room(entity, "")
            if room:
                room_counts[room] += 1

        if not room_counts:
            return None

        # Predict the room with most recent activity that isn't already
        # the most recently active (predict *next*, not current)
        sorted_rooms = sorted(
            room_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Pick the top room — if we have multiple rooms, predict the second
        # most active (it's likely to get more attention next)
        predicted_room = sorted_rooms[0][0]
        if len(sorted_rooms) > 1 and current_rooms:
            # If the top room is already the most recently active,
            # predict the next one
            if sorted_rooms[0][0] in current_rooms[:1]:
                predicted_room = sorted_rooms[1][0]

        total = sum(room_counts.values())
        confidence = room_counts.get(predicted_room, 0) / total if total > 0 else 0

        return {
            "type": "room_activation",
            "predicted": predicted_room,
            "confidence": round(confidence, 3),
            "method": "activity_frequency",
            "window_seconds": DEFAULT_WINDOW_SECONDS,
        }

    async def _predict_routine_trigger(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Predict whether a known routine is about to start.

        Uses cached pattern data to check if current time/context
        matches a known behavioral pattern.

        Args:
            context: Context snapshot.

        Returns:
            Prediction dict or None.
        """
        # Load patterns from cache
        patterns_cache = await self.hub.get_cache("patterns")
        if not patterns_cache or not patterns_cache.get("data"):
            return None

        patterns = patterns_cache["data"].get("patterns", [])
        if not patterns:
            return None

        now = datetime.now()
        current_minutes = now.hour * 60 + now.minute

        best_match = None
        best_confidence = 0.0

        for pattern in patterns:
            typical_time = pattern.get("typical_time", "")
            if not typical_time:
                continue

            try:
                parts = typical_time.split(":")
                pattern_minutes = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                continue

            # Check if we're within the pattern's variance window
            variance = pattern.get("variance_minutes", 30)
            distance = abs(current_minutes - pattern_minutes)
            # Handle midnight wrap
            distance = min(distance, 1440 - distance)

            if distance <= variance:
                confidence = pattern.get("confidence", 0)
                # Scale confidence by proximity (closer = higher)
                proximity_factor = 1.0 - (distance / max(variance, 1))
                adjusted_confidence = confidence * proximity_factor

                if adjusted_confidence > best_confidence:
                    best_confidence = adjusted_confidence
                    best_match = pattern

        if not best_match or best_confidence < MIN_CONFIDENCE:
            return None

        # Extract expected domains from the pattern's associated signals
        # so the scorer can verify domain overlap instead of just event count
        expected_domains = set()
        for signal in best_match.get("associated_signals", []):
            if isinstance(signal, str) and "." in signal:
                expected_domains.add(signal.split(".")[0])

        return {
            "type": "routine_trigger",
            "predicted": best_match.get("name", "unknown"),
            "confidence": round(best_confidence, 3),
            "method": "pattern_match",
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "pattern_id": best_match.get("pattern_id", ""),
            "expected_domains": list(expected_domains),
        }

    # ------------------------------------------------------------------
    # Prediction storage
    # ------------------------------------------------------------------

    async def _store_predictions(
        self,
        context: Dict[str, Any],
        predictions: List[Dict[str, Any]],
    ):
        """Store predictions in the database and track their windows.

        Args:
            context: Context snapshot.
            predictions: List of prediction dicts from _generate_predictions().
        """
        # Compute overall confidence as average of individual predictions
        confidences = [p.get("confidence", 0) for p in predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Determine window from the predictions (use max window)
        window_seconds = max(
            p.get("window_seconds", DEFAULT_WINDOW_SECONDS) for p in predictions
        )

        prediction_id = uuid.uuid4().hex
        now = datetime.now().isoformat()

        try:
            await self.hub.cache.insert_prediction(
                prediction_id=prediction_id,
                timestamp=now,
                context=context,
                predictions=predictions,
                confidence=round(avg_confidence, 3),
                window_seconds=window_seconds,
                is_exploration=False,
            )

            # Track events during this prediction's window
            self._window_events[prediction_id] = []

            self.logger.debug(
                f"Stored prediction {prediction_id[:8]}: "
                f"{len(predictions)} predictions, "
                f"confidence={avg_confidence:.3f}, "
                f"window={window_seconds}s"
            )
        except Exception as e:
            self.logger.error(f"Failed to store prediction: {e}")

    # ------------------------------------------------------------------
    # Prediction resolution
    # ------------------------------------------------------------------

    async def _resolution_loop(self):
        """Periodic loop to resolve expired prediction windows."""
        while True:
            try:
                # Phase 2: read interval from config store (constant as fallback)
                interval = await self.hub.cache.get_config_value(
                    "shadow.resolution_interval_s", RESOLUTION_INTERVAL_S
                ) or RESOLUTION_INTERVAL_S
                await asyncio.sleep(interval)
                await self._resolve_expired_predictions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resolution loop error: {e}")

    async def _resolve_expired_predictions(self):
        """Find and score all predictions whose windows have expired."""
        try:
            pending = await self.hub.cache.get_pending_predictions()
        except Exception as e:
            self.logger.error(f"Failed to get pending predictions: {e}")
            return

        if not pending:
            return

        self.logger.debug(f"Resolving {len(pending)} expired predictions")

        for prediction in pending:
            pred_id = prediction["id"]
            actual_events = self._window_events.pop(pred_id, [])

            outcome, actual_data = self._score_prediction(
                prediction, actual_events
            )

            try:
                await self.hub.cache.update_prediction_outcome(
                    prediction_id=pred_id,
                    outcome=outcome,
                    actual=actual_data,
                )

                self.logger.debug(
                    f"Resolved {pred_id[:8]}: {outcome} "
                    f"({len(actual_events)} events in window)"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to update prediction outcome {pred_id}: {e}"
                )

        # Clean up any stale window_events entries
        self._cleanup_stale_windows()

    def _score_prediction(
        self,
        prediction: Dict[str, Any],
        actual_events: List[Dict[str, Any]],
    ) -> tuple:
        """Score a prediction against actual events.

        Args:
            prediction: Prediction record from the database.
            actual_events: Events that occurred during the prediction window.

        Returns:
            Tuple of (outcome, actual_data) where outcome is one of:
            - "correct": prediction matched actual events
            - "disagreement": events occurred but prediction was wrong
            - "nothing": prediction expected something but nothing happened
        """
        predictions_list = prediction.get("predictions", [])
        if not predictions_list:
            return "nothing", None

        # Build summary of what actually happened
        actual_domains = set()
        actual_rooms = set()
        for evt in actual_events:
            domain = evt.get("domain", "")
            if domain:
                actual_domains.add(domain)
            entity = evt.get("entity_id", "")
            room = self._extract_room(entity, "")
            if room:
                actual_rooms.add(room)

        actual_data = {
            "event_count": len(actual_events),
            "domains": list(actual_domains),
            "rooms": list(actual_rooms),
        }

        if not actual_events:
            return "nothing", actual_data

        # Score each prediction type
        any_correct = False
        for pred in predictions_list:
            pred_type = pred.get("type", "")
            predicted_value = pred.get("predicted", "")

            if pred_type == "next_domain_action":
                if predicted_value in actual_domains:
                    any_correct = True
                    break

            elif pred_type == "room_activation":
                if predicted_value in actual_rooms:
                    any_correct = True
                    break

            elif pred_type == "routine_trigger":
                # Routine triggers must show domain overlap with the
                # pattern's expected domains — not just "any 2+ events"
                expected_domains = set(pred.get("expected_domains", []))
                if expected_domains:
                    overlap = actual_domains & expected_domains
                    # At least 2 expected domains must appear
                    if len(overlap) >= 2:
                        any_correct = True
                        break
                else:
                    # No expected_domains stored — require 3+ diverse
                    # domain events as a lenient fallback
                    if (len(actual_events) >= 3
                            and len(actual_domains) >= 2):
                        any_correct = True
                        break

        if any_correct:
            return "correct", actual_data
        else:
            return "disagreement", actual_data

    def _cleanup_stale_windows(self):
        """Remove window_events entries for predictions that are no longer pending.

        This handles cases where predictions were resolved externally or
        the in-memory tracking drifted from the database.
        """
        # Keep only entries added in the last hour (safety bound)
        # In practice, entries are removed when predictions are resolved
        max_entries = 100
        if len(self._window_events) > max_entries:
            # Remove oldest entries (dict preserves insertion order in Python 3.7+)
            excess = len(self._window_events) - max_entries
            keys_to_remove = list(self._window_events.keys())[:excess]
            for key in keys_to_remove:
                del self._window_events[key]
