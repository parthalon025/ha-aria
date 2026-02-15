"""Activity Labeler Module — LLM predicts activities, user corrects, system retrains."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from aria.hub.core import Module, IntelligenceHub
from aria.capabilities import Capability

logger = logging.getLogger(__name__)

CLASSIFIER_THRESHOLD = 50  # Labels needed before training classifier
PREDICTION_INTERVAL = timedelta(minutes=15)
OLLAMA_QUEUE_URL = "http://127.0.0.1:7683"

ACTIVITY_PROMPT_TEMPLATE = """Given the current smart home state:
- Power draw: {power_watts}W
- Lights on: {lights_on}
- Motion detected: {motion_rooms}
- Time: {time_of_day} ({hour}:{minute})
- Occupancy: {occupancy}
- Recent events: {recent_events}

What activity is the resident most likely doing?
Choose from: sleeping, cooking, watching_tv, working, cleaning, eating, away, relaxing, exercising, showering, unknown.
Respond with ONLY JSON: {{"activity": "...", "confidence": 0.0-1.0}}"""


class ActivityLabeler(Module):
    """Predicts household activities from sensor state with human-in-the-loop correction."""

    CAPABILITIES = [
        Capability(
            id="activity_labeler",
            name="Activity Labeler",
            description="LLM-predicted activity labels with user correction and classifier retraining.",
            module="activity_labeler",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_activity_labeler.py"],
            systemd_units=["aria-hub.service"],
            status="experimental",
            added_version="1.1.0",
            depends_on=["discovery", "activity_analytics"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub):
        super().__init__("activity_labeler", hub)
        self._classifier = None
        self._classifier_ready = False
        self._label_encoder = None

    async def initialize(self):
        """Check if classifier is ready from cached labels, schedule periodic prediction."""
        self.logger.info("Activity labeler module initializing...")

        # Check existing labels to see if classifier can be loaded
        cache_entry = await self.hub.get_cache("activity_labels")
        if cache_entry and cache_entry.get("data"):
            stats = cache_entry["data"].get("label_stats", {})
            if stats.get("classifier_ready", False):
                labels = cache_entry["data"].get("labels", [])
                if len(labels) >= CLASSIFIER_THRESHOLD:
                    try:
                        await self._train_classifier()
                    except Exception as e:
                        self.logger.warning(f"Failed to restore classifier from cached labels: {e}")

        await self.hub.schedule_task(
            task_id="activity_labeler_predict",
            coro=self._periodic_predict,
            interval=PREDICTION_INTERVAL,
            run_immediately=False,
        )
        self.logger.info("Activity labeler module initialized")

    async def predict_activity(self, context: dict) -> dict:
        """Predict current activity from sensor context.

        Uses classifier if ready, otherwise falls back to Ollama.

        Args:
            context: Sensor context dict with keys like power_watts, lights_on, etc.

        Returns:
            Dict with predicted, confidence, method, sensor_context, predicted_at.
        """
        if self._classifier_ready and self._classifier is not None:
            try:
                features = [self._context_to_features(context)]
                prediction_idx = self._classifier.predict(features)[0]
                probabilities = self._classifier.predict_proba(features)[0]
                confidence = float(max(probabilities))
                predicted = self._label_encoder.inverse_transform([prediction_idx])[0]
                return {
                    "predicted": predicted,
                    "confidence": round(confidence, 2),
                    "method": "classifier",
                    "sensor_context": context,
                    "predicted_at": datetime.now().isoformat(),
                }
            except Exception as e:
                self.logger.warning(f"Classifier prediction failed, falling back to Ollama: {e}")

        # Fallback to Ollama
        result = await self._query_ollama(context)
        return {
            "predicted": result.get("activity", "unknown"),
            "confidence": round(result.get("confidence", 0.0), 2),
            "method": "ollama",
            "sensor_context": context,
            "predicted_at": datetime.now().isoformat(),
        }

    async def record_label(
        self,
        predicted: str,
        actual: str,
        sensor_context: dict,
        source: str = "corrected",
    ) -> dict:
        """Store a label (correction or confirmation) and update stats.

        Args:
            predicted: The activity that was predicted.
            actual: The actual activity (from user correction or confirmation).
            sensor_context: The sensor context at prediction time.
            source: "corrected" if user changed it, "confirmed" if user agreed.

        Returns:
            Updated label stats dict.
        """
        # Read existing cache
        cache_entry = await self.hub.get_cache("activity_labels")
        if cache_entry and cache_entry.get("data"):
            data = cache_entry["data"]
        else:
            data = {
                "current_activity": None,
                "labels": [],
                "label_stats": {
                    "total_labels": 0,
                    "total_corrections": 0,
                    "accuracy": 0.0,
                    "activities_seen": [],
                    "classifier_ready": False,
                    "last_trained": None,
                },
            }

        # Create label entry
        label = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now().isoformat(),
            "sensor_context": sensor_context,
            "predicted_activity": predicted,
            "actual_activity": actual,
            "source": source,
        }
        data["labels"].append(label)

        # Update stats
        stats = data["label_stats"]
        stats["total_labels"] = len(data["labels"])
        if source == "corrected":
            stats["total_corrections"] = stats.get("total_corrections", 0) + 1

        # Compute accuracy: labels where predicted == actual / total
        correct = sum(1 for l in data["labels"] if l["predicted_activity"] == l["actual_activity"])
        stats["accuracy"] = round(correct / len(data["labels"]), 3) if data["labels"] else 0.0

        # Track unique activities
        all_activities = set()
        for l in data["labels"]:
            all_activities.add(l["actual_activity"])
            all_activities.add(l["predicted_activity"])
        stats["activities_seen"] = sorted(all_activities)

        # Train classifier if threshold crossed
        if len(data["labels"]) >= CLASSIFIER_THRESHOLD and not self._classifier_ready:
            try:
                await self._train_classifier_from_labels(data["labels"])
                stats["classifier_ready"] = True
                stats["last_trained"] = datetime.now().isoformat()
                self.logger.info(
                    f"Classifier trained with {len(data['labels'])} labels, "
                    f"{len(stats['activities_seen'])} activity types"
                )
            except Exception as e:
                self.logger.error(f"Failed to train classifier: {e}")

        data["label_stats"] = stats

        await self.hub.set_cache("activity_labels", data, {"source": "activity_labeler"})

        return stats

    async def _query_ollama(self, context: dict) -> dict:
        """Query Ollama via ollama-queue for activity prediction.

        Args:
            context: Sensor context dict.

        Returns:
            Dict with activity and confidence keys.
        """
        try:
            import aiohttp

            prompt = ACTIVITY_PROMPT_TEMPLATE.format(
                power_watts=context.get("power_watts", 0),
                lights_on=context.get("lights_on", 0),
                motion_rooms=context.get("motion_rooms", "none"),
                time_of_day=self._time_of_day(),
                hour=datetime.now().hour,
                minute=datetime.now().strftime("%M"),
                occupancy=context.get("occupancy", "unknown"),
                recent_events=context.get("recent_events", "none"),
            )

            payload = {
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OLLAMA_QUEUE_URL}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response_text = result.get("response", "{}")
                        parsed = json.loads(response_text)
                        return {
                            "activity": parsed.get("activity", "unknown"),
                            "confidence": float(parsed.get("confidence", 0.5)),
                        }
                    else:
                        self.logger.warning(f"Ollama queue returned status {resp.status}")
                        return {"activity": "unknown", "confidence": 0.0}

        except Exception as e:
            self.logger.warning(f"Ollama query failed: {e}")
            return {"activity": "unknown", "confidence": 0.0}

    async def _train_classifier(self):
        """Train classifier from cached labels."""
        cache_entry = await self.hub.get_cache("activity_labels")
        if not cache_entry or not cache_entry.get("data"):
            self.logger.warning("No cached labels for classifier training")
            return

        labels = cache_entry["data"].get("labels", [])
        if len(labels) < CLASSIFIER_THRESHOLD:
            self.logger.info(f"Only {len(labels)} labels, need {CLASSIFIER_THRESHOLD} for training")
            return

        await self._train_classifier_from_labels(labels)

    async def _train_classifier_from_labels(self, labels: List[dict]):
        """Train a GradientBoostingClassifier from label data.

        Args:
            labels: List of label dicts with sensor_context and actual_activity.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder

        features = []
        targets = []

        for label in labels:
            ctx = label.get("sensor_context", {})
            feat = self._context_to_features(ctx)
            features.append(feat)
            targets.append(label["actual_activity"])

        if len(set(targets)) < 2:
            self.logger.warning("Need at least 2 distinct activity types for classifier training")
            return

        encoder = LabelEncoder()
        encoded_targets = encoder.fit_transform(targets)

        clf = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
        )
        clf.fit(features, encoded_targets)

        self._classifier = clf
        self._label_encoder = encoder
        self._classifier_ready = True

    def _context_to_features(self, ctx: dict) -> list:
        """Convert sensor context to a numeric feature vector.

        Args:
            ctx: Sensor context dict.

        Returns:
            List of floats: [power_watts, lights_on, motion_room_count, hour, is_home]
        """
        power_watts = float(ctx.get("power_watts", 0))
        lights_on = int(ctx.get("lights_on", 0))
        motion_rooms = ctx.get("motion_rooms", "")
        if isinstance(motion_rooms, list):
            motion_room_count = len(motion_rooms)
        elif isinstance(motion_rooms, str) and motion_rooms and motion_rooms != "none":
            motion_room_count = len(motion_rooms.split(","))
        else:
            motion_room_count = 0

        hour = float(ctx.get("hour", datetime.now().hour))
        occupancy = ctx.get("occupancy", "unknown")
        is_home = 1.0 if occupancy in ("home", "on", "true", True) else 0.0

        return [power_watts, float(lights_on), float(motion_room_count), hour, is_home]

    async def _periodic_predict(self):
        """Read caches, build sensor context, predict activity, store result."""
        # Read activity summary for sensor data
        activity_entry = await self.hub.get_cache("activity_summary")
        intelligence_entry = await self.hub.get_cache("intelligence")

        # Build context from available cache data
        context = {
            "power_watts": 0,
            "lights_on": 0,
            "motion_rooms": "none",
            "hour": datetime.now().hour,
            "occupancy": "unknown",
            "recent_events": "none",
        }

        if activity_entry and activity_entry.get("data"):
            summary = activity_entry["data"]
            context["power_watts"] = summary.get("power_watts", 0)
            context["lights_on"] = summary.get("lights_on", 0)
            context["motion_rooms"] = summary.get("motion_rooms", "none")
            context["occupancy"] = summary.get("occupancy", "unknown")
            context["recent_events"] = summary.get("recent_events", "none")

        if intelligence_entry and intelligence_entry.get("data"):
            intel = intelligence_entry["data"]
            # Supplement with intelligence data if available
            if "power_watts" in intel and not context["power_watts"]:
                context["power_watts"] = intel.get("power_watts", 0)

        try:
            result = await self.predict_activity(context)

            # Store as current_activity in cache
            cache_entry = await self.hub.get_cache("activity_labels")
            if cache_entry and cache_entry.get("data"):
                data = cache_entry["data"]
            else:
                data = {
                    "current_activity": None,
                    "labels": [],
                    "label_stats": {
                        "total_labels": 0,
                        "total_corrections": 0,
                        "accuracy": 0.0,
                        "activities_seen": [],
                        "classifier_ready": False,
                        "last_trained": None,
                    },
                }

            data["current_activity"] = result
            await self.hub.set_cache("activity_labels", data, {"source": "activity_labeler"})

            self.logger.info(
                f"Activity prediction: {result['predicted']} "
                f"(confidence={result['confidence']}, method={result['method']})"
            )

        except Exception as e:
            self.logger.error(f"Periodic activity prediction failed: {e}")

    @staticmethod
    def _time_of_day() -> str:
        """Return time-of-day category based on current hour.

        Returns:
            One of: night, morning, afternoon, evening.
        """
        hour = datetime.now().hour
        if hour < 6:
            return "night"
        elif hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        else:
            return "evening"
