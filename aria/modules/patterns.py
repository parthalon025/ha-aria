"""Pattern Recognition Module - Detect behavioral patterns in time-series data.

Uses hierarchical clustering with DTW distance for temporal pattern detection,
association rules for signal correlation, and LLM for semantic interpretation.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from aria.capabilities import Capability
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)


class PatternRecognition(Module):
    """Detects behavioral patterns using clustering and association rules."""

    CAPABILITIES = [
        Capability(
            id="pattern_recognition",
            name="Pattern Recognition",
            description="Detects recurring event sequences using hierarchical clustering and association rules.",
            module="pattern_recognition",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_patterns.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(
        self,
        hub: IntelligenceHub,
        log_dir: Path,
        min_pattern_frequency: int = 3,
        min_support: float = 0.7,
        min_confidence: float = 0.8,
    ):
        """Initialize pattern recognition module.

        Args:
            hub: IntelligenceHub instance
            log_dir: Directory containing HA logbook files (~/ha-logs/)
            min_pattern_frequency: Minimum occurrences to consider a pattern
            min_support: Minimum support for association rules (0-1)
            min_confidence: Minimum confidence for association rules (0-1)
        """
        super().__init__("pattern_recognition", hub)
        self.log_dir = Path(log_dir)
        self.min_pattern_frequency = min_pattern_frequency
        self.min_support = min_support
        self.min_confidence = min_confidence

    async def initialize(self):
        """Initialize module - run initial pattern detection."""
        self.logger.info("Pattern Recognition module initializing...")

        try:
            patterns = await self.detect_patterns()
            self.logger.info(f"Initial pattern detection complete: {len(patterns)} patterns found")
        except Exception as e:
            self.logger.error(f"Initial pattern detection failed: {e}")

    async def detect_patterns(self) -> list[dict[str, Any]]:
        """Detect behavioral patterns in historical data.

        Returns:
            List of detected patterns with metadata
        """
        self.logger.info("Starting pattern detection...")

        # 1. Load and parse logbook data
        sequences_by_area = await self._extract_sequences()

        if not sequences_by_area:
            self.logger.warning("No sequences extracted from logbook data")
            return []

        # 2. Detect patterns per area
        all_patterns = []
        for area, sequences in sequences_by_area.items():
            self.logger.info(f"Processing {len(sequences)} sequences for area: {area}")

            if len(sequences) < self.min_pattern_frequency:
                self.logger.debug(f"Skipping {area}: insufficient sequences")
                continue

            # Cluster sequences
            clusters = await self._cluster_sequences(sequences)

            # Find association rules
            associations = await self._find_associations(sequences)

            # Generate patterns
            patterns = await self._generate_patterns(area, sequences, clusters, associations)

            all_patterns.extend(patterns)

        # 3. Interpret patterns with LLM
        for pattern in all_patterns:
            try:
                llm_desc = await self._interpret_pattern_llm(pattern)
                pattern["llm_description"] = llm_desc
            except Exception as e:
                self.logger.error(f"LLM interpretation failed for pattern {pattern['pattern_id']}: {e}")
                pattern["llm_description"] = "Failed to generate LLM description"

        # 4. Store in hub cache
        await self.hub.set_cache(
            "patterns",
            {
                "patterns": all_patterns,
                "pattern_count": len(all_patterns),
                "areas_analyzed": list(sequences_by_area.keys()),
            },
            {
                "source": "pattern_recognition",
                "min_frequency": self.min_pattern_frequency,
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
            },
        )

        self.logger.info(f"Pattern detection complete: {len(all_patterns)} patterns stored")
        return all_patterns

    async def _extract_sequences(self) -> dict[str, list[dict[str, Any]]]:
        """Extract temporal sequences from logbook data grouped by area.

        Uses both daily logbook files and intraday snapshots to build sequences.

        Returns:
            Dict mapping area name to list of sequences
        """
        sequences_by_area = defaultdict(list)

        # Strategy 1: Use intraday snapshots (better for limited data)
        intraday_dir = self.log_dir / "intelligence" / "intraday"
        if intraday_dir.exists():
            for date_dir in sorted(intraday_dir.glob("2026-*")):
                snapshot_files = sorted(date_dir.glob("*.json"))
                self.logger.info(f"Found {len(snapshot_files)} intraday snapshots for {date_dir.name}")

                for snapshot_file in snapshot_files:
                    try:
                        with open(snapshot_file) as f:
                            snapshot = json.load(f)

                        # Extract hour
                        hour = snapshot.get("hour", 0)

                        # Parse snapshot into sequences
                        snapshot_sequences = self._parse_snapshot_to_sequences(snapshot, date_dir.name, hour)

                        # Merge into sequences_by_area
                        for area, seq in snapshot_sequences.items():
                            sequences_by_area[area].append(seq)

                    except Exception as e:
                        self.logger.error(f"Failed to process {snapshot_file}: {e}")
                        continue

        # Strategy 2: Fallback to logbook files if no intraday data
        if not sequences_by_area:
            logbook_files = sorted(self.log_dir.glob("2026-*.json"))
            self.logger.info(f"Found {len(logbook_files)} logbook files")

            for logbook_file in logbook_files:
                try:
                    with open(logbook_file) as f:
                        events = json.load(f)

                    # Extract date from filename
                    date_str = logbook_file.stem  # e.g., "2026-02-11"
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # Group events by entity and area
                    daily_sequences = self._parse_events_to_sequences(events, date_obj)

                    # Merge into sequences_by_area
                    for area, seq in daily_sequences.items():
                        sequences_by_area[area].append(seq)

                except Exception as e:
                    self.logger.error(f"Failed to process {logbook_file}: {e}")
                    continue

        return dict(sequences_by_area)

    def _parse_snapshot_to_sequences(
        self, snapshot: dict[str, Any], date_str: str, hour: int
    ) -> dict[str, dict[str, Any]]:
        """Parse intraday snapshot into sequences per area.

        Args:
            snapshot: Intraday snapshot data
            date_str: Date string (YYYY-MM-DD)
            hour: Hour of day (0-23)

        Returns:
            Dict mapping area name to sequence data
        """
        sequences = {}

        # Extract motion sensor data
        motion_data = snapshot.get("motion", {})
        motion_active = motion_data.get("active_count", 0)

        # Extract lights data
        lights_data = snapshot.get("lights", {})
        lights_on = lights_data.get("on", 0)
        total_brightness = lights_data.get("total_brightness", 0)

        # Extract occupancy
        occupancy = snapshot.get("occupancy", {})
        people_home_list = occupancy.get("people_home", [])
        people_home_count = len(people_home_list) if isinstance(people_home_list, list) else 0

        # Create a pseudo-sequence for this time block
        # Use hour as proxy for time-series
        time_minutes = hour * 60

        # For now, create a general sequence
        # In future, could parse entities dict for per-area data
        area = "general"
        sequences[area] = {
            "date": date_str,
            "hour": hour,
            "light_times": [time_minutes] if lights_on > 0 else [],
            "motion_times": [time_minutes] if motion_active > 0 else [],
            "transactions": [
                f"{area}_light_on_h{hour}" if lights_on > 0 else f"{area}_light_off_h{hour}",
                f"{area}_motion_on_h{hour}" if motion_active > 0 else f"{area}_motion_off_h{hour}",
                f"{area}_occupied_h{hour}" if people_home_count > 0 else f"{area}_unoccupied_h{hour}",
            ],
            "event_count": lights_on + motion_active,
            "lights_on": lights_on,
            "motion_active": motion_active,
            "people_home_count": people_home_count,
            "brightness": total_brightness,
        }

        return sequences

    def _parse_events_to_sequences(
        self, events: list[dict[str, Any]], date: datetime.date
    ) -> dict[str, dict[str, Any]]:
        """Parse logbook events into daily sequences per area.

        Args:
            events: List of logbook events
            date: Date of the events

        Returns:
            Dict mapping area name to sequence data
        """
        # Group events by area (extracted from entity name)
        area_events = defaultdict(list)

        for event in events:
            entity_id = event.get("entity_id", "")
            name = event.get("name", "")
            state = event.get("state", "")
            when_str = event.get("when", "")

            # Skip non-actionable entities
            if not entity_id or entity_id.startswith("sensor."):
                continue

            # Parse timestamp
            try:
                when = datetime.fromisoformat(when_str.replace("+00:00", ""))
            except Exception:
                continue

            # Extract area from name (simple heuristic)
            area = self._extract_area_from_name(name, entity_id)

            area_events[area].append(
                {"entity_id": entity_id, "name": name, "state": state, "time": when.time(), "timestamp": when}
            )

        # Convert to sequence format
        sequences = {}
        for area, events_list in area_events.items():
            # Sort by time
            events_list.sort(key=lambda e: e["timestamp"])

            # Extract time series for lights and motion
            light_times = [
                e["time"].hour * 60 + e["time"].minute
                for e in events_list
                if e["entity_id"].startswith("light.") and e["state"] == "on"
            ]

            motion_times = [
                e["time"].hour * 60 + e["time"].minute
                for e in events_list
                if "motion" in e["entity_id"].lower() and e["state"] == "on"
            ]

            # Build transaction for association rules (hourly bins)
            transactions = set()
            for event in events_list:
                hour = event["time"].hour
                if event["entity_id"].startswith("light."):
                    transactions.add(f"{area}_light_{event['state']}_h{hour}")
                elif "motion" in event["entity_id"].lower():
                    transactions.add(f"{area}_motion_{event['state']}_h{hour}")

            sequences[area] = {
                "date": date.isoformat(),
                "light_times": light_times,
                "motion_times": motion_times,
                "transactions": list(transactions),
                "event_count": len(events_list),
            }

        return sequences

    def _extract_area_from_name(self, name: str, entity_id: str) -> str:
        """Extract area name from entity name or ID.

        Args:
            name: Human-readable entity name
            entity_id: Entity ID

        Returns:
            Area name (lowercase)
        """
        # Common area keywords
        areas = [
            "bedroom",
            "kitchen",
            "living",
            "bathroom",
            "closet",
            "office",
            "garage",
            "hallway",
            "dining",
            "basement",
        ]

        name_lower = name.lower()
        entity_lower = entity_id.lower()

        for area in areas:
            if area in name_lower or area in entity_lower:
                return area

        return "general"

    async def _cluster_sequences(self, sequences: list[dict[str, Any]]) -> dict[int, list[int]]:
        """Cluster sequences using hierarchical clustering with DTW distance.

        Args:
            sequences: List of sequences with time series data

        Returns:
            Dict mapping cluster ID to list of sequence indices
        """
        # Extract light time series
        time_series = [seq["light_times"] for seq in sequences if seq["light_times"]]

        if len(time_series) < 2:
            return {}

        # Compute DTW distance matrix
        n = len(time_series)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._dtw_distance(time_series[i], time_series[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Hierarchical clustering
        try:
            condensed_dist = squareform(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method="average")

            # Cut tree to get clusters (using distance threshold)
            max_dist = np.percentile(condensed_dist, 50)  # Median distance
            cluster_labels = fcluster(linkage_matrix, max_dist, criterion="distance")

            # Group by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[int(label)].append(idx)

            return dict(clusters)

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {}

    def _dtw_distance(self, s1: list[int], s2: list[int]) -> float:
        """Compute Dynamic Time Warping distance between two time series.

        Args:
            s1: First time series (minutes since midnight)
            s2: Second time series

        Returns:
            DTW distance
        """
        if not s1 or not s2:
            return float("inf")

        n, m = len(s1), len(s2)
        dtw = np.full((n + 1, m + 1), float("inf"))
        dtw[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        return dtw[n, m]

    async def _find_associations(self, sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find association rules using Apriori algorithm.

        Args:
            sequences: List of sequences with transaction data

        Returns:
            List of association rules
        """
        # Collect all transactions
        transactions = [seq["transactions"] for seq in sequences if seq["transactions"]]

        if len(transactions) < self.min_pattern_frequency:
            return []

        try:
            # Encode transactions
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            # Find frequent itemsets
            frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)

            if frequent_itemsets.empty:
                return []

            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)

            # Convert to dict format
            associations = []
            for _, rule in rules.iterrows():
                associations.append(
                    {
                        "antecedents": list(rule["antecedents"]),
                        "consequents": list(rule["consequents"]),
                        "support": float(rule["support"]),
                        "confidence": float(rule["confidence"]),
                        "lift": float(rule["lift"]),
                    }
                )

            return associations

        except Exception as e:
            self.logger.error(f"Association rule mining failed: {e}")
            return []

    async def _generate_patterns(
        self,
        area: str,
        sequences: list[dict[str, Any]],
        clusters: dict[int, list[int]],
        associations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate pattern metadata from clustering and associations.

        Args:
            area: Area name
            sequences: List of sequences
            clusters: Cluster assignments
            associations: Association rules

        Returns:
            List of pattern dictionaries
        """
        patterns = []

        # Generate patterns from clusters
        for cluster_id, seq_indices in clusters.items():
            if len(seq_indices) < self.min_pattern_frequency:
                continue

            # Compute cluster statistics
            cluster_seqs = [sequences[i] for i in seq_indices]

            # Average time for light events
            all_times = [t for seq in cluster_seqs for t in seq["light_times"]]
            if not all_times:
                continue

            typical_time_minutes = int(np.median(all_times))
            variance_minutes = int(np.std(all_times))
            typical_hour = typical_time_minutes // 60
            typical_minute = typical_time_minutes % 60

            # Find associated signals from association rules
            area_prefix = f"{area}_"
            associated_signals = []
            for assoc in associations:
                items = assoc["antecedents"] + assoc["consequents"]
                area_items = [item for item in items if item.startswith(area_prefix)]
                if area_items:
                    associated_signals.extend(area_items)

            associated_signals = list(set(associated_signals))[:5]  # Top 5

            pattern = {
                "pattern_id": f"{area}_cluster_{cluster_id}",
                "name": f"{area.title()} Pattern {cluster_id}",
                "area": area,
                "typical_time": f"{typical_hour:02d}:{typical_minute:02d}",
                "variance_minutes": variance_minutes,
                "frequency": len(seq_indices),
                "total_days": len(sequences),
                "confidence": len(seq_indices) / len(sequences),
                "associated_signals": associated_signals,
                "cluster_size": len(seq_indices),
            }

            patterns.append(pattern)

        return patterns

    async def _interpret_pattern_llm(self, pattern: dict[str, Any]) -> str:
        """Use LLM to generate semantic description of pattern.

        Args:
            pattern: Pattern dictionary

        Returns:
            LLM-generated description
        """
        prompt = f"""Analyze this behavioral pattern and provide a short, semantic label (1-3 words max):

Area: {pattern["area"]}
Typical Time: {pattern["typical_time"]}
Variance: ±{pattern["variance_minutes"]} minutes
Frequency: {pattern["frequency"]} out of {pattern["total_days"]} days ({pattern["confidence"]:.0%})
Associated Signals: {", ".join(pattern["associated_signals"]) if pattern["associated_signals"] else "None"}

Examples: "Morning routine", "Bedtime", "Evening arrival", "Night light", "Weekend morning"

Label:"""

        try:
            # Call Ollama (use qwen2.5:7b for faster, simpler responses)
            response = await asyncio.to_thread(
                ollama.generate, model="qwen2.5:7b", prompt=prompt, options={"temperature": 0.3, "num_predict": 20}
            )

            # Extract response - ollama.generate returns a GenerateResponse object
            # Access response attribute directly
            text = getattr(response, "response", "").strip()
            text = self._strip_think_tags(text)

            # Clean up response
            text = text.strip().strip('"').strip("'")

            # Truncate if too long
            if len(text) > 50:
                text = text[:47] + "..."

            return text if text else "Unknown pattern"

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "Unknown pattern"

    def _strip_think_tags(self, text: str) -> str:
        """Strip <think>...</think> tags from deepseek-r1 output.

        Args:
            text: Raw LLM response

        Returns:
            Cleaned text
        """
        import re

        # Remove <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events.

        Args:
            event_type: Type of event
            data: Event data
        """
        # Future: Respond to new logbook data by updating patterns
        # For now, patterns are detected on-demand or via scheduled task
        pass
