"""Pattern Recognition Module — EventStore-first behavioral pattern detection.

Uses hierarchical clustering with DTW distance for temporal pattern detection,
association rules for signal correlation, and LLM for semantic interpretation.

Phase 3 rewrite: reads from EventStore (not logbook files), uses EntityGraph
for area resolution, detects patterns per day-type segment, and outputs
enriched fields (entity_chain, trigger_entity, first/last_seen, etc.).
"""

import asyncio
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
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
            id="pattern_detection",
            name="Pattern Detection",
            description="Detects recurring event sequences using hierarchical clustering and association rules.",
            module="pattern_recognition",
            layer="hub",
            config_keys=["patterns.analysis_interval", "patterns.max_areas", "patterns.min_events"],
            test_paths=["tests/hub/test_patterns.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(  # noqa: PLR0913 — matches configurable parameters
        self,
        hub: IntelligenceHub,
        min_pattern_frequency: int = 3,
        min_support: float = 0.7,
        min_confidence: float = 0.8,
        max_areas: int = 20,
        analysis_days: int = 30,
    ):
        super().__init__("pattern_recognition", hub)
        self.min_pattern_frequency = min_pattern_frequency
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_areas = max_areas
        self.analysis_days = analysis_days

    async def initialize(self):
        """Schedule periodic pattern detection via hub timer."""
        self.logger.info("Pattern Recognition module initializing...")

        async def detection_task():
            try:
                patterns = await self.detect_patterns()
                self.logger.info(f"Pattern detection complete: {len(patterns)} patterns found")
            except Exception as e:
                self.logger.error(f"Pattern detection failed: {e}")

        interval = timedelta(seconds=7200)
        try:
            interval_cfg = await self.hub.cache.get_config_value("patterns.analysis_interval", 7200)
            interval = timedelta(seconds=int(interval_cfg))
        except Exception:
            pass

        await self.hub.schedule_task(
            task_id="pattern_detection_periodic",
            coro=detection_task,
            interval=interval,
            run_immediately=True,
        )

    async def detect_patterns(self) -> list[dict[str, Any]]:
        """Detect behavioral patterns from EventStore data."""
        self.logger.info("Starting pattern detection...")

        sequences_by_area = await self._extract_sequences()

        if not sequences_by_area:
            self.logger.warning("No sequences extracted from EventStore")
            await self._store_empty_cache()
            return []

        all_patterns = []
        for area, day_type_sequences in sequences_by_area.items():
            for day_type, sequences in day_type_sequences.items():
                patterns = await self._analyze_segment(area, day_type, sequences)
                all_patterns.extend(patterns)

        for pattern in all_patterns:
            try:
                pattern["llm_description"] = await self._interpret_pattern_llm(pattern)
            except Exception as e:
                self.logger.error(f"LLM failed for {pattern['pattern_id']}: {e}")
                pattern["llm_description"] = "Failed to generate LLM description"

        await self._store_cache(all_patterns, list(sequences_by_area.keys()))
        self.logger.info(f"Pattern detection complete: {len(all_patterns)} patterns stored")
        return all_patterns

    # ── Data Extraction (EventStore) ─────────────────────────────────

    async def _extract_sequences(self) -> dict[str, dict[str, list[dict]]]:
        """Extract sequences from EventStore, grouped by area and day-type.

        Returns:
            {area_id: {day_type: [daily_sequence_dicts]}}
        """
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=self.analysis_days)
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        area_summary = await self.hub.event_store.area_event_summary(start_iso, end_iso)
        if not area_summary:
            return {}

        top_areas = self._select_top_areas(area_summary)
        result: dict[str, dict[str, list[dict]]] = {}

        for area_id in top_areas:
            events = await self.hub.event_store.query_by_area(area_id, start_iso, end_iso)
            events = self._resolve_missing_areas(events, area_id)
            day_type_seqs = self._build_daily_sequences(events)
            if day_type_seqs:
                result[area_id] = day_type_seqs

        return result

    def _select_top_areas(self, area_summary: dict[str, int]) -> list[str]:
        """Select top-N areas by event count."""
        sorted_areas = sorted(area_summary.items(), key=lambda x: x[1], reverse=True)
        return [area for area, _ in sorted_areas[: self.max_areas]]

    def _resolve_missing_areas(self, events: list[dict], expected_area: str) -> list[dict]:
        """Fill in missing area_id using EntityGraph."""
        for event in events:
            if event.get("area_id") is None:
                resolved = self.hub.entity_graph.get_area(event.get("entity_id", ""))
                event["area_id"] = resolved or expected_area
        return events

    def _build_daily_sequences(self, events: list[dict]) -> dict[str, list[dict]]:
        """Group events into daily sequences, segmented by day type.

        Returns:
            {day_type: [sequence_dict_per_day]}
        """
        by_date: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            date_str = event.get("timestamp", "")[:10]
            if date_str:
                by_date[date_str].append(event)

        day_type_seqs: dict[str, list[dict]] = defaultdict(list)
        for date_str, day_events in sorted(by_date.items()):
            day_type = self._classify_day_simple(date_str)
            seq = self._events_to_sequence(day_events, date_str)
            if seq:
                day_type_seqs[day_type].append(seq)

        return dict(day_type_seqs)

    def _classify_day_simple(self, date_str: str) -> str:
        """Simple weekday/weekend classification."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return "weekend" if dt.weekday() >= 5 else "workday"
        except ValueError:
            return "workday"

    def _events_to_sequence(self, events: list[dict], date_str: str) -> dict[str, Any] | None:
        """Convert a day's events into a sequence dict for clustering."""
        events.sort(key=lambda e: e.get("timestamp", ""))

        light_times = self._extract_entity_times(events, "light.", "on")
        motion_times = self._extract_entity_times(events, "binary_sensor.", "on", keyword="motion")
        transactions = self._build_transactions(events)

        if not light_times and not motion_times:
            return None

        timestamps = [e.get("timestamp", "") for e in events]
        entities = list({e.get("entity_id", "") for e in events if e.get("entity_id")})

        return {
            "date": date_str,
            "light_times": light_times,
            "motion_times": motion_times,
            "transactions": transactions,
            "event_count": len(events),
            "events": events,
            "entity_ids": entities,
            "first_timestamp": timestamps[0] if timestamps else "",
            "last_timestamp": timestamps[-1] if timestamps else "",
        }

    def _extract_entity_times(
        self, events: list[dict], domain_prefix: str, state: str, keyword: str | None = None
    ) -> list[int]:
        """Extract time-of-day (minutes since midnight) for matching events."""
        times = []
        for e in events:
            eid = e.get("entity_id", "")
            if not eid.startswith(domain_prefix):
                continue
            if keyword and keyword not in eid.lower():
                continue
            if e.get("new_state") != state:
                continue
            minutes = self._timestamp_to_minutes(e.get("timestamp", ""))
            if minutes is not None:
                times.append(minutes)
        return times

    def _build_transactions(self, events: list[dict]) -> list[str]:
        """Build hourly-bin transaction items for Apriori."""
        transactions = set()
        for e in events:
            eid = e.get("entity_id", "")
            state = e.get("new_state", "")
            minutes = self._timestamp_to_minutes(e.get("timestamp", ""))
            if minutes is None:
                continue
            hour = minutes // 60
            domain = eid.split(".")[0] if "." in eid else "unknown"
            transactions.add(f"{domain}_{state}_h{hour}")
        return list(transactions)

    @staticmethod
    def _timestamp_to_minutes(ts: str) -> int | None:
        """Parse ISO timestamp to minutes since midnight."""
        try:
            t = datetime.fromisoformat(ts.replace("+00:00", "").replace("Z", ""))
            return t.hour * 60 + t.minute
        except (ValueError, AttributeError):
            return None

    # ── Analysis Pipeline ────────────────────────────────────────────

    async def _analyze_segment(self, area: str, day_type: str, sequences: list[dict]) -> list[dict[str, Any]]:
        """Run clustering + association analysis on a (area, day_type) segment."""
        if len(sequences) < self.min_pattern_frequency:
            self.logger.debug(f"Skipping {area}/{day_type}: insufficient sequences")
            return []

        clusters = await self._cluster_sequences(sequences)
        associations = await self._find_associations(sequences)
        return await self._generate_patterns(area, day_type, sequences, clusters, associations)

    async def _cluster_sequences(self, sequences: list[dict[str, Any]]) -> dict[int, list[int]]:
        """Cluster sequences using hierarchical clustering with DTW distance."""
        time_series = [seq["light_times"] for seq in sequences if seq.get("light_times")]
        if len(time_series) < 2:
            return {}

        n = len(time_series)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._dtw_distance(time_series[i], time_series[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        try:
            condensed_dist = squareform(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method="average")
            max_dist = np.percentile(condensed_dist, 50)
            cluster_labels = fcluster(linkage_matrix, max_dist, criterion="distance")

            clusters: dict[int, list[int]] = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[int(label)].append(idx)
            return dict(clusters)
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            return {}

    def _dtw_distance(self, s1: list[int], s2: list[int]) -> float:
        """Compute Dynamic Time Warping distance between two time series."""
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
        """Find association rules using Apriori algorithm."""
        transactions = [seq["transactions"] for seq in sequences if seq.get("transactions")]
        if len(transactions) < self.min_pattern_frequency:
            return []

        try:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)
            if frequent_itemsets.empty:
                return []

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            return [
                {
                    "antecedents": list(row["antecedents"]),
                    "consequents": list(row["consequents"]),
                    "support": float(row["support"]),
                    "confidence": float(row["confidence"]),
                    "lift": float(row["lift"]),
                }
                for _, row in rules.iterrows()
            ]
        except Exception as e:
            self.logger.error(f"Association rule mining failed: {e}")
            return []

    # ── Pattern Generation ───────────────────────────────────────────

    async def _generate_patterns(
        self,
        area: str,
        day_type: str,
        sequences: list[dict[str, Any]],
        clusters: dict[int, list[int]],
        associations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate enriched pattern metadata from clustering and associations."""
        patterns = []
        for cluster_id, seq_indices in clusters.items():
            if len(seq_indices) < self.min_pattern_frequency:
                continue
            pattern = self._build_pattern(area, day_type, cluster_id, seq_indices, sequences, associations)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _build_pattern(  # noqa: PLR0913 — decomposed from _generate_patterns
        self,
        area: str,
        day_type: str,
        cluster_id: int,
        seq_indices: list[int],
        sequences: list[dict],
        associations: list[dict],
    ) -> dict[str, Any] | None:
        """Build a single pattern dict with all required fields."""
        cluster_seqs = [sequences[i] for i in seq_indices]
        all_times = [t for seq in cluster_seqs for t in seq.get("light_times", [])]
        if not all_times:
            return None

        typical_time_minutes = int(np.median(all_times))
        typical_hour = typical_time_minutes // 60
        typical_minute = typical_time_minutes % 60

        entity_chain = self._build_entity_chain(cluster_seqs)
        trigger_entity = entity_chain[0]["entity_id"] if entity_chain else ""
        first_seen, last_seen = self._compute_temporal_bounds(cluster_seqs)
        source_event_count = sum(seq.get("event_count", 0) for seq in cluster_seqs)

        return {
            "pattern_id": f"{area}_{day_type}_cluster_{cluster_id}",
            "name": f"{area.title()} {day_type.title()} Pattern {cluster_id}",
            "area": area,
            "day_type": day_type,
            "typical_time": f"{typical_hour:02d}:{typical_minute:02d}",
            "variance_minutes": int(np.std(all_times)),
            "frequency": len(seq_indices),
            "total_days": len(sequences),
            "confidence": len(seq_indices) / len(sequences),
            "associated_signals": self._extract_associated_signals(area, associations),
            "cluster_size": len(seq_indices),
            "entity_chain": entity_chain,
            "trigger_entity": trigger_entity,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "source_event_count": source_event_count,
        }

    def _build_entity_chain(self, cluster_seqs: list[dict]) -> list[dict[str, Any]]:
        """Build entity chain from the most common event ordering across sequences."""
        entity_order_counts: dict[tuple, int] = defaultdict(int)
        for seq in cluster_seqs:
            events = seq.get("events", [])
            if not events:
                continue
            chain_key = tuple((e.get("entity_id", ""), e.get("new_state", "")) for e in events[:5])
            entity_order_counts[chain_key] += 1

        if not entity_order_counts:
            return []

        most_common = max(entity_order_counts, key=entity_order_counts.get)
        chain = []
        for i, (entity_id, state) in enumerate(most_common):
            chain.append(
                {
                    "entity_id": entity_id,
                    "state": state,
                    "offset_seconds": i * 120,  # approximate 2min spacing
                }
            )
        return chain

    def _compute_temporal_bounds(self, cluster_seqs: list[dict]) -> tuple[str, str]:
        """Get first_seen and last_seen from sequence timestamps."""
        first = ""
        last = ""
        for seq in cluster_seqs:
            ft = seq.get("first_timestamp", "")
            lt = seq.get("last_timestamp", "")
            if ft and (not first or ft < first):
                first = ft
            if lt and (not last or lt > last):
                last = lt
        return first, last

    def _extract_associated_signals(self, area: str, associations: list[dict]) -> list[str]:
        """Extract associated signal names from association rules."""
        signals = []
        for assoc in associations:
            items = assoc.get("antecedents", []) + assoc.get("consequents", [])
            signals.extend(items)
        return list(set(signals))[:5]

    # ── LLM Interpretation ───────────────────────────────────────────

    async def _interpret_pattern_llm(self, pattern: dict[str, Any]) -> str:
        """Use LLM to generate semantic description of pattern."""
        prompt = f"""Analyze this behavioral pattern and provide a short, semantic label (1-3 words max):

Area: {pattern["area"]}
Day Type: {pattern.get("day_type", "unknown")}
Typical Time: {pattern["typical_time"]}
Variance: ±{pattern["variance_minutes"]} minutes
Frequency: {pattern["frequency"]} out of {pattern["total_days"]} days ({pattern["confidence"]:.0%})
Associated Signals: {", ".join(pattern["associated_signals"]) if pattern["associated_signals"] else "None"}

Examples: "Morning routine", "Bedtime", "Evening arrival", "Night light", "Weekend morning"

Label:"""

        try:
            response = await asyncio.to_thread(
                ollama.generate, model="qwen2.5:7b", prompt=prompt, options={"temperature": 0.3, "num_predict": 20}
            )
            text = getattr(response, "response", "").strip()
            text = self._strip_think_tags(text)
            text = text.strip().strip('"').strip("'")
            if len(text) > 50:
                text = text[:47] + "..."
            return text if text else "Unknown pattern"
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return "Unknown pattern"

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> tags from deepseek-r1 output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ── Cache Storage ────────────────────────────────────────────────

    async def _store_cache(self, patterns: list[dict], areas: list[str]) -> None:
        """Store pattern results in hub cache."""
        await self.hub.set_cache(
            "patterns",
            {
                "patterns": patterns,
                "pattern_count": len(patterns),
                "areas_analyzed": areas,
            },
            {
                "source": "pattern_recognition",
                "min_frequency": self.min_pattern_frequency,
                "min_support": self.min_support,
                "min_confidence": self.min_confidence,
            },
        )

    async def _store_empty_cache(self) -> None:
        """Store empty patterns cache."""
        await self._store_cache([], [])

    # ── Event Handler ────────────────────────────────────────────────

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events — pattern detection runs on schedule, not per-event."""
        pass
