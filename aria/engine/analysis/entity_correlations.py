"""Entity co-occurrence and conditional probability analysis.

Goes beyond Pearson r (which compares metric time series) to discover
entity-level behavioral patterns: "when X happens, Y usually follows."
Operates on logbook events and intraday snapshots.
"""

from collections import defaultdict
from datetime import datetime

# Domains worth tracking for co-occurrence (skip noisy sensors)
TRACKABLE_DOMAINS = {
    "light",
    "switch",
    "lock",
    "cover",
    "media_player",
    "climate",
    "automation",
    "binary_sensor",
    "input_boolean",
    "vacuum",
    "fan",
}

# Exclude noisy entities from correlation tracking
EXCLUDED_PATTERNS = {
    "sensor.time",
    "sensor.date",
    "binary_sensor.updater",
}


def _is_trackable(entity_id: str) -> bool:
    """Check if an entity is worth tracking for co-occurrence."""
    domain = entity_id.split(".")[0] if "." in entity_id else ""
    if domain not in TRACKABLE_DOMAINS:
        return False
    return all(not entity_id.startswith(pattern) for pattern in EXCLUDED_PATTERNS)


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse a logbook timestamp string into datetime."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _parse_trackable_events(logbook_entries):
    """Filter logbook entries to trackable entities with parsed timestamps."""
    events = []
    for entry in logbook_entries:
        eid = entry.get("entity_id", "")
        if not _is_trackable(eid):
            continue
        ts = _parse_timestamp(entry.get("when", ""))
        if ts is None:
            continue
        events.append({"entity_id": eid, "ts": ts, "hour": ts.hour})
    return events


def _count_co_occurrences(events, window_seconds):
    """Count co-occurring entity pairs within a sliding time window."""
    pair_counts = defaultdict(int)
    pair_hours = defaultdict(list)

    for i, event_a in enumerate(events):
        for j in range(i + 1, len(events)):
            event_b = events[j]
            delta = (event_b["ts"] - event_a["ts"]).total_seconds()
            if delta > window_seconds:
                break
            if event_a["entity_id"] == event_b["entity_id"]:
                continue

            # Canonical pair ordering (alphabetical) for consistent counting
            pair = tuple(sorted([event_a["entity_id"], event_b["entity_id"]]))
            pair_counts[pair] += 1
            pair_hours[pair].append(event_a["hour"])

    return pair_counts, pair_hours


def compute_co_occurrences(logbook_entries: list, window_minutes: int = 15) -> list:
    """Find entity pairs that frequently change state within a time window.

    Args:
        logbook_entries: List of logbook events with entity_id and when fields.
        window_minutes: Time window for co-occurrence (default 15 min).

    Returns:
        List of co-occurrence dicts sorted by count, each with:
        - entity_a, entity_b: entity IDs
        - count: number of co-occurrences
        - conditional_prob_a_given_b: P(A changes | B changed within window)
        - conditional_prob_b_given_a: P(B changes | A changed within window)
        - typical_hour: most common hour for co-occurrence
    """
    events = _parse_trackable_events(logbook_entries)

    if len(events) < 10:
        return []

    events.sort(key=lambda e: e["ts"])

    # Count per-entity events (for conditional probability denominator)
    entity_counts = defaultdict(int)
    for e in events:
        entity_counts[e["entity_id"]] += 1

    pair_counts, pair_hours = _count_co_occurrences(events, window_minutes * 60)

    if not pair_counts:
        return []

    # Build results with conditional probabilities
    results = _build_co_occurrence_results(pair_counts, entity_counts, pair_hours)
    results.sort(key=lambda r: -r["count"])
    return results[:50]  # Top 50 pairs


def _build_co_occurrence_results(pair_counts, entity_counts, pair_hours):
    """Build co-occurrence result dicts with conditional probabilities."""
    results = []
    for (entity_a, entity_b), count in pair_counts.items():
        if count < 3:  # Minimum co-occurrences to be meaningful
            continue

        # P(A|B) = co-occurrences / B_count, P(B|A) = co-occurrences / A_count
        # Capped at 1.0 — sliding window can produce inflated counts
        prob_a_given_b = round(min(1.0, count / entity_counts[entity_b]), 3) if entity_counts[entity_b] > 0 else 0
        prob_b_given_a = round(min(1.0, count / entity_counts[entity_a]), 3) if entity_counts[entity_a] > 0 else 0

        # Most common hour
        hours = pair_hours[(entity_a, entity_b)]
        typical_hour = max(set(hours), key=hours.count) if hours else None

        results.append(
            {
                "entity_a": entity_a,
                "entity_b": entity_b,
                "count": count,
                "conditional_prob_a_given_b": prob_a_given_b,
                "conditional_prob_b_given_a": prob_b_given_a,
                "typical_hour": typical_hour,
                "strength": _classify_strength(max(prob_a_given_b, prob_b_given_a)),
            }
        )
    return results


def _classify_strength(prob: float) -> str:
    """Classify co-occurrence strength from conditional probability."""
    if prob >= 0.8:
        return "very_strong"
    if prob >= 0.5:
        return "strong"
    if prob >= 0.3:
        return "moderate"
    return "weak"


def compute_hourly_patterns(logbook_entries: list) -> dict:
    """Discover per-entity hourly activity patterns.

    Returns dict mapping entity_id to hourly activity distribution,
    useful for detecting "unusual time" anomalies.
    """
    hourly = defaultdict(lambda: defaultdict(int))
    for entry in logbook_entries:
        eid = entry.get("entity_id", "")
        if not _is_trackable(eid):
            continue
        ts = _parse_timestamp(entry.get("when", ""))
        if ts is None:
            continue
        hourly[eid][ts.hour] += 1

    # Convert to patterns with peak hours
    patterns = {}
    for eid, hours in hourly.items():
        total = sum(hours.values())
        if total < 5:
            continue
        distribution = {h: round(c / total, 3) for h, c in sorted(hours.items())}
        peak_hour = max(hours, key=hours.get)
        patterns[eid] = {
            "total_events": total,
            "peak_hour": peak_hour,
            "distribution": distribution,
        }

    return patterns


def summarize_entity_correlations(co_occurrences: list, hourly_patterns: dict, top_n: int = 10) -> dict:
    """Produce a summary suitable for LLM context and storage.

    Returns dict with top patterns, key entities, and automation-worthy pairs.
    """
    # Top co-occurrence pairs
    top_pairs = co_occurrences[:top_n]

    # Automation-worthy: strong or very_strong with high conditional probability
    automation_worthy = [p for p in co_occurrences if p["strength"] in ("strong", "very_strong") and p["count"] >= 5][
        :top_n
    ]

    # Most active entities
    entity_activity = {}
    for pair in co_occurrences:
        for eid in (pair["entity_a"], pair["entity_b"]):
            entity_activity[eid] = entity_activity.get(eid, 0) + pair["count"]
    most_active = sorted(entity_activity.items(), key=lambda x: -x[1])[:10]

    return {
        "top_co_occurrences": top_pairs,
        "automation_worthy_pairs": automation_worthy,
        "most_correlated_entities": [{"entity_id": e, "co_occurrence_events": c} for e, c in most_active],
        "entities_with_patterns": len(hourly_patterns),
        "total_pairs_found": len(co_occurrences),
    }
