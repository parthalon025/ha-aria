"""Co-occurrence detection and adaptive time windows.

Finds entities that frequently change state together within a time window,
regardless of order. Used by the normalizer pipeline to identify scene-like
behavioral clusters before feeding into detection engines.
"""

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

ISO_FMT = "%Y-%m-%dT%H:%M:%S"


@dataclass
class BehavioralCluster:
    """A group of entities that frequently change state together."""

    entities: frozenset[str]
    actions: frozenset[str]  # normalized states observed
    time_window_minutes: float  # typical span of the cluster
    count: int  # number of times this set co-occurred
    typical_ordering: list[str] = field(default_factory=list)  # most common order


@dataclass
class _PairStats:
    """Accumulated statistics for an entity pair during co-occurrence scan."""

    orderings: Counter  # Counter[tuple[str, ...]]
    spans: list[float]  # time spans in minutes

    @classmethod
    def empty(cls) -> "_PairStats":
        return cls(orderings=Counter(), spans=[])


def _parse_ts(ts: str) -> datetime:
    """Parse ISO timestamp, truncating fractional seconds."""
    return datetime.strptime(ts[:19], ISO_FMT)


def _parse_events(events: list[dict]) -> list[tuple[datetime, dict]]:
    """Sort events by timestamp and parse timestamps."""
    sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
    result: list[tuple[datetime, dict]] = []
    for e in sorted_events:
        ts_str = e.get("timestamp", "")
        if not ts_str:
            continue
        try:
            result.append((_parse_ts(ts_str), e))
        except (ValueError, TypeError):
            continue
    return result


def _group_by_date(parsed: list[tuple[datetime, dict]]) -> dict[str, list[tuple[datetime, dict]]]:
    """Group parsed events by date string."""
    groups: dict[str, list[tuple[datetime, dict]]] = {}
    for ts, e in parsed:
        groups.setdefault(ts.strftime("%Y-%m-%d"), []).append((ts, e))
    return groups


def _collect_window_entities(
    day_events: list[tuple[datetime, dict]],
    anchor_idx: int,
    window_seconds: float,
) -> dict[str, datetime]:
    """Collect unique entities within a time window of the anchor event."""
    ts_i, evt_i = day_events[anchor_idx]
    eid_i = evt_i.get("entity_id", "")
    if not eid_i:
        return {}

    seen: dict[str, datetime] = {eid_i: ts_i}
    for j in range(anchor_idx + 1, len(day_events)):
        ts_j, evt_j = day_events[j]
        if (ts_j - ts_i).total_seconds() > window_seconds:
            break
        eid_j = evt_j.get("entity_id", "")
        if eid_j and eid_j != eid_i and eid_j not in seen:
            seen[eid_j] = ts_j
    return seen


def _record_pair_stats(
    seen: dict[str, datetime],
    pair_stats: dict[frozenset[str], _PairStats],
) -> None:
    """Record ordering and span stats for all entity pairs in a window."""
    entities = sorted(seen.keys())
    for a_idx in range(len(entities)):
        for b_idx in range(a_idx + 1, len(entities)):
            pair = frozenset({entities[a_idx], entities[b_idx]})
            stats = pair_stats.setdefault(pair, _PairStats.empty())
            ordering = tuple(sorted([entities[a_idx], entities[b_idx]], key=lambda e: seen[e]))
            stats.orderings[ordering] += 1
            span = abs((seen[entities[b_idx]] - seen[entities[a_idx]]).total_seconds()) / 60
            stats.spans.append(span)


def _count_daily_pairs(
    day_events: list[tuple[datetime, dict]],
    window_seconds: float,
) -> set[frozenset[str]]:
    """Find all entity pairs that co-occur on a given day (one count per day)."""
    pairs: set[frozenset[str]] = set()
    n = len(day_events)
    for i in range(n):
        ts_i, evt_i = day_events[i]
        eid_i = evt_i.get("entity_id", "")
        if not eid_i:
            continue
        for j in range(i + 1, n):
            ts_j, evt_j = day_events[j]
            if (ts_j - ts_i).total_seconds() > window_seconds:
                break
            eid_j = evt_j.get("entity_id", "")
            if eid_j and eid_j != eid_i:
                pairs.add(frozenset({eid_i, eid_j}))
    return pairs


def _collect_entity_states(parsed: list[tuple[datetime, dict]]) -> dict[str, set[str]]:
    """Build entity_id → set of observed states mapping."""
    result: dict[str, set[str]] = {}
    for _, e in parsed:
        eid = e.get("entity_id", "")
        state = e.get("new_state", "")
        if eid and state:
            result.setdefault(eid, set()).add(state)
    return result


def _build_pair_cluster(
    pair: frozenset[str],
    count: int,
    pair_stats: dict[frozenset[str], _PairStats],
    entity_states: dict[str, set[str]],
) -> BehavioralCluster:
    """Build a BehavioralCluster from a qualifying pair."""
    stats = pair_stats.get(pair, _PairStats.empty())
    typical = list(stats.orderings.most_common(1)[0][0]) if stats.orderings else sorted(pair)
    actions: set[str] = set()
    for eid in pair:
        actions.update(entity_states.get(eid, set()))
    avg_span = statistics.mean(stats.spans) if stats.spans else 0.0
    return BehavioralCluster(
        entities=pair,
        actions=frozenset(actions),
        time_window_minutes=avg_span,
        count=count,
        typical_ordering=typical,
    )


def find_co_occurring_sets(
    events: list[dict],
    window_minutes: float = 20,
    min_count: int = 5,
    min_set_size: int = 2,
) -> list[BehavioralCluster]:
    """Find entity sets that frequently change state within a time window.

    Args:
        events: Event dicts with "timestamp", "entity_id", "new_state" fields.
        window_minutes: Max time span (minutes) for entities to be co-occurring.
        min_count: Minimum number of days where the set co-occurred.
        min_set_size: Minimum entities in a cluster (default 2).

    Returns:
        List of BehavioralCluster sorted by count descending.
    """
    if not events:
        return []

    parsed = _parse_events(events)
    if len(parsed) < min_set_size:
        return []

    window_seconds = window_minutes * 60
    date_groups = _group_by_date(parsed)

    pair_counter: Counter[frozenset[str]] = Counter()
    pair_stats: dict[frozenset[str], _PairStats] = {}

    for _date, day_events in date_groups.items():
        # Record pair stats from each anchor window
        for i in range(len(day_events)):
            seen = _collect_window_entities(day_events, i, window_seconds)
            if len(seen) >= min_set_size:
                _record_pair_stats(seen, pair_stats)

        # Count each pair once per day
        for pair in _count_daily_pairs(day_events, window_seconds):
            pair_counter[pair] += 1

    entity_states = _collect_entity_states(parsed)

    # Build clusters from qualifying pairs
    pair_clusters = [
        _build_pair_cluster(pair, count, pair_stats, entity_states)
        for pair, count in pair_counter.most_common()
        if count >= min_count and len(pair) >= min_set_size
    ]

    merged = _merge_pairs_to_sets(pair_clusters, pair_counter, pair_stats, entity_states, min_count)
    return sorted(merged, key=lambda c: c.count, reverse=True)


def _find_connected_components(qualifying_pairs: set[frozenset[str]]) -> list[frozenset[str]]:
    """Find connected components in the pair graph via BFS."""
    entity_to_pairs: dict[str, list[frozenset[str]]] = {}
    for pair in qualifying_pairs:
        for eid in pair:
            entity_to_pairs.setdefault(eid, []).append(pair)

    components: list[frozenset[str]] = []
    visited: set[str] = set()

    for eid in entity_to_pairs:
        if eid in visited:
            continue
        component: set[str] = set()
        queue = [eid]
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            component.add(current)
            for pair in entity_to_pairs.get(current, []):
                for neighbor in pair:
                    if neighbor not in component:
                        queue.append(neighbor)
        visited.update(component)
        components.append(frozenset(component))
    return components


def _all_pairs_qualify(component: frozenset[str], qualifying_pairs: set[frozenset[str]]) -> bool:
    """Check if every possible pair within a component is in qualifying_pairs."""
    members = sorted(component)
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if frozenset({members[i], members[j]}) not in qualifying_pairs:
                return False
    return True


def _build_merged_cluster(
    entity_set: frozenset[str],
    pair_counter: Counter[frozenset[str]],
    pair_stats: dict[frozenset[str], _PairStats],
    entity_states: dict[str, set[str]],
) -> tuple[BehavioralCluster, set[frozenset[str]]]:
    """Build a merged cluster from a fully-connected entity set. Returns cluster + consumed pairs."""
    consumed: set[frozenset[str]] = set()
    all_spans: list[float] = []
    entity_avg_pos: dict[str, list[float]] = {}

    members = sorted(entity_set)
    min_pair_count = min(
        pair_counter.get(frozenset({members[i], members[j]}), 0)
        for i in range(len(members))
        for j in range(i + 1, len(members))
    )

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            pair = frozenset({members[i], members[j]})
            consumed.add(pair)
            stats = pair_stats.get(pair, _PairStats.empty())
            all_spans.extend(stats.spans)
            for ordering, cnt in stats.orderings.items():
                for pos, eid in enumerate(ordering):
                    entity_avg_pos.setdefault(eid, []).append(pos * cnt)

    typical = sorted(entity_set, key=lambda e: statistics.mean(entity_avg_pos.get(e, [0])))
    actions: set[str] = set()
    for eid in entity_set:
        actions.update(entity_states.get(eid, set()))

    cluster = BehavioralCluster(
        entities=entity_set,
        actions=frozenset(actions),
        time_window_minutes=statistics.mean(all_spans) if all_spans else 0.0,
        count=min_pair_count,
        typical_ordering=typical,
    )
    return cluster, consumed


def _merge_pairs_to_sets(
    pair_clusters: list[BehavioralCluster],
    pair_counter: Counter[frozenset[str]],
    pair_stats: dict[frozenset[str], _PairStats],
    entity_states: dict[str, set[str]],
    min_count: int,
) -> list[BehavioralCluster]:
    """Attempt to merge qualifying pairs into larger clusters."""
    if len(pair_clusters) < 2:
        return pair_clusters

    qualifying_pairs = {c.entities for c in pair_clusters}
    components = _find_connected_components(qualifying_pairs)

    result: list[BehavioralCluster] = []
    used_pairs: set[frozenset[str]] = set()

    for component in components:
        if len(component) < 2 or not _all_pairs_qualify(component, qualifying_pairs):
            continue
        cluster, consumed = _build_merged_cluster(component, pair_counter, pair_stats, entity_states)
        if cluster.count >= min_count:
            result.append(cluster)
            used_pairs.update(consumed)

    # Add remaining pairs that weren't merged
    for cluster in pair_clusters:
        if cluster.entities not in used_pairs:
            result.append(cluster)

    return result


def compute_adaptive_window(
    timestamps: list[str],
    max_sigma_minutes: float = 90,
) -> tuple[str, float, bool]:
    """Compute median time-of-day ± 2σ for a set of event timestamps.

    Args:
        timestamps: ISO 8601 timestamps to analyze.
        max_sigma_minutes: If σ exceeds this, skip_time_condition=True.

    Returns:
        (median_time, sigma_minutes, skip_time_condition)
        - median_time: "HH:MM" of the median time-of-day
        - sigma_minutes: standard deviation in minutes
        - skip_time_condition: True if σ > max_sigma_minutes
    """
    if not timestamps:
        return "00:00", 0.0, True

    minutes_list: list[float] = []
    for ts in timestamps:
        try:
            dt = _parse_ts(ts)
            minutes_list.append(dt.hour * 60 + dt.minute + dt.second / 60)
        except (ValueError, TypeError):
            continue

    if not minutes_list:
        return "00:00", 0.0, True

    if len(minutes_list) == 1:
        m = minutes_list[0]
        return f"{int(m // 60):02d}:{int(m % 60):02d}", 0.0, False

    median_m = statistics.median(minutes_list)
    sigma = statistics.stdev(minutes_list)
    return f"{int(median_m // 60) % 24:02d}:{int(median_m % 60):02d}", sigma, sigma > max_sigma_minutes
