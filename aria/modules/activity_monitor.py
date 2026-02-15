"""Activity Monitor Module - Adaptive snapshots and activity event logging.

Connects to HA WebSocket, tracks state_changed events, triggers extra
intraday snapshots when the home is occupied and active, and maintains
a rolling 24-hour activity log in 15-minute windows.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp

from aria.hub.core import Module, IntelligenceHub
from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY
from aria.capabilities import Capability


logger = logging.getLogger(__name__)


# Domains worth tracking for activity detection
TRACKED_DOMAINS = {
    "light",
    "switch",
    "binary_sensor",
    "lock",
    "media_player",
    "cover",
    "climate",
    "vacuum",
    "person",
    "device_tracker",
    "fan",
}

# sensor is only tracked when device_class == "power"
CONDITIONAL_DOMAINS = {"sensor"}

# Transitions that are pure noise
NOISE_TRANSITIONS = {
    ("unavailable", "unknown"),
    ("unknown", "unavailable"),
}

# Max snapshots per day and cooldown between triggered snapshots
DAILY_SNAPSHOT_CAP = 20
SNAPSHOT_COOLDOWN_S = 1800  # 30 minutes

# How often to flush buffered events into cache windows
FLUSH_INTERVAL_S = 900  # 15 minutes

# Rolling window retention
MAX_WINDOW_AGE_H = 24


class ActivityMonitor(Module):
    """Tracks HA state changes and triggers adaptive snapshots."""

    CAPABILITIES = [
        Capability(
            id="activity_monitoring",
            name="Activity Monitoring",
            description="WebSocket listener for state_changed events with 15-min windowed activity log and adaptive snapshots.",
            module="activity_monitor",
            layer="hub",
            config_keys=[
                "activity.daily_snapshot_cap",
                "activity.snapshot_cooldown_s",
                "activity.flush_interval_s",
                "activity.max_window_age_h",
            ],
            test_paths=["tests/hub/test_activity_monitor.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub, ha_url: str, ha_token: str):
        super().__init__("activity_monitor", hub)
        self.ha_url = ha_url
        self.ha_token = ha_token

        # In-memory event buffer (flushed every 15 min)
        self._activity_buffer: List[Dict[str, Any]] = []
        # Ring buffer of recent events — survives flushes, for dashboard display
        self._recent_events: deque = deque(maxlen=20)

        # Occupancy state
        self._occupancy_state = False
        self._occupancy_people: List[str] = []
        self._occupancy_since: Optional[str] = None

        # Snapshot control
        self._last_snapshot_time: Optional[datetime] = None
        self._snapshots_today = 0

        # Stats — single date tracker for all daily counters
        self._events_today = 0
        self._events_date = datetime.now().strftime("%Y-%m-%d")
        self._snapshot_date = self._events_date

        # In-memory today snapshot log (avoids full-file scan)
        self._snapshot_log_today_cache: List[Dict[str, Any]] = []

        # WebSocket liveness tracking
        self._ws_connected = False
        self._ws_last_connected_at: Optional[str] = None
        self._ws_disconnect_count = 0
        self._ws_total_disconnect_s = 0.0
        self._ws_last_disconnect_at: Optional[datetime] = None

        # Entity curation state (loaded from SQLite, falls back to domain filter)
        self._included_entities: Set[str] = set()
        self._excluded_entities: Set[str] = set()
        self._curation_loaded: bool = False

        # Path to aria CLI (for subprocess snapshot calls)
        self._aria_cli = self._find_aria_cli()

        # Persistent snapshot log (append-only JSONL, never pruned)
        self._snapshot_log_path = Path.home() / "ha-logs" / "intelligence" / "snapshot_log.jsonl"
        self._snapshot_log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_aria_cli() -> str:
        """Locate the aria CLI executable.

        Prefers the installed `aria` entry point on PATH. Falls back to
        running `python -m aria.cli` using the current interpreter.
        """
        aria_path = shutil.which("aria")
        if aria_path:
            return aria_path
        # Fallback: use current Python interpreter to invoke the module
        return sys.executable

    def _reset_daily_counters(self):
        """Reset daily counters if the date has changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._events_date:
            self._events_today = 0
            self._snapshots_today = 0
            self._snapshot_log_today_cache = []
            self._events_date = today
            self._snapshot_date = today

    async def initialize(self):
        """Start the WebSocket listener and buffer flush timer."""
        self.logger.info("Activity monitor initializing...")

        # Start WebSocket listener
        await self.hub.schedule_task(
            task_id="activity_ws_listener",
            coro=self._ws_listen_loop,
            interval=None,
            run_immediately=True,
        )

        # Start periodic buffer flush
        await self.hub.schedule_task(
            task_id="activity_buffer_flush",
            coro=self._flush_activity_buffer,
            interval=timedelta(seconds=FLUSH_INTERVAL_S),
            run_immediately=False,
        )

        # Load entity curation rules (non-fatal — falls back to domain filter)
        try:
            await self._load_curation_rules()
        except Exception as e:
            self.logger.warning(f"Failed to load curation rules, using domain fallback: {e}")

        self.logger.info("Activity monitor started")

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        if event_type == "curation_updated":
            try:
                await self._load_curation_rules()
            except Exception as e:
                self.logger.warning(f"Failed to reload curation rules: {e}")

    # ------------------------------------------------------------------
    # Entity curation
    # ------------------------------------------------------------------

    async def _load_curation_rules(self):
        """Load entity curation rules from SQLite via the hub's cache manager.

        Populates _included_entities (status in 'included', 'promoted') and
        _excluded_entities (status in 'excluded', 'auto_excluded').

        If the curation table is empty, leaves _curation_loaded False so the
        domain-based fallback continues to work.
        """
        included = await self.hub.cache.get_included_entity_ids()
        all_curation = await self.hub.cache.get_all_curation()

        excluded = {row["entity_id"] for row in all_curation if row["status"] in ("excluded", "auto_excluded")}

        if not included and not excluded:
            self.logger.info("Curation table empty — using domain fallback")
            self._curation_loaded = False
            return

        self._included_entities = included
        self._excluded_entities = excluded
        self._curation_loaded = True
        self.logger.info(f"Loaded curation rules: {len(included)} included, {len(excluded)} excluded")

    # ------------------------------------------------------------------
    # WebSocket listener (follows discovery.py pattern)
    # ------------------------------------------------------------------

    async def _ws_listen_loop(self):
        """Connect to HA WebSocket and listen for state_changed events."""
        ws_url = self.ha_url.replace("http", "ws", 1) + "/api/websocket"
        retry_delay = 5

        while self.hub.is_running():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        # 1. Wait for auth_required
                        msg = await ws.receive_json()
                        if msg.get("type") != "auth_required":
                            self.logger.error(f"Unexpected WS message: {msg}")
                            continue

                        # 2. Authenticate
                        await ws.send_json(
                            {
                                "type": "auth",
                                "access_token": self.ha_token,
                            }
                        )
                        auth_resp = await ws.receive_json()
                        if auth_resp.get("type") != "auth_ok":
                            self.logger.error(f"WS auth failed: {auth_resp}")
                            await asyncio.sleep(retry_delay)
                            continue

                        self.logger.info("Activity WebSocket connected — listening for state_changed")
                        retry_delay = 5  # reset backoff

                        # Track liveness
                        now = datetime.now()
                        if self._ws_last_disconnect_at:
                            gap = (now - self._ws_last_disconnect_at).total_seconds()
                            self._ws_total_disconnect_s += gap
                            self._ws_last_disconnect_at = None
                        self._ws_connected = True
                        self._ws_last_connected_at = now.isoformat()

                        # 2b. Seed occupancy from current person entity states
                        await self._seed_occupancy(session)

                        # 3. Subscribe to state_changed
                        await ws.send_json(
                            {
                                "id": 1,
                                "type": "subscribe_events",
                                "event_type": "state_changed",
                            }
                        )

                        # 4. Listen loop
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get("type") == "event":
                                    event_data = data.get("event", {}).get("data", {})
                                    self._handle_state_changed(event_data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Activity WebSocket error: {e} — retrying in {retry_delay}s")
            except Exception as e:
                self.logger.error(f"Activity WebSocket unexpected error: {e}")

            # Track disconnect
            if self._ws_connected:
                self._ws_connected = False
                self._ws_disconnect_count += 1
                self._ws_last_disconnect_at = datetime.now()

            # Backoff: 5s → 10s → 20s → 60s max
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_state_changed(self, data: Dict[str, Any]):
        """Filter and buffer a single state_changed event."""
        entity_id = data.get("entity_id", "")
        new_state = data.get("new_state") or {}
        old_state = data.get("old_state") or {}

        domain = entity_id.split(".")[0] if "." in entity_id else ""
        from_state = old_state.get("state", "")
        to_state = new_state.get("state", "")

        # Note: _included_entities/_excluded_entities are replaced atomically by
        # _load_curation_rules(). CPython's GIL ensures set reads are safe without
        # locks. Worst case during reload: a few events use stale curation data.
        if self._curation_loaded:
            if entity_id in self._excluded_entities:
                return
            if entity_id not in self._included_entities:
                # Unknown entity — fall back to domain-level filtering
                if domain not in TRACKED_DOMAINS and domain not in CONDITIONAL_DOMAINS:
                    return
                if domain in CONDITIONAL_DOMAINS:
                    device_class = new_state.get("attributes", {}).get("device_class", "")
                    if device_class != "power":
                        return
        else:
            # Curation not loaded — use domain-based filtering (first boot / fallback)
            if domain not in TRACKED_DOMAINS and domain not in CONDITIONAL_DOMAINS:
                return
            if domain in CONDITIONAL_DOMAINS:
                device_class = new_state.get("attributes", {}).get("device_class", "")
                if device_class != "power":
                    return

        from_state = old_state.get("state", "")
        to_state = new_state.get("state", "")

        # Filter noise transitions
        if (from_state, to_state) in NOISE_TRANSITIONS:
            return
        if from_state == to_state:
            return

        # Reset daily counters at midnight
        self._reset_daily_counters()

        self._events_today += 1

        # Build event record
        now = datetime.now()
        attrs = new_state.get("attributes", {})
        friendly_name = attrs.get("friendly_name", entity_id)
        device_class = attrs.get("device_class", "")
        event = {
            "entity_id": entity_id,
            "domain": domain,
            "device_class": device_class,
            "from": from_state,
            "to": to_state,
            "time": now.strftime("%H:%M:%S"),
            "timestamp": now.isoformat(),
            "friendly_name": friendly_name,
        }
        self._activity_buffer.append(event)
        self._recent_events.append(event)

        # Emit to event bus for shadow engine (non-blocking — fire and forget)
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self.hub.publish(
                    "state_changed",
                    {
                        "entity_id": entity_id,
                        "domain": domain,
                        "device_class": device_class,
                        "from": from_state,
                        "to": to_state,
                        "timestamp": event["timestamp"],
                        "friendly_name": friendly_name,
                    },
                )
            )
            task.add_done_callback(
                lambda t: self.logger.error(f"Event publish failed: {t.exception()}") if t.exception() else None
            )
        except RuntimeError as e:
            if "event loop" not in str(e).lower():
                self.logger.warning(f"Unexpected RuntimeError during event publish: {e}")

        # Update occupancy
        if domain in ("person", "device_tracker"):
            self._update_occupancy(entity_id, to_state, friendly_name)

        # Maybe trigger snapshot
        self._maybe_trigger_snapshot()

    def _update_occupancy(self, entity_id: str, state: str, friendly_name: str):
        """Track occupancy from person/device_tracker entities."""
        # person entities: state == "home" means home
        domain = entity_id.split(".")[0]
        if domain == "person":
            name = friendly_name or entity_id.split(".")[-1].replace("_", " ").title()

            if state == "home":
                if name not in self._occupancy_people:
                    self._occupancy_people.append(name)
                if not self._occupancy_state:
                    self._occupancy_state = True
                    self._occupancy_since = datetime.now().isoformat()
                    self.logger.info(f"Occupancy: home ({name})")
            else:
                if name in self._occupancy_people:
                    self._occupancy_people.remove(name)
                if not self._occupancy_people:
                    self._occupancy_state = False
                    self.logger.info("Occupancy: away (all people left)")

    async def _seed_occupancy(self, session: aiohttp.ClientSession):
        """Fetch current person.* entity states from HA REST API to seed occupancy.

        Without this, occupancy stays 'away' after hub restart until someone's
        person entity transitions — which never happens if everyone is already home.
        """
        try:
            url = f"{self.ha_url}/api/states"
            headers = {"Authorization": f"Bearer {self.ha_token}"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    self.logger.warning(f"Failed to seed occupancy (HTTP {resp.status})")
                    return
                states = await resp.json()

            people = []
            for entity in states:
                eid = entity.get("entity_id", "")
                if eid.startswith("person.") and entity.get("state") == "home":
                    name = entity.get("attributes", {}).get("friendly_name") or eid.split(".")[-1].replace("_", " ").title()
                    people.append(name)

            if people:
                self._occupancy_people = people
                self._occupancy_state = True
                self._occupancy_since = datetime.now().isoformat()
                self.logger.info(f"Occupancy seeded: {', '.join(people)} home")
            else:
                self._occupancy_people = []
                self._occupancy_state = False
                self._occupancy_since = None
                self.logger.info("Occupancy seeded: nobody home")

            # Immediately push to cache so dashboard doesn't show stale "away"
            await self._update_summary_cache()
        except Exception as e:
            self.logger.warning(f"Failed to seed occupancy: {e}")

    def _maybe_trigger_snapshot(self):
        """Trigger an extra intraday snapshot if conditions are met."""
        if not self._occupancy_state:
            return

        if self._snapshots_today >= DAILY_SNAPSHOT_CAP:
            return

        now = datetime.now()
        if self._last_snapshot_time:
            elapsed = (now - self._last_snapshot_time).total_seconds()
            if elapsed < SNAPSHOT_COOLDOWN_S:
                return

        # Need meaningful activity — at least 5 events in the buffer
        if len(self._activity_buffer) < 5:
            return

        self._last_snapshot_time = now
        self._snapshots_today += 1

        # Count events by domain in current buffer for context
        domain_counts = defaultdict(int)
        for evt in self._activity_buffer:
            domain_counts[evt["domain"]] += 1

        log_entry = {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "number": self._snapshots_today,
            "buffered_events": len(self._activity_buffer),
            "people": list(self._occupancy_people),
            "domains": dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)),
        }
        self._append_snapshot_log(log_entry)

        self.logger.info(f"Triggering adaptive snapshot ({self._snapshots_today}/{DAILY_SNAPSHOT_CAP} today)")

        # Fire-and-forget subprocess (log errors from the future)
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, self._run_snapshot)
        fut.add_done_callback(self._snapshot_done_callback)

    def _snapshot_done_callback(self, future):
        """Log errors from the snapshot executor future."""
        exc = future.exception()
        if exc:
            self.logger.error(f"Snapshot executor error: {exc}")

    def _run_snapshot(self):
        """Run aria snapshot-intraday in a subprocess."""
        try:
            if self._aria_cli.endswith("aria"):
                cmd = [self._aria_cli, "snapshot-intraday"]
            else:
                # Fallback: invoke via python -m
                cmd = [self._aria_cli, "-m", "aria.cli", "snapshot-intraday"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                self.logger.warning(f"Snapshot subprocess failed: {result.stderr[:200]}")
            else:
                self.logger.info("Adaptive snapshot completed")
        except FileNotFoundError:
            self.logger.warning(f"aria CLI not found at {self._aria_cli}")
        except subprocess.TimeoutExpired:
            self.logger.warning("Snapshot subprocess timed out after 30s")
        except Exception as e:
            self.logger.error(f"Snapshot subprocess error: {e}")

    # ------------------------------------------------------------------
    # Persistent snapshot log (JSONL, append-only)
    # ------------------------------------------------------------------

    def _append_snapshot_log(self, entry: Dict[str, Any]):
        """Append a snapshot record to the persistent JSONL log and in-memory cache."""
        self._snapshot_log_today_cache.append(entry)
        try:
            with open(self._snapshot_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write snapshot log: {e}")

    def _read_snapshot_log_today(self) -> List[Dict[str, Any]]:
        """Return today's snapshot entries from in-memory cache (O(1), no file scan)."""
        return list(self._snapshot_log_today_cache)

    # ------------------------------------------------------------------
    # Event sequence prediction (frequency-based next-event model)
    # ------------------------------------------------------------------

    def _event_sequence_prediction(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict the most likely next event domain based on recent event sequences.

        Uses a simple frequency model: given the last 5 event domains, what
        domain has historically followed that pattern most often?

        Falls back to overall domain frequency if no matching sequence is found.
        """
        # Build a flat list of domain sequences from windowed activity log
        all_domains: List[str] = []
        for w in windows:
            by_domain = w.get("by_domain", {})
            # Expand domain counts into a sequence (order within window is approximate)
            for domain, count in by_domain.items():
                all_domains.extend([domain] * count)

        if len(all_domains) < 6:
            return {}

        # Count what domain follows each 5-domain subsequence
        SEQ_LEN = 5
        sequence_followers: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(len(all_domains) - SEQ_LEN):
            key = "|".join(all_domains[i : i + SEQ_LEN])
            follower = all_domains[i + SEQ_LEN]
            sequence_followers[key][follower] += 1

        # Get the current trailing sequence from the recent events ring buffer
        recent_domains = [evt["domain"] for evt in list(self._recent_events)]
        if len(recent_domains) < SEQ_LEN:
            # Fall back to overall domain frequency
            domain_freq: Dict[str, int] = defaultdict(int)
            for d in all_domains:
                domain_freq[d] += 1
            if not domain_freq:
                return {}
            top = max(domain_freq, key=domain_freq.get)
            total = sum(domain_freq.values())
            return {
                "predicted_next_domain": top,
                "probability": round(domain_freq[top] / total, 2),
                "method": "frequency",
                "sample_size": total,
            }

        current_key = "|".join(recent_domains[-SEQ_LEN:])
        followers = sequence_followers.get(current_key, {})

        if followers:
            top = max(followers, key=followers.get)
            total = sum(followers.values())
            return {
                "predicted_next_domain": top,
                "probability": round(followers[top] / total, 2),
                "method": "sequence",
                "sample_size": total,
            }

        # Fall back to overall frequency
        domain_freq = defaultdict(int)
        for d in all_domains:
            domain_freq[d] += 1
        top = max(domain_freq, key=domain_freq.get)
        total = sum(domain_freq.values())
        return {
            "predicted_next_domain": top,
            "probability": round(domain_freq[top] / total, 2),
            "method": "frequency",
            "sample_size": total,
        }

    # ------------------------------------------------------------------
    # Activity pattern mining (frequent 3-event sequences)
    # ------------------------------------------------------------------

    def _detect_activity_patterns(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find frequent 3-event domain sequences from rolling 24h windows.

        A sequence is "frequent" if it occurs 3+ times in the last 24h.
        Returns the top patterns sorted by count descending.
        """
        # Build flat event domain list with timestamps from notable_changes
        # and domain counts
        domain_sequence: List[str] = []
        for w in windows:
            by_domain = w.get("by_domain", {})
            for domain, count in by_domain.items():
                domain_sequence.extend([domain] * count)

        if len(domain_sequence) < 3:
            return []

        # Count all 3-grams
        trigram_counts: Dict[str, int] = defaultdict(int)
        trigram_last_seen: Dict[str, str] = {}
        for i in range(len(domain_sequence) - 2):
            trigram = (domain_sequence[i], domain_sequence[i + 1], domain_sequence[i + 2])
            key = "|".join(trigram)
            trigram_counts[key] += 1

        # Get last_seen from windows (approximate: use window time)
        domain_offset = 0
        for w in windows:
            w_total = sum(w.get("by_domain", {}).values())
            for i in range(domain_offset, min(domain_offset + w_total - 2, len(domain_sequence) - 2)):
                key = "|".join([domain_sequence[i], domain_sequence[i + 1], domain_sequence[i + 2]])
                trigram_last_seen[key] = w.get("window_start", "")[:16]  # HH:MM
            domain_offset += w_total

        # Filter to frequent (3+) and non-trivial (not all same domain)
        MIN_COUNT = 3
        patterns = []
        for key, count in sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True):
            if count < MIN_COUNT:
                continue
            parts = key.split("|")
            # Skip trivial: all same domain
            if len(set(parts)) == 1:
                continue
            patterns.append(
                {
                    "sequence": parts,
                    "count": count,
                    "last_seen": trigram_last_seen.get(key, ""),
                }
            )
            if len(patterns) >= 10:  # cap at 10 patterns
                break

        return patterns

    # ------------------------------------------------------------------
    # Occupancy arrival prediction (day-of-week historical)
    # ------------------------------------------------------------------

    def _predict_next_arrival(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict when someone will arrive home based on historical occupancy transitions.

        Looks at the last 7 days of windows for transitions from away (occupancy=False)
        to home (occupancy=True), grouped by day of week. Averages the first arrival
        time for the current day of week.
        """
        now = datetime.now()
        current_dow = now.strftime("%A")

        # Only predict if nobody is currently home
        if self._occupancy_state:
            return {
                "status": "home",
                "message": "Someone is already home",
            }

        # Find occupancy transitions (away → home) in windows
        # Group by day of week
        arrivals_by_dow: Dict[str, List[str]] = defaultdict(list)

        for i in range(1, len(windows)):
            prev = windows[i - 1]
            curr = windows[i]
            if not prev.get("occupancy") and curr.get("occupancy"):
                # Transition: away → home
                ws = curr.get("window_start", "")
                if not ws:
                    continue
                try:
                    dt = datetime.fromisoformat(ws)
                    dow = dt.strftime("%A")
                    time_str = dt.strftime("%H:%M")
                    arrivals_by_dow[dow].append(time_str)
                except (ValueError, TypeError):
                    continue

        arrivals_today = arrivals_by_dow.get(current_dow, [])
        if not arrivals_today:
            # Fall back to all-day average
            all_arrivals = []
            for times in arrivals_by_dow.values():
                all_arrivals.extend(times)
            if not all_arrivals:
                return {}
            arrivals_today = all_arrivals

        # Compute average arrival time
        total_minutes = 0
        for t in arrivals_today:
            parts = t.split(":")
            total_minutes += int(parts[0]) * 60 + int(parts[1])
        avg_minutes = total_minutes // len(arrivals_today)
        avg_hour = avg_minutes // 60
        avg_min = avg_minutes % 60
        predicted_time = f"{avg_hour:02d}:{avg_min:02d}"

        # Only predict if the predicted time is in the future
        now_minutes = now.hour * 60 + now.minute
        if avg_minutes <= now_minutes:
            return {
                "status": "past_predicted",
                "message": f"Typical arrival ({predicted_time}) has passed",
                "predicted_arrival": predicted_time,
                "based_on": len(arrivals_today),
            }

        confidence = "low"
        if len(arrivals_today) >= 5:
            confidence = "high"
        elif len(arrivals_today) >= 3:
            confidence = "medium"

        return {
            "status": "predicted",
            "predicted_arrival": predicted_time,
            "confidence": confidence,
            "based_on": len(arrivals_today),
            "day_of_week": current_dow,
        }

    # ------------------------------------------------------------------
    # Activity anomaly detection (event rate vs historical average)
    # ------------------------------------------------------------------

    def _detect_activity_anomalies(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare current hour's event rate to historical average for this hour.

        Flags:
        - "unusual_activity" if current rate > 2x the average
        - "unusual_quiet" if 0 events when average > 10
        """
        now = datetime.now()
        current_hour = now.hour

        # Group windows by hour of day
        hourly_counts: Dict[int, List[int]] = defaultdict(list)
        for w in windows:
            ws = w.get("window_start", "")
            if not ws:
                continue
            try:
                dt = datetime.fromisoformat(ws)
                hourly_counts[dt.hour].append(w.get("event_count", 0))
            except (ValueError, TypeError):
                continue

        anomalies = []

        # Get counts for current hour
        current_hour_counts = hourly_counts.get(current_hour, [])
        if not current_hour_counts:
            return anomalies

        avg = sum(current_hour_counts) / len(current_hour_counts)

        # Current rate: events in buffer right now (within the current window)
        current_rate = len(self._activity_buffer)

        if current_rate > avg * 2 and avg > 0 and current_rate > 5:
            multiplier = round(current_rate / avg, 1)
            anomalies.append(
                {
                    "type": "unusual_activity",
                    "message": f"{multiplier}x normal events this hour ({current_rate} vs avg {round(avg, 1)})",
                    "severity": "info",
                    "hour": current_hour,
                }
            )

        if current_rate == 0 and avg > 10:
            anomalies.append(
                {
                    "type": "unusual_quiet",
                    "message": f"No events this period, but average is {round(avg, 1)} for this hour",
                    "severity": "info",
                    "hour": current_hour,
                }
            )

        return anomalies

    # ------------------------------------------------------------------
    # Buffer flush — 15-minute windows → cache
    # ------------------------------------------------------------------

    async def _flush_activity_buffer(self):
        """Group buffered events into a 15-min window and write to cache."""
        if not self._activity_buffer:
            await self._update_summary_cache()
            return

        now = datetime.now()
        # Window boundaries based on earliest event in buffer (not flush time)
        first_ts = self._activity_buffer[0].get("timestamp", now.isoformat())
        first_dt = datetime.fromisoformat(first_ts) if isinstance(first_ts, str) else now
        minute_slot = (first_dt.minute // 15) * 15
        window_start = first_dt.replace(minute=minute_slot, second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=15) - timedelta(seconds=1)

        # Group events by domain and entity
        by_domain: Dict[str, int] = defaultdict(int)
        by_entity: Dict[str, int] = defaultdict(int)
        notable: List[Dict[str, Any]] = []
        for evt in self._activity_buffer:
            by_domain[evt["domain"]] += 1
            entity_id = evt.get("entity_id", "")
            if entity_id:
                by_entity[entity_id] += 1
            # Notable = non-binary_sensor events, or lock/door events
            domain = evt["domain"]
            if domain in ("lock", "cover", "media_player", "climate", "vacuum"):
                notable.append(
                    {
                        "entity": evt["entity_id"],
                        "from": evt["from"],
                        "to": evt["to"],
                        "time": evt["time"][:5],  # HH:MM
                    }
                )

        window = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "event_count": len(self._activity_buffer),
            "by_domain": dict(by_domain),
            "by_entity": dict(by_entity),
            "notable_changes": notable[-10:],  # cap at 10
            "occupancy": self._occupancy_state,
        }

        # Read existing activity log from cache
        existing = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        windows = []
        if existing and existing.get("data"):
            windows = existing["data"].get("windows", [])

        # Prune windows older than 24 hours
        cutoff = (now - timedelta(hours=MAX_WINDOW_AGE_H)).isoformat()
        windows = [w for w in windows if w.get("window_start", "") >= cutoff]

        windows.append(window)

        activity_log = {
            "windows": windows,
            "last_updated": now.isoformat(),
            "events_today": self._events_today,
            "snapshots_today": self._snapshots_today,
        }

        await self.hub.set_cache(
            CACHE_ACTIVITY_LOG,
            activity_log,
            {
                "source": "activity_monitor",
                "window_count": len(windows),
            },
        )

        self.logger.debug(f"Flushed {len(self._activity_buffer)} events into activity window")

        # Clear buffer
        self._activity_buffer.clear()

        # Update summary cache
        await self._update_summary_cache()

    async def _update_summary_cache(self):
        """Write current activity summary for dashboard consumption."""
        now = datetime.now()

        # Recent activity: last 15 events from ring buffer (survives flushes)
        recent = []
        for evt in reversed(list(self._recent_events)):
            recent.append(
                {
                    "entity": evt["entity_id"],
                    "domain": evt["domain"],
                    "device_class": evt.get("device_class", ""),
                    "from": evt["from"],
                    "to": evt["to"],
                    "time": evt["time"][:5],
                    "friendly_name": evt.get("friendly_name", evt["entity_id"]),
                }
            )
            if len(recent) >= 15:
                break

        # Activity rate from cached windows
        activity_log = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        windows = []
        if activity_log and activity_log.get("data"):
            windows = activity_log["data"].get("windows", [])

        current_count = len(self._activity_buffer)

        # Average events/window over last hour (up to 4 windows)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        recent_windows = [w for w in windows if w.get("window_start", "") >= one_hour_ago]
        if recent_windows:
            avg_1h = sum(w["event_count"] for w in recent_windows) / len(recent_windows)
        else:
            avg_1h = 0

        trend = "stable"
        if current_count > avg_1h * 1.5 and avg_1h > 0:
            trend = "increasing"
        elif current_count < avg_1h * 0.5 and avg_1h > 0:
            trend = "decreasing"

        # Domain counts in last hour
        domains_1h: Dict[str, int] = defaultdict(int)
        for w in recent_windows:
            for domain, count in w.get("by_domain", {}).items():
                domains_1h[domain] += count
        # Also count current buffer
        for evt in self._activity_buffer:
            if evt.get("timestamp", "") >= one_hour_ago:
                domains_1h[evt["domain"]] += 1

        # Cooldown remaining
        cooldown_remaining = 0
        if self._last_snapshot_time:
            elapsed = (now - self._last_snapshot_time).total_seconds()
            cooldown_remaining = max(0, SNAPSHOT_COOLDOWN_S - elapsed)

        # Compute intelligence enhancements from windowed history
        event_predictions = self._event_sequence_prediction(windows)
        patterns = self._detect_activity_patterns(windows)
        occupancy_prediction = self._predict_next_arrival(windows)
        anomalies = self._detect_activity_anomalies(windows)

        summary = {
            "occupancy": {
                "anyone_home": self._occupancy_state,
                "people": list(self._occupancy_people),
                "since": self._occupancy_since,
            },
            "recent_activity": recent,
            "activity_rate": {
                "current": current_count,
                "avg_1h": round(avg_1h, 1),
                "trend": trend,
            },
            "snapshot_status": {
                "last_triggered": self._last_snapshot_time.isoformat() if self._last_snapshot_time else None,
                "today_count": self._snapshots_today,
                "daily_cap": DAILY_SNAPSHOT_CAP,
                "cooldown_remaining_s": int(cooldown_remaining),
                "log_today": self._read_snapshot_log_today(),
            },
            "domains_active_1h": dict(sorted(domains_1h.items(), key=lambda x: x[1], reverse=True)),
            "websocket": {
                "connected": self._ws_connected,
                "last_connected_at": self._ws_last_connected_at,
                "disconnect_count": self._ws_disconnect_count,
                "total_disconnect_s": round(self._ws_total_disconnect_s, 1),
            },
            "event_predictions": event_predictions,
            "patterns": patterns,
            "occupancy_prediction": occupancy_prediction,
            "anomalies": anomalies,
        }

        await self.hub.set_cache(
            CACHE_ACTIVITY_SUMMARY,
            summary,
            {
                "source": "activity_monitor",
            },
        )
