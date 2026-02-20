"""ARIA Hub - Core orchestration and module management."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aria.hub.cache import CacheManager
from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore

logger = logging.getLogger(__name__)


class Module:
    """Base class for hub modules."""

    CAPABILITIES: list = []  # Subclasses declare their capabilities

    def __init__(self, module_id: str, hub: "IntelligenceHub"):
        self.module_id = module_id
        self.hub = hub
        self.logger = logging.getLogger(f"module.{module_id}")

    async def initialize(self):
        """Initialize module resources."""
        pass

    async def shutdown(self):
        """Cleanup module resources."""
        pass

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub event.

        Args:
            event_type: Type of event (e.g., "cache_updated", "module_registered")
            data: Event data
        """
        pass

    async def on_config_updated(self, config: dict[str, Any]):
        """Called when a config key is updated via the API.

        Override in subclasses to react to live config changes.
        The default implementation is a no-op so modules that do not
        care about runtime config changes do not need to implement this.

        Args:
            config: Dict with at least ``key`` and ``value`` of the changed
                    parameter (same payload published by PUT /api/config/{key}).
        """
        pass


def _is_entry_expired(line: str, cutoff: datetime) -> bool:
    """Check if a snapshot_log JSONL line has a timestamp before cutoff."""
    try:
        entry = json.loads(line)
        ts = entry.get("timestamp") or entry.get("date") or ""
        if ts:
            parsed = datetime.fromisoformat(ts)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed < cutoff
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Keep malformed lines (don't silently drop data)
    return False


class IntelligenceHub:
    """Central hub for managing modules, cache, and WebSocket events."""

    def __init__(self, cache_path: str):
        """Initialize intelligence hub.

        Args:
            cache_path: Path to SQLite cache database
        """
        self.cache = CacheManager(cache_path)
        self.modules: dict[str, Module] = {}
        self.module_status: dict[str, str] = {}  # module_id -> "running" | "failed"
        self.subscribers: dict[str, set[Callable]] = {}
        self.tasks: set[asyncio.Task] = set()
        self._running = False
        self._start_time: datetime | None = None
        self.hardware_profile = None  # Set during initialize() via scan_hardware()
        self._request_count: int = 0
        self._event_count: int = 0
        self._audit_logger = None
        self._capability_registry = None  # Cached capability registry (created on first access)
        self.logger = logging.getLogger("hub")
        self.entity_graph = EntityGraph()
        # EventStore lives alongside hub.db (same directory)
        events_db_path = str(Path(cache_path).parent / "events.db")
        self.event_store = EventStore(events_db_path)

    async def initialize(self):
        """Initialize hub and cache."""
        self.logger.info("Initializing ARIA Hub...")
        await self.cache.initialize()
        await self.event_store.initialize()
        self._running = True

        # Compute hardware profile once at startup — modules read from hub.hardware_profile
        try:
            from aria.engine.hardware import scan_hardware

            self.hardware_profile = scan_hardware()
            self.logger.info(
                "Hardware profile: %.1fGB RAM, %d cores, GPU=%s",
                self.hardware_profile.ram_gb,
                self.hardware_profile.cpu_cores,
                "yes" if self.hardware_profile.gpu_available else "no",
            )
        except Exception as e:
            self.logger.warning("Failed to scan hardware: %s — modules will scan individually", e)

        # Schedule daily retention pruning
        await self.schedule_task(
            "prune_events",
            self._prune_stale_data,
            interval=timedelta(hours=24),
            run_immediately=True,
        )

        # Schedule daily event store pruning
        await self.schedule_task(
            "event_store_prune",
            self._prune_event_store,
            interval=timedelta(hours=24),
            run_immediately=False,
        )

        # Propagate config changes to all modules that implement on_config_updated
        async def _dispatch_config_updated(data: dict[str, Any]):
            await self.on_config_updated(data)

        self.subscribe("config_updated", _dispatch_config_updated)

        # Refresh entity graph when entities/devices/areas cache is updated
        async def _on_cache_updated_entity_graph(data: dict):
            category = data.get("category", "")
            if category in ("entities", "devices", "areas"):
                await self._refresh_entity_graph()

        self.subscribe("cache_updated", _on_cache_updated_entity_graph)

        self._start_time = datetime.now(tz=UTC)
        self.logger.info("Hub initialized successfully")

    def set_audit_logger(self, audit_logger):
        """Attach audit logger to hub."""
        self._audit_logger = audit_logger

    async def emit_audit(  # noqa: PLR0913
        self,
        event_type: str,
        source: str,
        action: str = "decision",
        subject: str | None = None,
        detail: dict | None = None,
        request_id: str | None = None,
        severity: str = "info",
    ) -> None:
        """Emit a custom audit event. No-op if audit logger not attached."""
        if self._audit_logger:
            await self._audit_logger.log(
                event_type=event_type,
                source=source,
                action=action,
                subject=subject,
                detail=detail,
                request_id=request_id,
                severity=severity,
            )

    async def shutdown(self):
        """Shutdown hub and all modules."""
        self.logger.info("Shutting down ARIA Hub...")
        self._running = False

        # Shutdown all modules
        for module_id, module in self.modules.items():
            self.logger.info(f"Shutting down module: {module_id}")
            try:
                await module.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down module {module_id}: {e}")

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close event store
        try:
            await self.event_store.close()
        except Exception as e:
            self.logger.error(f"Error closing event store: {e}")

        # Close cache
        await self.cache.close()
        self.logger.info("Hub shutdown complete")

    def register_module(self, module: Module):
        """Register a module with the hub.

        Args:
            module: Module instance to register
        """
        if module.module_id in self.modules:
            if self._audit_logger:
                asyncio.create_task(
                    self._audit_logger.log(
                        event_type="module.collision",
                        source="hub",
                        action="attempt",
                        subject=module.module_id,
                        detail={
                            "existing_class": type(self.modules[module.module_id]).__name__,
                            "new_class": type(module).__name__,
                        },
                        severity="warning",
                    )
                )
            raise ValueError(f"Module {module.module_id} already registered")

        self.modules[module.module_id] = module
        self.module_status[module.module_id] = "registered"
        self.logger.info(f"Registered module: {module.module_id}")

        # Log event
        asyncio.create_task(
            self.cache.log_event(event_type="module_registered", metadata={"module_id": module.module_id})
        )

        if self._audit_logger:
            asyncio.create_task(
                self._audit_logger.log(
                    event_type="module.register",
                    source="hub",
                    action="register",
                    subject=module.module_id,
                    detail={"class": type(module).__name__},
                )
            )

    async def on_config_updated(self, config: dict[str, Any]):
        """Propagate a config_updated event to all registered modules.

        Called automatically whenever ``config_updated`` is published on the
        event bus (i.e. after every successful PUT /api/config/{key}).  Modules
        that override ``Module.on_config_updated`` will receive the new config
        payload; modules that do not override it will silently skip (no-op
        default in base class).

        Args:
            config: Dict with ``key`` and ``value`` of the changed parameter.
        """
        key = config.get("key", "<unknown>")
        self.logger.debug("Propagating config_updated (key=%s) to %d module(s)", key, len(self.modules))
        for module_id, module in self.modules.items():
            try:
                await module.on_config_updated(config)
            except Exception as exc:
                self.logger.error(
                    "Error in module %s on_config_updated (key=%s): %s",
                    module_id,
                    key,
                    exc,
                )

    def unregister_module(self, module_id: str) -> bool:
        """Unregister a module from the hub.

        Args:
            module_id: ID of module to unregister

        Returns:
            True if module was unregistered, False if not found
        """
        if module_id not in self.modules:
            return False

        del self.modules[module_id]
        self.logger.info(f"Unregistered module: {module_id}")

        # Log event
        asyncio.create_task(self.cache.log_event(event_type="module_unregistered", metadata={"module_id": module_id}))

        return True

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to hub events.

        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()

        self.subscribers[event_type].add(callback)
        self.logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from hub events.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self.subscribers:
            self.subscribers[event_type].discard(callback)
            self.logger.debug(f"Unsubscribed from event: {event_type}")

    async def publish(self, event_type: str, data: dict[str, Any]):
        """Publish event through both dispatch paths.

        Dispatch path 1 — explicit subscribers: calls every callback registered
        via ``hub.subscribe(event_type, callback)``.  Only callbacks registered
        for the specific ``event_type`` are invoked.

        Dispatch path 2 — module broadcast: calls ``module.on_event(event_type,
        data)`` on *every* registered module regardless of event type.  Modules
        that do not care about an event use the no-op base-class implementation.

        Both paths fire on every ``publish()`` call.  Modules that also register
        an explicit ``subscribe()`` callback will therefore receive the event
        twice — once via the callback and once via ``on_event()``.  Design
        intentionally: subscribe() callbacks are for targeted routing (e.g.
        wiring shadow_engine to state_changed); on_event() broadcasts allow
        any module to observe all events without upfront subscription.

        Backpressure monitoring: logs a WARNING if the total dispatch wall-clock
        time exceeds 100 ms.  This is observability only — there is no queue
        drop or blocking; slow subscribers will delay subsequent publishes.

        Args:
            event_type: Type of event
            data: Event data
        """
        self.logger.debug(f"Publishing event: {event_type}")
        self._event_count += 1

        # Log event
        await self.cache.log_event(event_type=event_type, data=data)

        dispatch_start = time.monotonic()

        # Dispatch path 1: explicit subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                cb_start = time.monotonic()
                try:
                    await callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
                cb_elapsed_ms = (time.monotonic() - cb_start) * 1000
                if cb_elapsed_ms > 100:
                    self.logger.warning(
                        "Slow subscriber callback for event '%s': %.1f ms (threshold 100 ms)",
                        event_type,
                        cb_elapsed_ms,
                    )

        # Dispatch path 2: broadcast to all modules via on_event()
        for module in self.modules.values():
            mod_start = time.monotonic()
            try:
                await module.on_event(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in module {module.module_id} event handler: {e}")
            mod_elapsed_ms = (time.monotonic() - mod_start) * 1000
            if mod_elapsed_ms > 100:
                self.logger.warning(
                    "Slow on_event() in module '%s' for event '%s': %.1f ms (threshold 100 ms)",
                    module.module_id,
                    event_type,
                    mod_elapsed_ms,
                )

        total_elapsed_ms = (time.monotonic() - dispatch_start) * 1000
        if total_elapsed_ms > 100:
            self.logger.warning(
                "Event '%s' total dispatch took %.1f ms (threshold 100 ms) — %d subscriber(s), %d module(s)",
                event_type,
                total_elapsed_ms,
                len(self.subscribers.get(event_type, [])),
                len(self.modules),
            )

    async def schedule_task(
        self, task_id: str, coro: Callable, interval: timedelta | None = None, run_immediately: bool = True
    ):
        """Schedule a task to run periodically.

        Args:
            task_id: Unique task identifier
            coro: Async coroutine to run
            interval: Run interval (None = run once)
            run_immediately: If True, run immediately then schedule
        """

        async def run_task():
            self.logger.info(f"Task {task_id}: starting")

            if run_immediately:
                try:
                    await coro()
                except Exception as e:
                    self.logger.error(f"Task {task_id} error: {e}")

            if interval:
                while self._running:
                    await asyncio.sleep(interval.total_seconds())
                    try:
                        await coro()
                    except Exception as e:
                        self.logger.error(f"Task {task_id} error: {e}")

        task = asyncio.create_task(run_task())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        self.logger.info(f"Scheduled task: {task_id}" + (f" (interval: {interval})" if interval else " (one-time)"))

    async def _prune_event_store(self):
        """Prune old events from EventStore based on retention config."""
        try:
            retention_days = int(await self.cache.get_config_value("events.retention_days", 90))
            cutoff = (datetime.now(tz=UTC) - timedelta(days=retention_days)).isoformat()
            pruned = await self.event_store.prune_before(cutoff)
            if pruned:
                self.logger.info("Pruned %d old events (retention=%d days)", pruned, retention_days)
        except Exception as e:
            self.logger.error("Event store pruning failed: %s", e)

    async def _prune_stale_data(self):
        """Prune old events, resolved predictions, and snapshot log entries."""
        events_deleted = await self.cache.prune_events(retention_days=7)
        preds_deleted = await self.cache.prune_predictions(retention_days=30)
        snapshot_log_pruned = await self._prune_snapshot_log(retention_days=90)
        if events_deleted or preds_deleted or snapshot_log_pruned:
            self.logger.info(
                f"Retention pruning: {events_deleted} events, {preds_deleted} predictions deleted"
                f", {snapshot_log_pruned} snapshot log entries pruned"
            )

    async def _prune_snapshot_log(self, retention_days: int = 90) -> int:
        """Prune snapshot_log.jsonl entries older than retention_days.

        Reads the JSONL file, filters out entries with a timestamp older
        than the cutoff, and rewrites the file with only recent entries.

        Returns:
            Number of entries pruned.
        """
        log_path = Path.home() / "ha-logs" / "intelligence" / "snapshot_log.jsonl"
        if not log_path.exists():
            return 0

        cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)

        def _prune_file() -> int:
            try:
                lines = log_path.read_text().splitlines()
            except OSError as e:
                logger.warning("Failed to read snapshot_log.jsonl: %s", e)
                return 0

            kept = []
            pruned_count = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if _is_entry_expired(line, cutoff):
                    pruned_count += 1
                else:
                    kept.append(line)

            if pruned_count > 0:
                try:
                    log_path.write_text("\n".join(kept) + "\n" if kept else "")
                except OSError as e:
                    logger.warning("Failed to write pruned snapshot_log.jsonl: %s", e)
                    return 0

            return pruned_count

        return await asyncio.to_thread(_prune_file)

    async def _refresh_entity_graph(self):
        """Rebuild entity graph from current cache data."""
        try:
            entities_entry = await self.cache.get("entities")
            devices_entry = await self.cache.get("devices")
            areas_entry = await self.cache.get("areas")

            entities_data = entities_entry.get("data", {}) if entities_entry else {}
            devices_data = devices_entry.get("data", {}) if devices_entry else {}
            areas_data = areas_entry.get("data", []) if areas_entry else []

            self.entity_graph.update(entities_data, devices_data, areas_data)
            self.logger.debug("Entity graph refreshed: %d entities", self.entity_graph.entity_count)
        except Exception as e:
            self.logger.warning("Failed to refresh entity graph: %s", e)

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        """Get data from cache.

        Args:
            category: Cache category

        Returns:
            Cache entry or None if not found
        """
        return await self.cache.get(category)

    async def get_cache_fresh(self, category: str, max_age: timedelta, caller: str = "") -> dict[str, Any] | None:
        """Get cache data with freshness check.

        Returns the cache entry like get_cache(), but logs a warning if
        the data is older than max_age. Always returns the data regardless
        of age — staleness is informational, not blocking.

        Args:
            category: Cache category
            max_age: Maximum acceptable age
            caller: Module name for log attribution

        Returns:
            Cache entry or None if not found
        """
        entry = await self.cache.get(category)
        if entry and entry.get("last_updated"):
            try:
                updated = datetime.fromisoformat(entry["last_updated"])
                age = datetime.now(tz=UTC) - updated.replace(tzinfo=UTC)
                if age > max_age:
                    who = f" ({caller})" if caller else ""
                    self.logger.warning(f"Stale cache: '{category}' is {age} old (max {max_age}){who}")
            except (ValueError, TypeError):
                pass
        return entry

    async def set_cache(self, category: str, data: dict[str, Any], metadata: dict[str, Any] | None = None) -> int:
        """Set data in cache and publish update event.

        Args:
            category: Cache category
            data: Data to store
            metadata: Optional metadata

        Returns:
            New version number
        """
        version = await self.cache.set(category, data, metadata)

        if self._audit_logger:
            await self._audit_logger.log(
                event_type="cache.write",
                source="hub",
                action="set",
                subject=category,
                detail={"version": version, "data_keys": list(data.keys()) if isinstance(data, dict) else None},
            )

        # Publish cache update event
        await self.publish(
            "cache_updated", {"category": category, "version": version, "timestamp": datetime.now(tz=UTC).isoformat()}
        )

        return version

    def get_module(self, module_id: str) -> Module | None:
        """Get registered module by ID.

        Args:
            module_id: Module identifier

        Returns:
            Module instance or None if not found
        """
        return self.modules.get(module_id)

    def get_capability_registry(self):
        """Get cached capability registry (created once, reused across requests)."""
        if self._capability_registry is None:
            from aria.capabilities import CapabilityRegistry

            self._capability_registry = CapabilityRegistry()
            self._capability_registry.collect_from_modules(hub=self)
        return self._capability_registry

    def is_running(self) -> bool:
        """Check if hub is running.

        Returns:
            True if hub is running
        """
        return self._running

    def mark_module_running(self, module_id: str):
        """Mark a module as successfully initialized."""
        self.module_status[module_id] = "running"

    def mark_module_failed(self, module_id: str):
        """Mark a module as failed to initialize."""
        self.module_status[module_id] = "failed"

    def get_uptime_seconds(self) -> float:
        """Get hub uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now(tz=UTC) - self._start_time).total_seconds()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on hub and modules.

        Returns:
            Health check results
        """
        return {
            "status": "ok" if self._running else "stopped",
            "uptime_seconds": round(self.get_uptime_seconds()),
            "modules": {module_id: self.module_status.get(module_id, "unknown") for module_id in self.modules},
            "cache": {"categories": await self.cache.list_categories()},
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
