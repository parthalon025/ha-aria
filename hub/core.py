"""Intelligence Hub - Core orchestration and module management."""

import asyncio
import logging
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime, timedelta
import json

from hub.cache import CacheManager


logger = logging.getLogger(__name__)


class Module:
    """Base class for hub modules."""

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

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        """Handle hub event.

        Args:
            event_type: Type of event (e.g., "cache_updated", "module_registered")
            data: Event data
        """
        pass


class IntelligenceHub:
    """Central hub for managing modules, cache, and WebSocket events."""

    def __init__(self, cache_path: str):
        """Initialize intelligence hub.

        Args:
            cache_path: Path to SQLite cache database
        """
        self.cache = CacheManager(cache_path)
        self.modules: Dict[str, Module] = {}
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.tasks: Set[asyncio.Task] = set()
        self._running = False
        self.logger = logging.getLogger("hub")

    async def initialize(self):
        """Initialize hub and cache."""
        self.logger.info("Initializing Intelligence Hub...")
        await self.cache.initialize()
        self._running = True
        self.logger.info("Hub initialized successfully")

    async def shutdown(self):
        """Shutdown hub and all modules."""
        self.logger.info("Shutting down Intelligence Hub...")
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

        # Close cache
        await self.cache.close()
        self.logger.info("Hub shutdown complete")

    def register_module(self, module: Module):
        """Register a module with the hub.

        Args:
            module: Module instance to register
        """
        if module.module_id in self.modules:
            raise ValueError(f"Module {module.module_id} already registered")

        self.modules[module.module_id] = module
        self.logger.info(f"Registered module: {module.module_id}")

        # Log event
        asyncio.create_task(
            self.cache.log_event(
                event_type="module_registered",
                metadata={"module_id": module.module_id}
            )
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
        asyncio.create_task(
            self.cache.log_event(
                event_type="module_unregistered",
                metadata={"module_id": module_id}
            )
        )

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

    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish event to all subscribers.

        Args:
            event_type: Type of event
            data: Event data
        """
        self.logger.debug(f"Publishing event: {event_type}")

        # Log event
        await self.cache.log_event(
            event_type=event_type,
            data=data
        )

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")

        # Notify modules
        for module in self.modules.values():
            try:
                await module.on_event(event_type, data)
            except Exception as e:
                self.logger.error(
                    f"Error in module {module.module_id} event handler: {e}"
                )

    async def schedule_task(
        self,
        task_id: str,
        coro: Callable,
        interval: Optional[timedelta] = None,
        run_immediately: bool = True
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

        self.logger.info(
            f"Scheduled task: {task_id}"
            + (f" (interval: {interval})" if interval else " (one-time)")
        )

    async def get_cache(self, category: str) -> Optional[Dict[str, Any]]:
        """Get data from cache.

        Args:
            category: Cache category

        Returns:
            Cache entry or None if not found
        """
        return await self.cache.get(category)

    async def get_cache_fresh(
        self,
        category: str,
        max_age: timedelta,
        caller: str = ""
    ) -> Optional[Dict[str, Any]]:
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
                age = datetime.now() - updated
                if age > max_age:
                    who = f" ({caller})" if caller else ""
                    self.logger.warning(
                        f"Stale cache: '{category}' is {age} old "
                        f"(max {max_age}){who}"
                    )
            except (ValueError, TypeError):
                pass
        return entry

    async def set_cache(
        self,
        category: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Set data in cache and publish update event.

        Args:
            category: Cache category
            data: Data to store
            metadata: Optional metadata

        Returns:
            New version number
        """
        version = await self.cache.set(category, data, metadata)

        # Publish cache update event
        await self.publish(
            "cache_updated",
            {
                "category": category,
                "version": version,
                "timestamp": datetime.now().isoformat()
            }
        )

        return version

    async def get_module(self, module_id: str) -> Optional[Module]:
        """Get registered module by ID.

        Args:
            module_id: Module identifier

        Returns:
            Module instance or None if not found
        """
        return self.modules.get(module_id)

    def is_running(self) -> bool:
        """Check if hub is running.

        Returns:
            True if hub is running
        """
        return self._running

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on hub and modules.

        Returns:
            Health check results
        """
        return {
            "hub": {
                "running": self._running,
                "modules_count": len(self.modules),
                "tasks_count": len(self.tasks),
                "subscribers_count": sum(len(subs) for subs in self.subscribers.values())
            },
            "modules": {
                module_id: {
                    "registered": True
                }
                for module_id in self.modules.keys()
            },
            "cache": {
                "categories": await self.cache.list_categories()
            },
            "timestamp": datetime.now().isoformat()
        }
