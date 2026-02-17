"""Per-model fallback tracking with TTL-based expiry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    model_name: str
    from_tier: int
    to_tier: int
    error: str
    memory_mb: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class FallbackTracker:
    def __init__(self, ttl_days: int = 7):
        self.ttl = timedelta(days=ttl_days)
        self._events: dict[str, FallbackEvent] = {}

    def record(
        self,
        model_name: str,
        from_tier: int,
        to_tier: int,
        error: str,
        memory_mb: float | None = None,
    ) -> None:
        event = FallbackEvent(
            model_name=model_name,
            from_tier=from_tier,
            to_tier=to_tier,
            error=error,
            memory_mb=memory_mb,
        )
        self._events[model_name] = event
        logger.warning(f"Model fallback: {model_name} tier {from_tier}→{to_tier} ({error})")

    def is_fallen_back(self, model_name: str) -> bool:
        event = self._events.get(model_name)
        if event is None:
            return False
        if datetime.now() - event.timestamp > self.ttl:
            del self._events[model_name]
            return False
        return True

    def get_effective_tier(self, model_name: str, original_tier: int) -> int:
        if self.is_fallen_back(model_name):
            return self._events[model_name].to_tier
        return original_tier

    def active_fallbacks(self) -> list[FallbackEvent]:
        now = datetime.now()
        expired = [k for k, v in self._events.items() if now - v.timestamp > self.ttl]
        for k in expired:
            del self._events[k]
        return list(self._events.values())

    def clear(self, model_name: str) -> None:
        self._events.pop(model_name, None)

    def to_dict(self) -> list[dict]:
        return [
            {
                "model": e.model_name,
                "from_tier": e.from_tier,
                "to_tier": e.to_tier,
                "error": e.error,
                "memory_mb": e.memory_mb,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.active_fallbacks()
        ]
