"""Tests for Pattern Recognition module."""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

from aria.modules.patterns import PatternRecognition


# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock — avoids SQLite, matching test_intelligence.py pattern."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.modules = {}

    async def set_cache(self, category: str, data: Any, metadata: Optional[Dict] = None):
        self._cache[category] = {"data": data, "metadata": metadata}

    async def get_cache(self, category: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(category)

    def register_module(self, module):
        self.modules[module.module_id] = module

    async def publish(self, event_type: str, data: Dict[str, Any]):
        pass


# ============================================================================
# Fixture data — 7 days × 3 areas → produces 3 clusters (≥2 patterns)
# ============================================================================

# Light-on times per area per day (minutes since midnight, ±1 min variation)
_MORNING_TIMES = [389, 390, 391, 389, 390, 391, 390]   # ~06:29-06:31
_MIDDAY_TIMES = [734, 735, 736, 734, 735, 736, 735]     # ~12:14-12:16
_EVENING_TIMES = [1259, 1260, 1261, 1259, 1260, 1261, 1260]  # ~20:59-21:01


def _mock_ollama_generate(**kwargs):
    """Return a deterministic label based on area mentioned in the prompt."""
    prompt = kwargs.get("prompt", "")
    if "bedroom" in prompt.lower():
        label = "Morning routine"
    elif "kitchen" in prompt.lower():
        label = "Lunch prep"
    elif "living" in prompt.lower():
        label = "Evening wind-down"
    else:
        label = "Daily activity"
    response = MagicMock()
    response.response = label
    return response


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def log_dir(tmp_path):
    """Create logbook directory with 7 days of data across 3 areas."""
    for day_idx in range(7):
        date_str = f"2026-02-{day_idx + 1:02d}"
        events = []

        # Bedroom — morning light + motion
        m = _MORNING_TIMES[day_idx]
        h, mn = divmod(m, 60)
        events.append({
            "entity_id": "light.bedroom_main",
            "name": "Bedroom Light",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{mn:02d}:00",
        })
        events.append({
            "entity_id": "binary_sensor.bedroom_motion",
            "name": "Bedroom Motion",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{max(0, mn - 2):02d}:00",
        })

        # Kitchen — midday light + motion
        m = _MIDDAY_TIMES[day_idx]
        h, mn = divmod(m, 60)
        events.append({
            "entity_id": "light.kitchen_main",
            "name": "Kitchen Light",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{mn:02d}:00",
        })
        events.append({
            "entity_id": "binary_sensor.kitchen_motion",
            "name": "Kitchen Motion",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{max(0, mn - 1):02d}:00",
        })

        # Living room — evening light + motion
        m = _EVENING_TIMES[day_idx]
        h, mn = divmod(m, 60)
        events.append({
            "entity_id": "light.living_room_main",
            "name": "Living Room Light",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{mn:02d}:00",
        })
        events.append({
            "entity_id": "binary_sensor.living_room_motion",
            "name": "Living Room Motion",
            "state": "on",
            "when": f"{date_str}T{h:02d}:{max(0, mn - 3):02d}:00",
        })

        (tmp_path / f"{date_str}.json").write_text(json.dumps(events))

    return tmp_path


@pytest.fixture
def patterns_module(hub, log_dir):
    """Create pattern recognition module with mock hub and fixture data."""
    module = PatternRecognition(
        hub=hub,
        log_dir=log_dir,
        min_pattern_frequency=1,
        min_support=0.3,
        min_confidence=0.3,
    )
    hub.register_module(module)
    return module


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_registration(hub, patterns_module):
    """Test that pattern recognition module registers successfully."""
    assert "pattern_recognition" in hub.modules
    assert hub.modules["pattern_recognition"] == patterns_module


@pytest.mark.asyncio
async def test_pattern_detection(hub, patterns_module):
    """Test that patterns are detected from historical data."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        await patterns_module.initialize()

    cache_data = await hub.get_cache("patterns")
    assert cache_data is not None
    assert "data" in cache_data

    patterns_data = cache_data["data"]
    assert "patterns" in patterns_data
    assert "pattern_count" in patterns_data
    assert "areas_analyzed" in patterns_data

    assert patterns_data["pattern_count"] >= 2


@pytest.mark.asyncio
async def test_pattern_structure(hub, patterns_module):
    """Test that detected patterns have required fields."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        await patterns_module.initialize()

    cache_data = await hub.get_cache("patterns")
    patterns = cache_data["data"]["patterns"]

    assert len(patterns) > 0

    for pattern in patterns:
        # Required fields
        assert "pattern_id" in pattern
        assert "name" in pattern
        assert "typical_time" in pattern
        assert "variance_minutes" in pattern
        assert "frequency" in pattern
        assert "confidence" in pattern
        assert "associated_signals" in pattern
        assert "llm_description" in pattern

        # Validate formats
        assert isinstance(pattern["typical_time"], str)
        assert ":" in pattern["typical_time"]  # HH:MM format
        assert isinstance(pattern["variance_minutes"], int)
        assert isinstance(pattern["frequency"], int)
        assert isinstance(pattern["confidence"], float)
        assert 0 <= pattern["confidence"] <= 1
        assert isinstance(pattern["associated_signals"], list)
        assert isinstance(pattern["llm_description"], str)
        assert len(pattern["llm_description"]) > 0


@pytest.mark.asyncio
async def test_llm_interpretation(hub, patterns_module):
    """Test that LLM generates semantic descriptions."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        await patterns_module.initialize()

    cache_data = await hub.get_cache("patterns")
    patterns = cache_data["data"]["patterns"]

    for pattern in patterns:
        llm_desc = pattern["llm_description"]
        # Should not be empty or default error message
        assert llm_desc != ""
        assert llm_desc != "Failed to generate LLM description"
        # Should be reasonably short (acceptance criteria: semantic label)
        assert len(llm_desc) < 100


@pytest.mark.asyncio
async def test_dtw_distance():
    """Test DTW distance calculation."""
    module = PatternRecognition(
        hub=None,
        log_dir=Path("/tmp"),
        min_pattern_frequency=1,
    )

    # Identical sequences
    s1 = [60, 120, 180]
    s2 = [60, 120, 180]
    assert module._dtw_distance(s1, s2) == 0.0

    # Different lengths
    s1 = [60, 120]
    s2 = [60, 120, 180]
    dist = module._dtw_distance(s1, s2)
    assert dist > 0

    # Very different sequences
    s1 = [60, 120, 180]
    s2 = [300, 360, 420]
    dist = module._dtw_distance(s1, s2)
    assert dist > 300  # Should be large


@pytest.mark.asyncio
async def test_extract_area_from_name():
    """Test area extraction from entity names."""
    module = PatternRecognition(
        hub=None,
        log_dir=Path("/tmp"),
        min_pattern_frequency=1,
    )

    assert module._extract_area_from_name("Bedroom Light", "light.bedroom_main") == "bedroom"
    assert module._extract_area_from_name("Kitchen Motion", "binary_sensor.kitchen_motion") == "kitchen"
    assert module._extract_area_from_name("Unknown Device", "sensor.temp_1") == "general"


@pytest.mark.asyncio
async def test_strip_think_tags():
    """Test removal of deepseek-r1 thinking tags."""
    module = PatternRecognition(
        hub=None,
        log_dir=Path("/tmp"),
        min_pattern_frequency=1,
    )

    text = "<think>Some reasoning here</think>Morning routine"
    result = module._strip_think_tags(text)
    assert result == "Morning routine"

    text = "Before<think>reasoning</think>After"
    result = module._strip_think_tags(text)
    assert result == "BeforeAfter"


@pytest.mark.asyncio
async def test_on_event_handler(hub, patterns_module):
    """Test that module responds to hub events."""
    # Module should not crash on events
    await patterns_module.on_event("test_event", {"data": "test"})
    await patterns_module.on_event("cache_updated", {"category": "test"})
    # No assertion needed - just verify no exceptions
