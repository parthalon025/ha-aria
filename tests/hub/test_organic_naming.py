"""Tests for organic discovery naming (heuristic + Ollama LLM)."""

from unittest.mock import AsyncMock, patch

import pytest

from aria.modules.organic_discovery.naming import heuristic_description, heuristic_name

# --- Fixtures ---


def _cluster(
    entity_ids=None,
    domains=None,
    areas=None,
    device_classes=None,
    temporal_pattern=None,
):
    """Build a cluster_info dict with sensible defaults."""
    info = {
        "entity_ids": entity_ids or ["light.living_room_1", "light.living_room_2"],
        "domains": domains or {"light": 2},
        "areas": areas or {"living_room": 2},
    }
    if device_classes is not None:
        info["device_classes"] = device_classes
    if temporal_pattern is not None:
        info["temporal_pattern"] = temporal_pattern
    return info


# --- heuristic_name tests ---


class TestHeuristicName:
    """Tests for heuristic_name."""

    def test_single_domain_included(self):
        """Single-domain cluster name includes the domain."""
        info = _cluster(domains={"light": 5}, areas={"kitchen": 5})
        name = heuristic_name(info)
        assert "light" in name

    def test_device_class_preferred_over_domain(self):
        """When device_class is present, it appears instead of domain."""
        info = _cluster(
            domains={"sensor": 4},
            areas={"garage": 4},
            device_classes={"temperature": 4},
        )
        name = heuristic_name(info)
        assert "temperature" in name
        # domain should NOT appear when device_class is more specific
        assert "sensor" not in name

    def test_dominant_area_included(self):
        """Area included when >= 60% of entities are in one area."""
        info = _cluster(
            entity_ids=[f"light.entity_{i}" for i in range(5)],
            domains={"light": 5},
            areas={"bedroom": 4, "hallway": 1},
        )
        name = heuristic_name(info)
        assert "bedroom" in name

    def test_no_dominant_area_excluded(self):
        """Area omitted when no single area reaches 60%."""
        info = _cluster(
            entity_ids=[f"light.entity_{i}" for i in range(6)],
            domains={"light": 6},
            areas={"bedroom": 2, "kitchen": 2, "office": 2},
        )
        name = heuristic_name(info)
        assert "bedroom" not in name
        assert "kitchen" not in name
        assert "office" not in name

    def test_mixed_domain_when_no_majority(self):
        """'mixed' appears when no single domain >= 50%."""
        info = _cluster(
            entity_ids=[f"entity.e_{i}" for i in range(6)],
            domains={"light": 2, "switch": 2, "sensor": 2},
            areas={"living_room": 6},
        )
        name = heuristic_name(info)
        assert "mixed" in name

    def test_time_prefix_morning(self):
        """Morning peak_hours produce a morning prefix."""
        info = _cluster(
            domains={"light": 4},
            areas={"bedroom": 4},
            temporal_pattern={"peak_hours": [6, 7, 8], "weekday_bias": 0.0},
        )
        name = heuristic_name(info)
        assert name.startswith("morning_")

    def test_time_prefix_afternoon(self):
        """Afternoon peak_hours produce an afternoon prefix."""
        info = _cluster(
            domains={"light": 3},
            areas={"office": 3},
            temporal_pattern={"peak_hours": [13, 14, 15], "weekday_bias": 0.0},
        )
        name = heuristic_name(info)
        assert name.startswith("afternoon_")

    def test_time_prefix_evening(self):
        """Evening peak_hours produce an evening prefix."""
        info = _cluster(
            domains={"light": 3},
            areas={"living_room": 3},
            temporal_pattern={"peak_hours": [19, 20, 21], "weekday_bias": 0.0},
        )
        name = heuristic_name(info)
        assert name.startswith("evening_")

    def test_time_prefix_night(self):
        """Night peak_hours produce a night prefix."""
        info = _cluster(
            domains={"light": 3},
            areas={"bedroom": 3},
            temporal_pattern={"peak_hours": [0, 1, 23], "weekday_bias": 0.0},
        )
        name = heuristic_name(info)
        assert name.startswith("night_")

    def test_always_snake_case(self):
        """Name must be lowercase snake_case — no spaces, no uppercase."""
        info = _cluster(
            domains={"binary_sensor": 3},
            areas={"Master Bedroom": 3},
            device_classes={"Motion Detector": 3},
        )
        name = heuristic_name(info)
        assert " " not in name
        assert name == name.lower()
        assert all(c.isalnum() or c == "_" for c in name)

    def test_fallback_to_cluster(self):
        """Empty inputs produce a name containing 'cluster'."""
        info = {
            "entity_ids": [],
            "domains": {},
            "areas": {},
        }
        name = heuristic_name(info)
        assert "cluster" in name

    def test_deterministic(self):
        """Same input always produces the same name."""
        info = _cluster(
            domains={"switch": 3, "light": 2},
            areas={"kitchen": 3, "dining": 2},
        )
        results = {heuristic_name(info) for _ in range(20)}
        assert len(results) == 1

    def test_no_temporal_pattern_no_time_prefix(self):
        """Without temporal_pattern, no time prefix appears."""
        info = _cluster(domains={"light": 3}, areas={"bedroom": 3})
        name = heuristic_name(info)
        for prefix in ("morning_", "afternoon_", "evening_", "night_"):
            assert not name.startswith(prefix)


# --- heuristic_description tests ---


class TestHeuristicDescription:
    """Tests for heuristic_description."""

    def test_non_empty_and_minimum_length(self):
        """Description must be non-empty and > 10 chars."""
        info = _cluster(domains={"light": 2}, areas={"kitchen": 2})
        desc = heuristic_description(info)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_includes_entity_count(self):
        """Description mentions the entity count."""
        info = _cluster(
            entity_ids=[f"light.l_{i}" for i in range(7)],
            domains={"light": 7},
            areas={"bathroom": 7},
        )
        desc = heuristic_description(info)
        assert "7" in desc

    def test_includes_domain_breakdown(self):
        """Description mentions at least one domain."""
        info = _cluster(
            domains={"switch": 3, "light": 2},
            areas={"kitchen": 5},
        )
        desc = heuristic_description(info)
        assert "switch" in desc.lower() or "light" in desc.lower()

    def test_includes_area(self):
        """Description mentions at least one area."""
        info = _cluster(
            domains={"light": 3},
            areas={"living_room": 3},
        )
        desc = heuristic_description(info)
        assert "living_room" in desc or "living room" in desc.lower()

    def test_includes_peak_hours_when_temporal(self):
        """Description mentions peak hours when temporal_pattern is present."""
        info = _cluster(
            domains={"light": 3},
            areas={"bedroom": 3},
            temporal_pattern={"peak_hours": [7, 8, 9], "weekday_bias": 0.5},
        )
        desc = heuristic_description(info)
        # Should mention peak hours in some form
        assert "peak" in desc.lower() or "hour" in desc.lower() or "7" in desc

    def test_deterministic(self):
        """Same input always produces the same description."""
        info = _cluster(
            domains={"switch": 3, "light": 2},
            areas={"kitchen": 3, "dining": 2},
        )
        results = {heuristic_description(info) for _ in range(20)}
        assert len(results) == 1


# --- Ollama LLM naming tests ---

# Shared cluster fixture for Ollama tests
CLUSTER_MIXED_ROOM = _cluster(
    entity_ids=["light.office_desk", "switch.office_fan", "sensor.office_temp"],
    domains={"light": 1, "switch": 1, "sensor": 1},
    areas={"office": 3},
    temporal_pattern={"peak_hours": [8, 9, 10], "weekday_bias": 0.8},
)


class TestOllamaName:
    @pytest.mark.asyncio
    async def test_ollama_name_returns_string(self):
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "Morning kitchen routine"
            from aria.modules.organic_discovery.naming import ollama_name

            result = await ollama_name(CLUSTER_MIXED_ROOM)
            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_ollama_name_fallback_on_error(self):
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = Exception("Ollama down")
            from aria.modules.organic_discovery.naming import ollama_name

            result = await ollama_name(CLUSTER_MIXED_ROOM)
            assert isinstance(result, str)
            assert len(result) > 0  # Should fall back to heuristic

    @pytest.mark.asyncio
    async def test_ollama_name_cleans_response(self):
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "  Morning Kitchen Routine!  "
            from aria.modules.organic_discovery.naming import ollama_name

            result = await ollama_name(CLUSTER_MIXED_ROOM)
            assert " " not in result  # Should be snake_case
            assert result == result.lower()

    @pytest.mark.asyncio
    async def test_ollama_name_fallback_on_short_result(self):
        """LLM returning a too-short name falls back to heuristic."""
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "ab"  # Only 2 chars after cleaning
            from aria.modules.organic_discovery.naming import ollama_name

            result = await ollama_name(CLUSTER_MIXED_ROOM)
            # Should fall back to heuristic, which produces a real name
            assert isinstance(result, str)
            assert len(result) > 2


class TestOllamaDescription:
    @pytest.mark.asyncio
    async def test_ollama_description_returns_string(self):
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "These are all lighting entities in the office area."
            from aria.modules.organic_discovery.naming import ollama_description

            result = await ollama_description(CLUSTER_MIXED_ROOM)
            assert isinstance(result, str)
            assert len(result) > 10

    @pytest.mark.asyncio
    async def test_ollama_description_fallback_on_error(self):
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = Exception("Ollama down")
            from aria.modules.organic_discovery.naming import ollama_description

            result = await ollama_description(CLUSTER_MIXED_ROOM)
            assert isinstance(result, str)
            assert len(result) > 10  # Heuristic always produces >10 chars

    @pytest.mark.asyncio
    async def test_ollama_description_truncated_to_200(self):
        """Long LLM responses are capped at 200 characters."""
        with patch(
            "aria.modules.organic_discovery.naming._call_ollama",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "A" * 300
            from aria.modules.organic_discovery.naming import ollama_description

            result = await ollama_description(CLUSTER_MIXED_ROOM)
            assert len(result) <= 200


# --- Ollama queue routing test ---


class TestOllamaQueueRouting:
    """Verify _call_ollama routes through ollama-queue proxy (port 7683)."""

    @pytest.mark.asyncio
    async def test_call_ollama_uses_queue_port(self):
        """_call_ollama must POST through ollama-queue proxy on port 7683."""
        from unittest.mock import MagicMock

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"response": "test_name"})

        # post() returns a context manager, not a coroutine
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            from aria.modules.organic_discovery.naming import _call_ollama

            await _call_ollama("test prompt")

        url_called = mock_session.post.call_args[0][0]
        assert "7683" in url_called, f"Expected port 7683 (ollama-queue proxy), got: {url_called}"
        assert "11434" not in url_called, "Should not use direct Ollama port 11434"
