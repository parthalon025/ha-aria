"""Tests for Phase 2 cache tables: config, entity_curation, config_history."""

import sys
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.cache import CacheManager
from aria.hub.constants import CACHE_CONFIG, CACHE_ENTITY_CURATION, CACHE_CONFIG_HISTORY


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def cache(tmp_path):
    """Create and initialize a CacheManager with a temp DB."""
    db_path = str(tmp_path / "test_hub.db")
    cm = CacheManager(db_path)
    await cm.initialize()
    yield cm
    await cm.close()


async def _seed_config(cache, key="shadow.min_confidence", value="0.3",
                       value_type="number", category="Shadow Engine",
                       min_value=0.05, max_value=0.9, step=0.05):
    """Helper to seed a single config parameter."""
    await cache.upsert_config_default(
        key=key, default_value=value, value_type=value_type,
        label=key, description=f"Test param {key}", category=category,
        min_value=min_value, max_value=max_value, step=step,
    )


async def _seed_curation(cache, entity_id="light.living_room",
                         status="included", tier=3, reason="Default"):
    """Helper to seed a single curation record."""
    await cache.upsert_curation(
        entity_id=entity_id, status=status, tier=tier, reason=reason,
    )


# ============================================================================
# Constants
# ============================================================================


class TestConstants:

    def test_config_constant(self):
        assert CACHE_CONFIG == "config"

    def test_entity_curation_constant(self):
        assert CACHE_ENTITY_CURATION == "entity_curation"

    def test_config_history_constant(self):
        assert CACHE_CONFIG_HISTORY == "config_history"


# ============================================================================
# Table creation
# ============================================================================


class TestTableCreation:

    @pytest.mark.asyncio
    async def test_config_table_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='config'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_entity_curation_table_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_curation'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_config_history_table_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='config_history'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_curation_indexes_exist(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_entity_curation_%'"
        )
        rows = await cursor.fetchall()
        names = {row["name"] for row in rows}
        assert "idx_entity_curation_tier" in names
        assert "idx_entity_curation_status" in names

    @pytest.mark.asyncio
    async def test_history_index_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_config_history_key'"
        )
        assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_reinitialize_is_safe(self, cache):
        await cache.initialize()
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='config'"
        )
        assert await cursor.fetchone() is not None


# ============================================================================
# Config CRUD
# ============================================================================


class TestConfigCRUD:

    @pytest.mark.asyncio
    async def test_upsert_config_default_inserts(self, cache):
        inserted = await cache.upsert_config_default(
            key="test.param", default_value="42", value_type="number",
            label="Test", description="A test param", category="Testing",
            min_value=0, max_value=100,
        )
        assert inserted is True

        config = await cache.get_config("test.param")
        assert config is not None
        assert config["key"] == "test.param"
        assert config["value"] == "42"
        assert config["default_value"] == "42"
        assert config["value_type"] == "number"
        assert config["category"] == "Testing"

    @pytest.mark.asyncio
    async def test_upsert_config_default_ignores_existing(self, cache):
        await _seed_config(cache, key="x", value="10", min_value=0, max_value=100)
        # Change the value manually
        await cache.set_config("x", "99")

        # Re-seed should NOT overwrite
        inserted = await cache.upsert_config_default(
            key="x", default_value="10", value_type="number",
        )
        assert inserted is False

        config = await cache.get_config("x")
        assert config["value"] == "99"  # preserved

    @pytest.mark.asyncio
    async def test_get_config_nonexistent(self, cache):
        result = await cache.get_config("does.not.exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_config(self, cache):
        await _seed_config(cache, key="a.one", category="A")
        await _seed_config(cache, key="b.two", category="B")

        configs = await cache.get_all_config()
        assert len(configs) == 2
        # Ordered by category then key
        assert configs[0]["key"] == "a.one"
        assert configs[1]["key"] == "b.two"

    @pytest.mark.asyncio
    async def test_set_config_updates_value(self, cache):
        await _seed_config(cache, key="test.k", value="1", min_value=0, max_value=100)

        result = await cache.set_config("test.k", "2")
        assert result["value"] == "2"
        assert result["default_value"] == "1"  # default unchanged

    @pytest.mark.asyncio
    async def test_set_config_writes_history(self, cache):
        await _seed_config(cache, key="test.k", value="1", min_value=0, max_value=100)
        await cache.set_config("test.k", "2", changed_by="alice")

        history = await cache.get_config_history(key="test.k")
        assert len(history) == 1
        assert history[0]["old_value"] == "1"
        assert history[0]["new_value"] == "2"
        assert history[0]["changed_by"] == "alice"

    @pytest.mark.asyncio
    async def test_set_config_validates_min(self, cache):
        await _seed_config(cache, key="x", value="0.3", min_value=0.05, max_value=0.9)

        with pytest.raises(ValueError, match="below minimum"):
            await cache.set_config("x", "0.01")

    @pytest.mark.asyncio
    async def test_set_config_validates_max(self, cache):
        await _seed_config(cache, key="x", value="0.3", min_value=0.05, max_value=0.9)

        with pytest.raises(ValueError, match="above maximum"):
            await cache.set_config("x", "1.5")

    @pytest.mark.asyncio
    async def test_set_config_validates_number_type(self, cache):
        await _seed_config(cache, key="x", value="0.3")

        with pytest.raises(ValueError, match="Expected number"):
            await cache.set_config("x", "not-a-number")

    @pytest.mark.asyncio
    async def test_set_config_nonexistent_raises(self, cache):
        with pytest.raises(ValueError, match="not found"):
            await cache.set_config("no.such.key", "42")

    @pytest.mark.asyncio
    async def test_reset_config(self, cache):
        await _seed_config(cache, key="test.k", value="10", min_value=0, max_value=100)
        await cache.set_config("test.k", "99")

        result = await cache.reset_config("test.k")
        assert result["value"] == "10"

    @pytest.mark.asyncio
    async def test_reset_config_writes_history(self, cache):
        await _seed_config(cache, key="test.k", value="10", min_value=0, max_value=100)
        await cache.set_config("test.k", "99")
        await cache.reset_config("test.k", changed_by="bob")

        history = await cache.get_config_history(key="test.k")
        assert len(history) == 2
        assert history[0]["new_value"] == "10"  # most recent = reset
        assert history[0]["changed_by"] == "bob"

    @pytest.mark.asyncio
    async def test_get_config_value_number(self, cache):
        await _seed_config(cache, key="x", value="42")
        val = await cache.get_config_value("x")
        assert val == 42
        assert isinstance(val, int)

    @pytest.mark.asyncio
    async def test_get_config_value_float(self, cache):
        await _seed_config(cache, key="x", value="0.75")
        val = await cache.get_config_value("x")
        assert val == 0.75
        assert isinstance(val, float)

    @pytest.mark.asyncio
    async def test_get_config_value_fallback(self, cache):
        val = await cache.get_config_value("missing.key", fallback=99)
        assert val == 99

    @pytest.mark.asyncio
    async def test_get_config_value_boolean(self, cache):
        await cache.upsert_config_default(
            key="flag", default_value="true", value_type="boolean",
        )
        val = await cache.get_config_value("flag")
        assert val is True

    @pytest.mark.asyncio
    async def test_set_config_validates_select_options(self, cache):
        await cache.upsert_config_default(
            key="mode", default_value="fast", value_type="select",
            options="fast,slow,auto",
        )
        # Valid
        await cache.set_config("mode", "slow")
        # Invalid
        with pytest.raises(ValueError, match="not in options"):
            await cache.set_config("mode", "turbo")


# ============================================================================
# Entity curation CRUD
# ============================================================================


class TestCurationCRUD:

    @pytest.mark.asyncio
    async def test_upsert_and_get(self, cache):
        await _seed_curation(cache, entity_id="light.kitchen", status="included", tier=3)

        result = await cache.get_curation("light.kitchen")
        assert result is not None
        assert result["entity_id"] == "light.kitchen"
        assert result["status"] == "included"
        assert result["tier"] == 3
        assert result["human_override"] is False

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, cache):
        await _seed_curation(cache, entity_id="light.kitchen", status="included", tier=3)
        await cache.upsert_curation(
            entity_id="light.kitchen", status="excluded", tier=1,
            reason="Too noisy",
        )

        result = await cache.get_curation("light.kitchen")
        assert result["status"] == "excluded"
        assert result["tier"] == 1
        assert result["reason"] == "Too noisy"

    @pytest.mark.asyncio
    async def test_get_curation_nonexistent(self, cache):
        result = await cache.get_curation("no.such.entity")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_curation(self, cache):
        await _seed_curation(cache, entity_id="light.a", tier=3)
        await _seed_curation(cache, entity_id="sensor.b", tier=1, status="auto_excluded")
        await _seed_curation(cache, entity_id="switch.c", tier=2, status="excluded")

        results = await cache.get_all_curation()
        assert len(results) == 3
        # Ordered by tier then entity_id
        assert results[0]["tier"] == 1
        assert results[1]["tier"] == 2
        assert results[2]["tier"] == 3

    @pytest.mark.asyncio
    async def test_get_curation_summary(self, cache):
        await _seed_curation(cache, entity_id="a", tier=1, status="auto_excluded")
        await _seed_curation(cache, entity_id="b", tier=1, status="auto_excluded")
        await _seed_curation(cache, entity_id="c", tier=2, status="excluded")
        await _seed_curation(cache, entity_id="d", tier=3, status="included")
        await _seed_curation(cache, entity_id="e", tier=3, status="included")
        await _seed_curation(cache, entity_id="f", tier=3, status="promoted")

        summary = await cache.get_curation_summary()
        assert summary["total"] == 6
        assert summary["per_tier"][1] == 2
        assert summary["per_tier"][2] == 1
        assert summary["per_tier"][3] == 3
        assert summary["per_status"]["auto_excluded"] == 2
        assert summary["per_status"]["included"] == 2
        assert summary["per_status"]["promoted"] == 1

    @pytest.mark.asyncio
    async def test_bulk_update_curation(self, cache):
        await _seed_curation(cache, entity_id="a", status="included")
        await _seed_curation(cache, entity_id="b", status="included")
        await _seed_curation(cache, entity_id="c", status="included")

        count = await cache.bulk_update_curation(
            ["a", "b"], status="excluded", decided_by="alice"
        )
        assert count == 2

        a = await cache.get_curation("a")
        assert a["status"] == "excluded"
        assert a["human_override"] is True
        assert a["decided_by"] == "alice"

        c = await cache.get_curation("c")
        assert c["status"] == "included"  # unchanged

    @pytest.mark.asyncio
    async def test_bulk_update_empty_list(self, cache):
        count = await cache.bulk_update_curation([], status="excluded")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_included_entity_ids(self, cache):
        await _seed_curation(cache, entity_id="light.a", status="included")
        await _seed_curation(cache, entity_id="light.b", status="promoted")
        await _seed_curation(cache, entity_id="sensor.c", status="excluded")
        await _seed_curation(cache, entity_id="update.d", status="auto_excluded")

        included = await cache.get_included_entity_ids()
        assert included == {"light.a", "light.b"}

    @pytest.mark.asyncio
    async def test_curation_with_metrics(self, cache):
        metrics = {"event_rate_day": 150.5, "unique_states": 4}
        await cache.upsert_curation(
            entity_id="sensor.x", status="included", tier=3,
            metrics=metrics,
        )

        result = await cache.get_curation("sensor.x")
        assert result["metrics"] == metrics


# ============================================================================
# Config history
# ============================================================================


class TestConfigHistory:

    @pytest.mark.asyncio
    async def test_history_empty(self, cache):
        history = await cache.get_config_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_history_tracks_changes(self, cache):
        await _seed_config(cache, key="x", value="1", min_value=0, max_value=100)
        await cache.set_config("x", "2")
        await cache.set_config("x", "3")

        history = await cache.get_config_history()
        assert len(history) == 2
        # Most recent first
        assert history[0]["new_value"] == "3"
        assert history[0]["old_value"] == "2"
        assert history[1]["new_value"] == "2"
        assert history[1]["old_value"] == "1"

    @pytest.mark.asyncio
    async def test_history_filter_by_key(self, cache):
        await _seed_config(cache, key="a", value="1", min_value=0, max_value=100)
        await _seed_config(cache, key="b", value="10", min_value=0, max_value=100)
        await cache.set_config("a", "2")
        await cache.set_config("b", "20")

        history = await cache.get_config_history(key="a")
        assert len(history) == 1
        assert history[0]["key"] == "a"

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, cache):
        await _seed_config(cache, key="x", value="0", min_value=0, max_value=100)
        for i in range(10):
            await cache.set_config("x", str(i + 1))

        history = await cache.get_config_history(limit=3)
        assert len(history) == 3


# ============================================================================
# Existing cache unaffected
# ============================================================================


class TestExistingCacheUnaffected:

    @pytest.mark.asyncio
    async def test_original_cache_still_works(self, cache):
        await cache.set("test_category", {"key": "value"})
        result = await cache.get("test_category")
        assert result["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_predictions_still_work(self, cache):
        await cache.insert_prediction(
            prediction_id="test-001",
            timestamp=datetime.now().isoformat(),
            context={"test": True},
            predictions=[{"action": "test"}],
            confidence=0.9,
            window_seconds=60,
        )
        rows = await cache.get_recent_predictions(limit=1)
        assert len(rows) == 1
