"""Tests for config schema extension — description_layman/technical columns."""

import pytest
import pytest_asyncio

from aria.hub.cache import CacheManager


@pytest_asyncio.fixture
async def cache(tmp_path):
    cm = CacheManager(str(tmp_path / "hub.db"))
    await cm.initialize()
    yield cm
    await cm.close()


class TestConfigSchemaExtension:
    @pytest.mark.asyncio
    async def test_upsert_accepts_layman_and_technical(self, cache):
        inserted = await cache.upsert_config_default(
            key="test.param",
            default_value="42",
            value_type="number",
            label="Test",
            description="Old desc",
            category="Test",
            description_layman="Simple explanation",
            description_technical="Detailed technical explanation",
        )
        assert inserted is True
        config = await cache.get_config("test.param")
        assert config["description_layman"] == "Simple explanation"
        assert config["description_technical"] == "Detailed technical explanation"

    @pytest.mark.asyncio
    async def test_existing_configs_have_null_descriptions(self, cache):
        """Configs inserted without new fields get None."""
        await cache.upsert_config_default(
            key="old.param",
            default_value="1",
            value_type="number",
            label="Old",
            description="desc",
            category="Test",
        )
        config = await cache.get_config("old.param")
        assert config["description_layman"] is None
        assert config["description_technical"] is None

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, cache):
        """Running initialize twice doesn't fail (ALTER TABLE already exists)."""
        await cache.upsert_config_default(
            key="test.param",
            default_value="1",
            value_type="number",
            label="Test",
            description="desc",
            category="Test",
            description_layman="Simple",
            description_technical="Technical",
        )
        # Re-initialize (simulates restart)
        await cache.initialize()
        config = await cache.get_config("test.param")
        assert config["description_layman"] == "Simple"
        assert config["description_technical"] == "Technical"
