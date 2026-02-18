"""Tests for audit middleware on hub core methods."""

import asyncio

import pytest

from aria.hub.audit import AuditLogger
from aria.hub.core import IntelligenceHub, Module


@pytest.fixture
async def hub_with_audit(tmp_path):
    cache_path = str(tmp_path / "hub.db")
    audit_path = str(tmp_path / "audit.db")
    hub = IntelligenceHub(cache_path)
    await hub.initialize()
    audit = AuditLogger()
    await audit.initialize(audit_path)
    hub.set_audit_logger(audit)
    yield hub, audit
    await hub.shutdown()
    await audit.shutdown()


class TestRegisterModuleAudit:
    @pytest.mark.asyncio
    async def test_register_emits_audit_event(self, hub_with_audit):
        hub, audit = hub_with_audit
        mod = Module("test_mod", hub)
        hub.register_module(mod)
        await asyncio.sleep(0.1)  # let create_task run
        await audit.flush()
        events = await audit.query_events(event_type="module.register")
        assert len(events) == 1
        assert events[0]["subject"] == "test_mod"

    @pytest.mark.asyncio
    async def test_collision_emits_warning(self, hub_with_audit):
        hub, audit = hub_with_audit
        mod1 = Module("dup_mod", hub)
        mod2 = Module("dup_mod", hub)
        hub.register_module(mod1)
        with pytest.raises(ValueError):
            hub.register_module(mod2)
        await asyncio.sleep(0.1)
        await audit.flush()
        events = await audit.query_events(event_type="module.collision")
        assert len(events) == 1
        assert events[0]["severity"] == "warning"


class TestSetCacheAudit:
    @pytest.mark.asyncio
    async def test_set_cache_emits_audit(self, hub_with_audit):
        hub, audit = hub_with_audit
        await hub.set_cache("test_category", {"value": 42})
        await audit.flush()
        events = await audit.query_events(event_type="cache.write")
        assert len(events) == 1
        assert events[0]["subject"] == "test_category"


class TestHubAuditMethod:
    @pytest.mark.asyncio
    async def test_hub_audit_convenience(self, hub_with_audit):
        hub, audit = hub_with_audit
        await hub.emit_audit(
            event_type="custom.event",
            source="test_module",
            action="decision",
            subject="entity_x",
            detail={"reason": "test"},
        )
        await audit.flush()
        events = await audit.query_events(event_type="custom.event")
        assert len(events) == 1
        assert events[0]["detail"]["reason"] == "test"

    @pytest.mark.asyncio
    async def test_hub_audit_noop_without_logger(self, tmp_path):
        hub = IntelligenceHub(str(tmp_path / "hub.db"))
        await hub.initialize()
        # Should not raise even without audit logger
        await hub.emit_audit("test.event", "hub", "test")
        await hub.shutdown()


class TestStartupAudit:
    @pytest.mark.asyncio
    async def test_startup_snapshot_logged(self, tmp_path):
        from aria.hub.audit import AuditLogger

        audit = AuditLogger()
        await audit.initialize(str(tmp_path / "audit.db"))
        await audit.log_startup(
            modules={"activity": "running"},
            config_snapshot={"key": "value"},
            duration_ms=500.0,
        )
        startups = await audit.query_startups(limit=1)
        assert len(startups) == 1
        assert startups[0]["duration_ms"] == 500.0
        assert startups[0]["modules_loaded"]["activity"] == "running"
        await audit.shutdown()
