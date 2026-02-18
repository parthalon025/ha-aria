"""Tests for aria audit CLI subcommands."""

import pytest

from aria.hub.audit import AuditLogger


@pytest.fixture
async def audit_with_data(tmp_path):
    al = AuditLogger()
    await al.initialize(str(tmp_path / "audit.db"))
    await al.log(event_type="cache.write", source="hub", action="set", subject="intelligence")
    await al.log(
        event_type="config.change",
        source="user",
        action="set",
        subject="shadow.exploration_rate",
        detail={"old": 0.15, "new": 0.20},
        severity="warning",
    )
    await al.flush()
    await al.log_request("req-1", "GET", "/api/cache", 200, 12.5, "127.0.0.1")
    await al.log_startup({"activity": "running"}, {"key": "val"}, 500.0)
    await al.log_curation_change("sensor.temp", None, "active", None, 1, "discovery", "auto")
    yield al
    await al.shutdown()


class TestAuditEventsCommand:
    @pytest.mark.asyncio
    async def test_events_returns_results(self, audit_with_data):
        events = await audit_with_data.query_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_events_filter_by_type(self, audit_with_data):
        events = await audit_with_data.query_events(event_type="cache.write")
        assert len(events) == 1


class TestAuditRequestsCommand:
    @pytest.mark.asyncio
    async def test_requests_returns_results(self, audit_with_data):
        reqs = await audit_with_data.query_requests()
        assert len(reqs) == 1


class TestAuditStartupsCommand:
    @pytest.mark.asyncio
    async def test_startups_returns_results(self, audit_with_data):
        startups = await audit_with_data.query_startups()
        assert len(startups) == 1


class TestAuditCurationCommand:
    @pytest.mark.asyncio
    async def test_curation_returns_results(self, audit_with_data):
        history = await audit_with_data.query_curation("sensor.temp")
        assert len(history) == 1


class TestAuditVerifyCommand:
    @pytest.mark.asyncio
    async def test_integrity_all_valid(self, audit_with_data):
        result = await audit_with_data.verify_integrity()
        assert result["total"] == 2
        assert result["invalid"] == 0
