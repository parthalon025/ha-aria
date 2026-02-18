"""Tests for AuditLogger — schema, write path, queries, integrity."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aria.hub.audit import AuditLogger


@pytest.fixture
async def audit_logger(tmp_path):
    db_path = str(tmp_path / "audit.db")
    al = AuditLogger()
    await al.initialize(db_path)
    yield al
    await al.shutdown()


class TestSchema:
    """Tables created, WAL mode enabled."""

    async def test_tables_created(self, audit_logger):
        async with audit_logger._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
        assert "audit_events" in tables
        assert "audit_requests" in tables
        assert "audit_startups" in tables
        assert "audit_curation_history" in tables

    async def test_wal_mode_enabled(self, audit_logger):
        async with audit_logger._db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        assert row[0] == "wal"


class TestLogEvent:
    """log() and flush, checksum integrity, severity default, request_id correlation."""

    async def test_log_and_flush(self, audit_logger):
        await audit_logger.log(
            event_type="test",
            source="unit_test",
            action="create",
            subject="item_1",
            detail={"key": "value"},
        )
        await audit_logger.flush()

        events = await audit_logger.query_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "test"
        assert events[0]["source"] == "unit_test"
        assert events[0]["action"] == "create"
        assert events[0]["subject"] == "item_1"
        assert events[0]["detail"] == {"key": "value"}

    async def test_checksum_integrity(self, audit_logger):
        import hashlib

        await audit_logger.log(
            event_type="test",
            source="unit_test",
            action="verify",
            subject="item_2",
            detail={"foo": "bar"},
        )
        await audit_logger.flush()

        events = await audit_logger.query_events()
        event = events[0]
        detail_str = json.dumps({"foo": "bar"})
        payload = event["timestamp"] + "test" + "unit_test" + "verify" + detail_str
        expected = hashlib.sha256(payload.encode()).hexdigest()
        assert event["checksum"] == expected

    async def test_severity_default(self, audit_logger):
        await audit_logger.log(event_type="test", source="unit_test", action="default_sev")
        await audit_logger.flush()

        events = await audit_logger.query_events()
        assert events[0]["severity"] == "info"

    async def test_request_id_correlation(self, audit_logger):
        await audit_logger.log(
            event_type="test",
            source="unit_test",
            action="correlated",
            request_id="req-123",
        )
        await audit_logger.flush()

        events = await audit_logger.query_events(request_id="req-123")
        assert len(events) == 1
        assert events[0]["request_id"] == "req-123"


class TestLogRequest:
    """log_request writes API request directly."""

    async def test_log_request(self, audit_logger):
        await audit_logger.log_request(
            request_id="req-001",
            method="GET",
            path="/api/cache",
            status_code=200,
            duration_ms=42.5,
            client_ip="127.0.0.1",
        )

        requests = await audit_logger.query_requests()
        assert len(requests) == 1
        assert requests[0]["method"] == "GET"
        assert requests[0]["path"] == "/api/cache"
        assert requests[0]["status_code"] == 200
        assert requests[0]["duration_ms"] == 42.5

    async def test_log_request_with_error(self, audit_logger):
        await audit_logger.log_request(
            request_id="req-002",
            method="POST",
            path="/api/fail",
            status_code=500,
            duration_ms=100.0,
            client_ip="127.0.0.1",
            error="Internal Server Error",
        )

        requests = await audit_logger.query_requests()
        assert len(requests) == 1
        assert requests[0]["error"] == "Internal Server Error"
        assert requests[0]["status_code"] == 500


class TestLogStartup:
    """log_startup writes startup snapshot directly."""

    async def test_log_startup(self, audit_logger):
        await audit_logger.log_startup(
            python_version="3.14.3",
            modules_loaded={"intelligence": True, "activity": True},
            config_snapshot={"port": 8001},
            system_memory_mb=32000,
            pid=12345,
            duration_ms=1500.0,
        )

        startups = await audit_logger.query_startups()
        assert len(startups) == 1
        assert startups[0]["python_version"] == "3.14.3"
        assert startups[0]["modules_loaded"] == {"intelligence": True, "activity": True}
        assert startups[0]["config_snapshot"] == {"port": 8001}
        assert startups[0]["system_memory_mb"] == 32000
        assert startups[0]["pid"] == 12345


class TestLogCuration:
    """log_curation_change writes entity curation change directly."""

    async def test_log_curation_change(self, audit_logger):
        await audit_logger.log_curation_change(
            entity_id="light.living_room",
            old_status="active",
            new_status="stale",
            old_tier="standard",
            new_tier="reduced",
            reason="No state changes in 72h",
            changed_by="curation_engine",
        )

        history = await audit_logger.query_curation(entity_id="light.living_room")
        assert len(history) == 1
        assert history[0]["entity_id"] == "light.living_room"
        assert history[0]["old_status"] == "active"
        assert history[0]["new_status"] == "stale"
        assert history[0]["old_tier"] == "standard"
        assert history[0]["new_tier"] == "reduced"
        assert history[0]["reason"] == "No state changes in 72h"
        assert history[0]["changed_by"] == "curation_engine"


class TestBufferOverflow:
    """Dropped events counter when buffer_size=5 and 10 events."""

    async def test_dropped_events(self, tmp_path):
        db_path = str(tmp_path / "audit_overflow.db")
        al = AuditLogger(buffer_size=5)
        await al.initialize(db_path)

        try:
            # Log 10 events — 5 should be dropped
            for i in range(10):
                await al.log(
                    event_type="overflow",
                    source="test",
                    action=f"event_{i}",
                )

            stats = al.get_buffer_stats()
            assert stats["dropped_events"] == 5
        finally:
            await al.shutdown()


class TestShutdownFlush:
    """Shutdown flushes buffer."""

    async def test_shutdown_flushes(self, tmp_path):
        db_path = str(tmp_path / "audit_shutdown.db")
        al = AuditLogger()
        await al.initialize(db_path)

        await al.log(event_type="test", source="shutdown_test", action="before_shutdown")

        # Shutdown should flush
        await al.shutdown()

        # Reopen to verify data persisted
        al2 = AuditLogger()
        await al2.initialize(db_path)
        events = await al2.query_events()
        assert len(events) == 1
        assert events[0]["action"] == "before_shutdown"
        await al2.shutdown()


class TestPruning:
    """Prune removes old events, keeps recent."""

    async def test_prune_old_events(self, audit_logger):
        # Insert an old event directly
        old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        await audit_logger._db.execute(
            "INSERT INTO audit_events (timestamp, event_type, source, action, severity, checksum) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (old_ts, "old", "test", "old_action", "info", "fakechecksum"),
        )
        await audit_logger._db.commit()

        # Insert a recent event
        await audit_logger.log(event_type="recent", source="test", action="recent_action")
        await audit_logger.flush()

        # Prune events older than 30 days
        pruned = await audit_logger.prune(retention_days=30)
        assert pruned > 0

        events = await audit_logger.query_events()
        assert all(e["event_type"] != "old" for e in events)
        assert any(e["event_type"] == "recent" for e in events)


class TestExportArchive:
    """Export creates JSONL file."""

    async def test_export_creates_jsonl(self, audit_logger, tmp_path):
        await audit_logger.log(
            event_type="export_test",
            source="test",
            action="exported",
            detail={"data": 42},
        )
        await audit_logger.flush()

        output_dir = tmp_path / "archive"
        output_dir.mkdir()

        files = await audit_logger.export_archive(
            before_date=datetime.now(UTC) + timedelta(days=1),
            output_dir=str(output_dir),
        )

        assert len(files) > 0
        for f in files:
            assert f.endswith(".jsonl")
            path = Path(f)
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) >= 1
            record = json.loads(lines[0])
            assert record["event_type"] == "export_test"


class TestStats:
    """Stats counts by type and severity."""

    async def test_get_stats(self, audit_logger):
        await audit_logger.log(event_type="alpha", source="test", action="a1")
        await audit_logger.log(event_type="alpha", source="test", action="a2")
        await audit_logger.log(event_type="beta", source="test", action="b1", severity="warning")
        await audit_logger.flush()

        stats = await audit_logger.get_stats()
        assert stats["by_type"]["alpha"] == 2
        assert stats["by_type"]["beta"] == 1
        assert stats["by_severity"]["info"] == 2
        assert stats["by_severity"]["warning"] == 1


class TestQueryTimeline:
    """Timeline returns events for a subject, chronological."""

    async def test_timeline_for_subject(self, audit_logger):
        await audit_logger.log(event_type="a", source="test", action="first", subject="entity_1")
        await audit_logger.log(event_type="b", source="test", action="second", subject="entity_1")
        await audit_logger.log(event_type="c", source="test", action="other", subject="entity_2")
        await audit_logger.flush()

        timeline = await audit_logger.query_timeline(subject="entity_1")
        assert len(timeline) == 2
        assert timeline[0]["action"] == "first"
        assert timeline[1]["action"] == "second"


class TestVerifyIntegrity:
    """Integrity check returns valid for clean events."""

    async def test_integrity_valid(self, audit_logger):
        await audit_logger.log(
            event_type="integrity",
            source="test",
            action="check",
            detail={"verified": True},
        )
        await audit_logger.flush()

        result = await audit_logger.verify_integrity()
        assert result["total"] == 1
        assert result["valid"] == 1
        assert result["invalid"] == 0
        assert result["details"] == []

    async def test_integrity_detects_tamper(self, audit_logger):
        await audit_logger.log(event_type="integrity", source="test", action="tampered")
        await audit_logger.flush()

        # Tamper with the stored event
        await audit_logger._db.execute("UPDATE audit_events SET action='hacked'")
        await audit_logger._db.commit()

        result = await audit_logger.verify_integrity()
        assert result["invalid"] == 1
        assert len(result["details"]) == 1
