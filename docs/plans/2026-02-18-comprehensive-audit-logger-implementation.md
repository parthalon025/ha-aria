# Comprehensive Audit Logger — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full-spectrum audit logging system for ARIA that captures all significant events in a queryable, exportable, tamper-evident SQLite database with CLI access, real-time streaming, and alerting integration.

**Architecture:** Dedicated `AuditLogger` class in `aria/hub/audit.py` with separate SQLite DB (`audit.db`), write-behind async buffer, FastAPI middleware for request tracing, WebSocket streaming, CLI subcommands, and watchdog alerting bridge.

**Tech Stack:** Python asyncio, aiosqlite, FastAPI middleware, WebSocket, argparse CLI, SHA-256 checksums

**Design doc:** `docs/plans/2026-02-18-comprehensive-audit-logger-design.md`

---

### Task 1: AuditLogger Core — Schema & Write Path

**Files:**
- Create: `aria/hub/audit.py`
- Test: `tests/hub/test_audit.py`

This task builds the `AuditLogger` class: DB init, schema creation, `log()` with write-behind buffer, `log_request()`, `log_startup()`, `log_curation_change()`, checksum generation, `shutdown()` flush.

**Step 1: Write the failing tests**

Create `tests/hub/test_audit.py`:

```python
"""Tests for AuditLogger core — schema, writes, buffer, checksums."""

import asyncio
import hashlib
import json
import os
import tempfile

import pytest
import aiosqlite

from aria.hub.audit import AuditLogger


@pytest.fixture
async def audit_logger(tmp_path):
    """Create an AuditLogger with a temp DB."""
    db_path = str(tmp_path / "audit.db")
    al = AuditLogger()
    await al.initialize(db_path)
    yield al
    await al.shutdown()


class TestSchema:
    @pytest.mark.asyncio
    async def test_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        al = AuditLogger()
        await al.initialize(db_path)
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in await cursor.fetchall()]
        await al.shutdown()
        assert "audit_events" in tables
        assert "audit_requests" in tables
        assert "audit_startups" in tables
        assert "audit_curation_history" in tables

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        al = AuditLogger()
        await al.initialize(db_path)
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA journal_mode")
            mode = (await cursor.fetchone())[0]
        await al.shutdown()
        assert mode == "wal"


class TestLogEvent:
    @pytest.mark.asyncio
    async def test_log_and_flush(self, audit_logger):
        await audit_logger.log(
            event_type="cache.write",
            source="test_module",
            action="set",
            subject="intelligence",
            detail={"size": 1024},
            severity="info",
        )
        await audit_logger.flush()
        events = await audit_logger.query_events(limit=10)
        assert len(events) == 1
        assert events[0]["event_type"] == "cache.write"
        assert events[0]["source"] == "test_module"
        assert events[0]["subject"] == "intelligence"

    @pytest.mark.asyncio
    async def test_checksum_integrity(self, audit_logger):
        await audit_logger.log(
            event_type="config.change",
            source="hub",
            action="set",
            detail={"key": "x"},
        )
        await audit_logger.flush()
        events = await audit_logger.query_events(limit=1)
        ev = events[0]
        # Recompute checksum
        raw = ev["timestamp"] + ev["event_type"] + ev["source"] + ev["action"] + json.dumps(ev.get("detail"))
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert ev["checksum"] == expected

    @pytest.mark.asyncio
    async def test_severity_default_is_info(self, audit_logger):
        await audit_logger.log(
            event_type="test.event", source="hub", action="test"
        )
        await audit_logger.flush()
        events = await audit_logger.query_events(limit=1)
        assert events[0]["severity"] == "info"

    @pytest.mark.asyncio
    async def test_request_id_correlation(self, audit_logger):
        await audit_logger.log(
            event_type="cache.write",
            source="hub",
            action="set",
            request_id="req-123",
        )
        await audit_logger.flush()
        events = await audit_logger.query_events(request_id="req-123")
        assert len(events) == 1


class TestLogRequest:
    @pytest.mark.asyncio
    async def test_log_request(self, audit_logger):
        await audit_logger.log_request(
            request_id="req-abc",
            method="GET",
            path="/api/cache",
            status_code=200,
            duration_ms=12.5,
            client_ip="127.0.0.1",
        )
        requests = await audit_logger.query_requests(limit=10)
        assert len(requests) == 1
        assert requests[0]["method"] == "GET"
        assert requests[0]["path"] == "/api/cache"
        assert requests[0]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_log_request_with_error(self, audit_logger):
        await audit_logger.log_request(
            request_id="req-err",
            method="PUT",
            path="/api/config/bad",
            status_code=500,
            duration_ms=3.1,
            client_ip="127.0.0.1",
            error="Internal server error",
        )
        requests = await audit_logger.query_requests(status_min=400)
        assert len(requests) == 1
        assert requests[0]["error"] == "Internal server error"


class TestLogStartup:
    @pytest.mark.asyncio
    async def test_log_startup(self, audit_logger):
        await audit_logger.log_startup(
            modules={"activity_monitor": "running", "shadow_engine": "failed"},
            config_snapshot={"activity.window_minutes": 15},
            duration_ms=1234.5,
        )
        # Query via raw DB since no query method yet for startups
        async with aiosqlite.connect(audit_logger._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM audit_startups")
            rows = [dict(r) for r in await cursor.fetchall()]
        assert len(rows) == 1
        assert rows[0]["duration_ms"] == 1234.5
        modules = json.loads(rows[0]["modules_loaded"])
        assert modules["shadow_engine"] == "failed"


class TestLogCuration:
    @pytest.mark.asyncio
    async def test_log_curation_change(self, audit_logger):
        await audit_logger.log_curation_change(
            entity_id="sensor.living_room_temp",
            old_status=None,
            new_status="active",
            old_tier=None,
            new_tier=1,
            reason="discovery",
            changed_by="auto",
        )
        async with aiosqlite.connect(audit_logger._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM audit_curation_history")
            rows = [dict(r) for r in await cursor.fetchall()]
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "sensor.living_room_temp"
        assert rows[0]["new_status"] == "active"


class TestBufferOverflow:
    @pytest.mark.asyncio
    async def test_dropped_events_counter(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        al = AuditLogger(buffer_size=5)
        await al.initialize(db_path)
        # Fill buffer beyond capacity
        for i in range(10):
            await al.log(
                event_type="test.flood", source="hub", action="test"
            )
        stats = al.get_buffer_stats()
        assert stats["dropped_events"] > 0
        await al.shutdown()


class TestShutdownFlush:
    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffer(self, tmp_path):
        db_path = str(tmp_path / "audit.db")
        al = AuditLogger()
        await al.initialize(db_path)
        await al.log(event_type="test.event", source="hub", action="test")
        # Don't manually flush — shutdown should do it
        await al.shutdown()
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM audit_events")
            row = await cursor.fetchone()
        assert row[0] == 1
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.hub.audit'`

**Step 3: Implement AuditLogger**

Create `aria/hub/audit.py`:

```python
"""ARIA Audit Logger — tamper-evident event logging with write-behind buffer."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT,
    detail TEXT,
    request_id TEXT,
    severity TEXT NOT NULL DEFAULT 'info',
    checksum TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_source ON audit_events(source);
CREATE INDEX IF NOT EXISTS idx_audit_events_subject ON audit_events(subject);
CREATE INDEX IF NOT EXISTS idx_audit_events_request_id ON audit_events(request_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity);

CREATE TABLE IF NOT EXISTS audit_requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER,
    duration_ms REAL,
    client_ip TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_requests_timestamp ON audit_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_requests_path ON audit_requests(path);
CREATE INDEX IF NOT EXISTS idx_audit_requests_status ON audit_requests(status_code);

CREATE TABLE IF NOT EXISTS audit_startups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    python_version TEXT NOT NULL,
    modules_loaded TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    system_memory_mb INTEGER,
    pid INTEGER,
    duration_ms REAL
);

CREATE TABLE IF NOT EXISTS audit_curation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    old_tier INTEGER,
    new_tier INTEGER NOT NULL,
    reason TEXT,
    changed_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_curation_entity ON audit_curation_history(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_curation_timestamp ON audit_curation_history(timestamp);
"""

DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_FLUSH_INTERVAL = 0.5  # seconds
DEFAULT_FLUSH_BATCH = 100


def _compute_checksum(
    timestamp: str, event_type: str, source: str, action: str, detail: str | None
) -> str:
    raw = timestamp + event_type + source + action + (detail or "")
    return hashlib.sha256(raw.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_system_memory_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


class AuditLogger:
    """Centralized audit logging for ARIA hub."""

    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self._db_path: str | None = None
        self._conn: aiosqlite.Connection | None = None
        self._buffer: asyncio.Queue | None = None
        self._buffer_size = buffer_size
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._dropped_events = 0
        self._total_written = 0

    async def initialize(self, db_path: str) -> None:
        """Open audit.db, create tables, start write buffer."""
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = await aiosqlite.connect(db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()
        self._buffer = asyncio.Queue(maxsize=self._buffer_size)
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def shutdown(self) -> None:
        """Flush buffer and close connection."""
        self._running = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def log(
        self,
        event_type: str,
        source: str,
        action: str,
        subject: str | None = None,
        detail: dict | None = None,
        request_id: str | None = None,
        severity: str = "info",
    ) -> None:
        """Buffer an audit event for async write."""
        if not self._buffer:
            return
        timestamp = _now_iso()
        detail_json = json.dumps(detail) if detail else None
        checksum = _compute_checksum(timestamp, event_type, source, action, detail_json)
        row = (timestamp, event_type, source, action, subject, detail_json, request_id, severity, checksum)
        try:
            self._buffer.put_nowait(("event", row))
        except asyncio.QueueFull:
            self._dropped_events += 1
            logger.warning("Audit buffer full — event dropped (total dropped: %d)", self._dropped_events)

    async def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: str,
        error: str | None = None,
    ) -> None:
        """Log an API request (written immediately, not buffered)."""
        if not self._conn:
            return
        timestamp = _now_iso()
        await self._conn.execute(
            "INSERT OR REPLACE INTO audit_requests (id, timestamp, method, path, status_code, duration_ms, client_ip, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, timestamp, method, path, status_code, duration_ms, client_ip, error),
        )
        await self._conn.commit()

    async def log_startup(
        self,
        modules: dict[str, str],
        config_snapshot: dict,
        duration_ms: float,
    ) -> None:
        """Log hub startup context."""
        if not self._conn:
            return
        await self._conn.execute(
            "INSERT INTO audit_startups (timestamp, python_version, modules_loaded, config_snapshot, system_memory_mb, pid, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _now_iso(),
                sys.version,
                json.dumps(modules),
                json.dumps(config_snapshot),
                _get_system_memory_mb(),
                os.getpid(),
                duration_ms,
            ),
        )
        await self._conn.commit()

    async def log_curation_change(
        self,
        entity_id: str,
        old_status: str | None,
        new_status: str,
        old_tier: int | None,
        new_tier: int,
        reason: str | None = None,
        changed_by: str = "auto",
    ) -> None:
        """Log entity curation status change."""
        if not self._conn:
            return
        await self._conn.execute(
            "INSERT INTO audit_curation_history (timestamp, entity_id, old_status, new_status, old_tier, new_tier, reason, changed_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now_iso(), entity_id, old_status, new_status, old_tier, new_tier, reason, changed_by),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def query_events(
        self,
        event_type: str | None = None,
        source: str | None = None,
        subject: str | None = None,
        severity: str | None = None,
        since: str | None = None,
        until: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit events with optional filters."""
        if not self._conn:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM audit_events{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def query_requests(
        self,
        path: str | None = None,
        method: str | None = None,
        status_min: int | None = None,
        status_max: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query API request log."""
        if not self._conn:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if path:
            clauses.append("path = ?")
            params.append(path)
        if method:
            clauses.append("method = ?")
            params.append(method)
        if status_min is not None:
            clauses.append("status_code >= ?")
            params.append(status_min)
        if status_max is not None:
            clauses.append("status_code <= ?")
            params.append(status_max)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM audit_requests{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def query_timeline(self, subject: str, since: str | None = None, until: str | None = None) -> list[dict]:
        """Get all events for a given subject, ordered chronologically."""
        if not self._conn:
            return []
        clauses = ["subject = ?"]
        params: list[Any] = [subject]
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = " WHERE " + " AND ".join(clauses)
        cursor = await self._conn.execute(f"SELECT * FROM audit_events{where} ORDER BY timestamp ASC", params)
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_stats(self, since: str | None = None) -> dict:
        """Get audit statistics."""
        if not self._conn:
            return {}
        where = f" WHERE timestamp >= '{since}'" if since else ""
        cursor = await self._conn.execute(
            f"SELECT event_type, severity, COUNT(*) as cnt FROM audit_events{where} GROUP BY event_type, severity"
        )
        rows = await cursor.fetchall()
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        total = 0
        for r in rows:
            by_type[r[0]] = by_type.get(r[0], 0) + r[2]
            by_severity[r[1]] = by_severity.get(r[1], 0) + r[2]
            total += r[2]
        return {
            "total_events": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "buffer": self.get_buffer_stats(),
        }

    async def query_startups(self, limit: int = 10) -> list[dict]:
        """Query hub startup snapshots."""
        if not self._conn:
            return []
        cursor = await self._conn.execute(
            "SELECT * FROM audit_startups ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def query_curation(self, entity_id: str, limit: int = 50) -> list[dict]:
        """Query entity curation history."""
        if not self._conn:
            return []
        cursor = await self._conn.execute(
            "SELECT * FROM audit_curation_history WHERE entity_id = ? ORDER BY timestamp DESC LIMIT ?",
            (entity_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def verify_integrity(self, since: str | None = None) -> dict:
        """Verify checksums of audit events."""
        if not self._conn:
            return {"total": 0, "valid": 0, "invalid": 0, "details": []}
        where = f" WHERE timestamp >= '{since}'" if since else ""
        cursor = await self._conn.execute(f"SELECT * FROM audit_events{where}")
        rows = await cursor.fetchall()
        total = 0
        valid = 0
        invalid_details: list[dict] = []
        for r in rows:
            total += 1
            row_dict = self._row_to_dict(r)
            expected = _compute_checksum(
                row_dict["timestamp"],
                row_dict["event_type"],
                row_dict["source"],
                row_dict["action"],
                row_dict.get("detail") if isinstance(row_dict.get("detail"), str) else json.dumps(row_dict.get("detail")),
            )
            if row_dict.get("checksum") == expected:
                valid += 1
            else:
                invalid_details.append({"id": row_dict["id"], "expected": expected, "actual": row_dict.get("checksum")})
        return {"total": total, "valid": valid, "invalid": total - valid, "details": invalid_details}

    async def prune(self, retention_days: int) -> int:
        """Delete audit events older than retention_days."""
        if not self._conn:
            return 0
        cutoff = datetime.now(timezone.utc).isoformat()
        # Compute cutoff date
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff = cutoff_dt.isoformat()
        cursor = await self._conn.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff,))
        await self._conn.execute("DELETE FROM audit_requests WHERE timestamp < ?", (cutoff,))
        await self._conn.commit()
        return cursor.rowcount

    async def export_archive(self, before_date: str, output_dir: str) -> str:
        """Export events before a date to JSONL archive file."""
        os.makedirs(output_dir, exist_ok=True)
        month = before_date[:7]  # YYYY-MM
        output_path = os.path.join(output_dir, f"{month}.jsonl")
        if not self._conn:
            return output_path
        cursor = await self._conn.execute(
            "SELECT * FROM audit_events WHERE timestamp < ? ORDER BY timestamp ASC", (before_date,)
        )
        rows = await cursor.fetchall()
        with open(output_path, "a") as f:
            for r in rows:
                f.write(json.dumps(self._row_to_dict(r)) + "\n")
        return output_path

    # ------------------------------------------------------------------
    # Buffer internals
    # ------------------------------------------------------------------

    async def flush(self) -> int:
        """Flush buffered events to DB. Returns count written."""
        if not self._conn or not self._buffer:
            return 0
        batch: list[tuple] = []
        while not self._buffer.empty():
            try:
                kind, row = self._buffer.get_nowait()
                if kind == "event":
                    batch.append(row)
            except asyncio.QueueEmpty:
                break
        if not batch:
            return 0
        await self._conn.executemany(
            "INSERT INTO audit_events (timestamp, event_type, source, action, subject, detail, request_id, severity, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        await self._conn.commit()
        self._total_written += len(batch)
        return len(batch)

    async def _flush_loop(self) -> None:
        """Background loop: flush every 500ms or when 100 items accumulate."""
        while self._running:
            try:
                await asyncio.sleep(DEFAULT_FLUSH_INTERVAL)
                if self._buffer and self._buffer.qsize() > 0:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Audit flush error: %s", e)

    def get_buffer_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "queue_size": self._buffer.qsize() if self._buffer else 0,
            "buffer_capacity": self._buffer_size,
            "dropped_events": self._dropped_events,
            "total_written": self._total_written,
        }

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict:
        """Convert aiosqlite Row to dict, parsing JSON detail field."""
        d = dict(row)
        if "detail" in d and isinstance(d["detail"], str):
            try:
                d["detail"] = json.loads(d["detail"])
            except (json.JSONDecodeError, TypeError):
                pass
        if "modules_loaded" in d and isinstance(d["modules_loaded"], str):
            try:
                d["modules_loaded"] = json.loads(d["modules_loaded"])
            except (json.JSONDecodeError, TypeError):
                pass
        if "config_snapshot" in d and isinstance(d["config_snapshot"], str):
            try:
                d["config_snapshot"] = json.loads(d["config_snapshot"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_audit.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/hub/audit.py tests/hub/test_audit.py
git commit -m "feat: add AuditLogger core — schema, write-behind buffer, checksums"
```

---

### Task 2: Config Defaults & Retention Pruning

**Files:**
- Modify: `aria/hub/config_defaults.py` (add 10 audit.* parameters at end of CONFIG_DEFAULTS list)
- Test: `tests/hub/test_audit.py` (add pruning and stats tests)

**Step 1: Write the failing tests**

Add to `tests/hub/test_audit.py`:

```python
class TestPruning:
    @pytest.mark.asyncio
    async def test_prune_removes_old_events(self, audit_logger):
        # Insert an event with old timestamp
        await audit_logger._conn.execute(
            "INSERT INTO audit_events (timestamp, event_type, source, action, severity, checksum) VALUES (?, ?, ?, ?, ?, ?)",
            ("2020-01-01T00:00:00+00:00", "old.event", "hub", "test", "info", "abc"),
        )
        await audit_logger._conn.commit()
        deleted = await audit_logger.prune(retention_days=90)
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_prune_keeps_recent_events(self, audit_logger):
        await audit_logger.log(event_type="recent.event", source="hub", action="test")
        await audit_logger.flush()
        deleted = await audit_logger.prune(retention_days=90)
        events = await audit_logger.query_events()
        assert len(events) == 1  # recent event kept


class TestExportArchive:
    @pytest.mark.asyncio
    async def test_export_creates_jsonl(self, audit_logger, tmp_path):
        await audit_logger._conn.execute(
            "INSERT INTO audit_events (timestamp, event_type, source, action, severity, checksum) VALUES (?, ?, ?, ?, ?, ?)",
            ("2020-06-15T00:00:00+00:00", "old.event", "hub", "test", "info", "abc"),
        )
        await audit_logger._conn.commit()
        output_dir = str(tmp_path / "archive")
        path = await audit_logger.export_archive("2021-01-01T00:00:00+00:00", output_dir)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_counts(self, audit_logger):
        await audit_logger.log(event_type="cache.write", source="hub", action="set")
        await audit_logger.log(event_type="cache.write", source="hub", action="set", severity="warning")
        await audit_logger.log(event_type="config.change", source="user", action="set")
        await audit_logger.flush()
        stats = await audit_logger.get_stats()
        assert stats["total_events"] == 3
        assert stats["by_type"]["cache.write"] == 2
        assert stats["by_severity"]["info"] == 2
        assert stats["by_severity"]["warning"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_audit.py -k "Pruning or Export or Stats" -v`
Expected: PASS (these test methods from Task 1 implementation)

**Step 3: Add audit config defaults**

Append to `CONFIG_DEFAULTS` list in `aria/hub/config_defaults.py`:

```python
    # ── Audit Logger ───────────────────────────────────────────────────
    {
        "key": "audit.enabled",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Audit Logging Enabled",
        "description": "Master switch for audit event logging.",
        "category": "Audit",
    },
    {
        "key": "audit.retention_days",
        "default_value": "90",
        "value_type": "number",
        "label": "Audit Retention (days)",
        "description": "Days to retain audit records before pruning.",
        "category": "Audit",
        "min_value": 7,
        "max_value": 365,
        "step": 1,
    },
    {
        "key": "audit.log_api_requests",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Log API Requests",
        "description": "Log all API requests to audit database.",
        "category": "Audit",
    },
    {
        "key": "audit.log_cache_writes",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Log Cache Writes",
        "description": "Log cache write operations to audit database.",
        "category": "Audit",
    },
    {
        "key": "audit.buffer_size",
        "default_value": "10000",
        "value_type": "number",
        "label": "Write Buffer Size",
        "description": "Maximum events in write-behind buffer before dropping.",
        "category": "Audit",
        "min_value": 1000,
        "max_value": 100000,
        "step": 1000,
    },
    {
        "key": "audit.flush_interval_ms",
        "default_value": "500",
        "value_type": "number",
        "label": "Flush Interval (ms)",
        "description": "Maximum time between database flushes.",
        "category": "Audit",
        "min_value": 100,
        "max_value": 5000,
        "step": 100,
    },
    {
        "key": "audit.alert_on_errors",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Alert on Errors",
        "description": "Send Telegram alerts for error-severity audit events.",
        "category": "Audit",
    },
    {
        "key": "audit.alert_threshold",
        "default_value": "10",
        "value_type": "number",
        "label": "Alert Threshold",
        "description": "Number of errors in window before alerting.",
        "category": "Audit",
        "min_value": 1,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "audit.alert_window_minutes",
        "default_value": "5",
        "value_type": "number",
        "label": "Alert Window (min)",
        "description": "Window for error counting in alerts.",
        "category": "Audit",
        "min_value": 1,
        "max_value": 60,
        "step": 1,
    },
    {
        "key": "audit.archive_on_prune",
        "default_value": "true",
        "value_type": "boolean",
        "label": "Archive Before Prune",
        "description": "Export expired records to JSONL before pruning.",
        "category": "Audit",
    },
```

**Step 4: Run tests to verify everything passes**

Run: `.venv/bin/python -m pytest tests/hub/test_audit.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/hub/config_defaults.py tests/hub/test_audit.py
git commit -m "feat: add audit config defaults (10 parameters) and pruning/export tests"
```

---

### Task 3: Wire AuditLogger into Hub Core

**Files:**
- Modify: `aria/hub/core.py` (add `_audit_logger` attribute, `hub.audit()` method, wrap `register_module`, `set_cache`, `shutdown`)
- Test: `tests/hub/test_audit_middleware.py` (new)

**Step 1: Write the failing tests**

Create `tests/hub/test_audit_middleware.py`:

```python
"""Tests for audit middleware on hub core methods."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.hub.audit import AuditLogger
from aria.hub.core import IntelligenceHub, Module


@pytest.fixture
async def hub_with_audit(tmp_path):
    """Hub with real AuditLogger attached."""
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
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_audit_middleware.py -v`
Expected: FAIL — `AttributeError: 'IntelligenceHub' object has no attribute 'set_audit_logger'`

**Step 3: Modify `aria/hub/core.py`**

Add to `IntelligenceHub.__init__`:
```python
self._audit_logger = None
```

Add methods:
```python
def set_audit_logger(self, audit_logger):
    """Attach audit logger to hub."""
    self._audit_logger = audit_logger

async def emit_audit(
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
```

Modify `register_module`:
```python
def register_module(self, module: Module):
    if module.module_id in self.modules:
        if self._audit_logger:
            asyncio.create_task(self._audit_logger.log(
                event_type="module.collision",
                source="hub",
                action="attempt",
                subject=module.module_id,
                detail={
                    "existing_class": type(self.modules[module.module_id]).__name__,
                    "new_class": type(module).__name__,
                },
                severity="warning",
            ))
        raise ValueError(f"Module {module.module_id} already registered")

    self.modules[module.module_id] = module
    self.module_status[module.module_id] = "registered"
    self.logger.info(f"Registered module: {module.module_id}")

    if self._audit_logger:
        asyncio.create_task(self._audit_logger.log(
            event_type="module.register",
            source="hub",
            action="register",
            subject=module.module_id,
            detail={"class": type(module).__name__},
        ))

    asyncio.create_task(
        self.cache.log_event(event_type="module_registered", metadata={"module_id": module.module_id})
    )
```

Modify `set_cache` to add audit:
```python
async def set_cache(self, category, data, metadata=None):
    version = await self.cache.set(category, data, metadata)
    if self._audit_logger:
        await self._audit_logger.log(
            event_type="cache.write",
            source="hub",
            action="set",
            subject=category,
            detail={"version": version, "data_keys": list(data.keys()) if isinstance(data, dict) else None},
        )
    await self.publish("cache_updated", {"category": category, "version": version, "timestamp": datetime.now().isoformat()})
    return version
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_audit_middleware.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/hub/core.py tests/hub/test_audit_middleware.py
git commit -m "feat: wire AuditLogger into hub core — register, set_cache, emit_audit"
```

---

### Task 4: FastAPI Request Middleware & Audit API Routes

**Files:**
- Modify: `aria/hub/api.py` (add request middleware, `/api/audit/*` routes)
- Test: `tests/hub/test_api_audit.py` (new)

**Step 1: Write the failing tests**

Create `tests/hub/test_api_audit.py`:

```python
"""Tests for /api/audit/* endpoints and request middleware."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from aria.hub.api import create_api
from aria.hub.audit import AuditLogger


@pytest.fixture
async def audit_logger(tmp_path):
    al = AuditLogger()
    await al.initialize(str(tmp_path / "audit.db"))
    yield al
    await al.shutdown()


@pytest.fixture
def mock_hub(audit_logger):
    hub = MagicMock()
    hub.cache = MagicMock()
    hub.modules = {}
    hub.subscribers = {}
    hub.subscribe = MagicMock()
    hub._request_count = 0
    hub._audit_logger = audit_logger
    hub.get_uptime_seconds = MagicMock(return_value=0)
    hub.get_module = MagicMock(return_value=None)
    hub.get_cache = AsyncMock(return_value=None)
    return hub


@pytest.fixture
def client(mock_hub):
    app = create_api(mock_hub)
    return TestClient(app)


class TestRequestMiddleware:
    def test_request_id_header(self, client):
        resp = client.get("/")
        assert "X-Request-ID" in resp.headers

    def test_request_logged_to_audit(self, client, audit_logger):
        client.get("/health")
        # Force sync — TestClient is synchronous
        import asyncio
        loop = asyncio.get_event_loop()
        requests = loop.run_until_complete(audit_logger.query_requests(limit=10))
        assert len(requests) >= 1


class TestAuditEventsEndpoint:
    def test_get_events_empty(self, client):
        resp = client.get("/api/audit/events")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["events"], list)

    def test_get_events_with_filters(self, client):
        resp = client.get("/api/audit/events?type=cache.write&severity=error&limit=10")
        assert resp.status_code == 200


class TestAuditRequestsEndpoint:
    def test_get_requests(self, client):
        resp = client.get("/api/audit/requests")
        assert resp.status_code == 200


class TestAuditStatsEndpoint:
    def test_get_stats(self, client):
        resp = client.get("/api/audit/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_events" in data


class TestAuditStartupsEndpoint:
    def test_get_startups(self, client):
        resp = client.get("/api/audit/startups")
        assert resp.status_code == 200


class TestAuditIntegrityEndpoint:
    def test_verify_integrity(self, client):
        resp = client.get("/api/audit/integrity")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "valid" in data
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_api_audit.py -v`
Expected: FAIL — routes not defined

**Step 3: Modify `aria/hub/api.py`**

Add to `create_api()`:
1. Replace the existing timing middleware with a combined audit + timing middleware
2. Add a `_register_audit_routes` function
3. Call it from `create_api()`

Middleware (replace existing):
```python
@app.middleware("http")
async def audit_request_middleware(request: Request, call_next):
    from uuid import uuid4
    request_id = str(uuid4())
    request.state.request_id = request_id
    hub._request_count += 1
    start = time.monotonic()

    response = await call_next(request)

    duration_ms = (time.monotonic() - start) * 1000
    if duration_ms > 1000:
        logger.warning(f"{request.method} {request.url.path} took {duration_ms/1000:.2f}s")
    else:
        logger.debug(f"{request.method} {request.url.path} took {duration_ms:.1f}ms")

    response.headers["X-Request-ID"] = request_id

    if hasattr(hub, "_audit_logger") and hub._audit_logger:
        error_msg = None if response.status_code < 400 else f"HTTP {response.status_code}"
        await hub._audit_logger.log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=request.client.host if request.client else "unknown",
            error=error_msg,
        )

    return response
```

Route registration function:
```python
def _register_audit_routes(router: APIRouter, hub: IntelligenceHub):
    @router.get("/api/audit/events")
    async def get_audit_events(
        type: str | None = None,
        source: str | None = None,
        subject: str | None = None,
        severity: str | None = None,
        since: str | None = None,
        until: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"events": [], "total": 0}
        events = await hub._audit_logger.query_events(
            event_type=type, source=source, subject=subject,
            severity=severity, since=since, until=until,
            request_id=request_id, limit=limit, offset=offset,
        )
        return {"events": events, "total": len(events)}

    @router.get("/api/audit/requests")
    async def get_audit_requests(
        path: str | None = None,
        method: str | None = None,
        status_min: int | None = None,
        since: str | None = None,
        limit: int = 100,
    ):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"requests": [], "total": 0}
        reqs = await hub._audit_logger.query_requests(
            path=path, method=method, status_min=status_min, since=since, limit=limit,
        )
        return {"requests": reqs, "total": len(reqs)}

    @router.get("/api/audit/timeline/{subject}")
    async def get_audit_timeline(subject: str, since: str | None = None, until: str | None = None):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"events": []}
        events = await hub._audit_logger.query_timeline(subject, since=since, until=until)
        return {"events": events}

    @router.get("/api/audit/stats")
    async def get_audit_stats(since: str | None = None):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"total_events": 0, "by_type": {}, "by_severity": {}, "buffer": {}}
        return await hub._audit_logger.get_stats(since=since)

    @router.get("/api/audit/startups")
    async def get_audit_startups(limit: int = 10):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"startups": []}
        startups = await hub._audit_logger.query_startups(limit=limit)
        return {"startups": startups}

    @router.get("/api/audit/curation/{entity_id}")
    async def get_audit_curation(entity_id: str, limit: int = 50):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"history": []}
        history = await hub._audit_logger.query_curation(entity_id, limit=limit)
        return {"history": history}

    @router.get("/api/audit/integrity")
    async def get_audit_integrity(since: str | None = None):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            return {"total": 0, "valid": 0, "invalid": 0, "details": []}
        return await hub._audit_logger.verify_integrity(since=since)

    @router.post("/api/audit/export")
    async def post_audit_export(before_date: str = Body(...), format: str = Body("jsonl")):
        if not hasattr(hub, "_audit_logger") or not hub._audit_logger:
            raise HTTPException(status_code=503, detail="Audit logger not available")
        output_dir = os.path.expanduser("~/ha-logs/intelligence/audit-archive")
        path = await hub._audit_logger.export_archive(before_date, output_dir)
        return {"exported_to": path}
```

Add `_register_audit_routes(router, hub)` to `create_api()` alongside the other route registrations.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_api_audit.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_audit.py
git commit -m "feat: add audit request middleware and /api/audit/* REST endpoints"
```

---

### Task 5: Wire Audit into Hub Startup (cli.py)

**Files:**
- Modify: `aria/cli.py` (initialize AuditLogger in `start()`, log startup snapshot, schedule audit pruning)

**Step 1: Write the failing test**

Add to `tests/hub/test_audit_middleware.py`:

```python
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
```

**Step 2: Run test to verify it passes** (this tests the already-implemented method)

Run: `.venv/bin/python -m pytest tests/hub/test_audit_middleware.py::TestStartupAudit -v`
Expected: PASS

**Step 3: Modify `aria/cli.py`**

In the `start()` function (the `serve` command handler):

After hub initialization, before module registration:
```python
from aria.hub.audit import AuditLogger

audit_logger = AuditLogger()
audit_db_path = os.path.join(os.path.dirname(cache_path), "audit.db")
await audit_logger.initialize(audit_db_path)
hub.set_audit_logger(audit_logger)
```

After module registration completes, log startup:
```python
startup_duration = (time.monotonic() - start_time) * 1000
config_snapshot = {}
try:
    config_snapshot = await hub.cache.get_all_config()
except Exception:
    pass
await audit_logger.log_startup(
    modules=hub.module_status,
    config_snapshot=config_snapshot,
    duration_ms=startup_duration,
)
```

In the shutdown handler, before hub shutdown:
```python
await audit_logger.shutdown()
```

Schedule audit pruning alongside existing prune:
```python
async def _prune_audit():
    retention = 90  # default; could read from config
    try:
        config_val = await hub.cache.get_config_value("audit.retention_days")
        if config_val:
            retention = int(config_val)
    except Exception:
        pass
    deleted = await audit_logger.prune(retention)
    if deleted:
        logger.info(f"Audit pruning: {deleted} records removed")

await hub.schedule_task("prune_audit", _prune_audit, interval=timedelta(hours=24))
```

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/hub/ -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/cli.py tests/hub/test_audit_middleware.py
git commit -m "feat: wire AuditLogger into hub startup — init, snapshot, shutdown, prune"
```

---

### Task 6: CLI Subcommands (`aria audit`)

**Files:**
- Modify: `aria/cli.py` (add `audit` subparser and handler)
- Test: `tests/test_cli_audit.py` (new)

**Step 1: Write the failing tests**

Create `tests/test_cli_audit.py`:

```python
"""Tests for aria audit CLI subcommands."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.hub.audit import AuditLogger


@pytest.fixture
async def audit_with_data(tmp_path):
    al = AuditLogger()
    await al.initialize(str(tmp_path / "audit.db"))
    await al.log(event_type="cache.write", source="hub", action="set", subject="intelligence")
    await al.log(event_type="config.change", source="user", action="set", subject="shadow.exploration_rate",
                 detail={"old": 0.15, "new": 0.20}, severity="warning")
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
```

**Step 2: Run tests to verify they pass** (these test query methods already implemented)

Run: `.venv/bin/python -m pytest tests/test_cli_audit.py -v`
Expected: PASS

**Step 3: Add CLI subcommands to `aria/cli.py`**

Add audit subparser in `_build_parser()`:
```python
audit_parser = subparsers.add_parser("audit", help="Query audit log")
audit_sub = audit_parser.add_subparsers(dest="audit_command")

ev_parser = audit_sub.add_parser("events", help="Query audit events")
ev_parser.add_argument("--type", dest="event_type")
ev_parser.add_argument("--source")
ev_parser.add_argument("--subject")
ev_parser.add_argument("--severity")
ev_parser.add_argument("--since")
ev_parser.add_argument("--until")
ev_parser.add_argument("--request-id")
ev_parser.add_argument("--limit", type=int, default=50)
ev_parser.add_argument("--json", action="store_true")

req_parser = audit_sub.add_parser("requests", help="Query API request log")
req_parser.add_argument("--path")
req_parser.add_argument("--method")
req_parser.add_argument("--status", type=int)
req_parser.add_argument("--since")
req_parser.add_argument("--limit", type=int, default=50)
req_parser.add_argument("--json", action="store_true")

tl_parser = audit_sub.add_parser("timeline", help="Timeline for a subject")
tl_parser.add_argument("subject")
tl_parser.add_argument("--since")
tl_parser.add_argument("--json", action="store_true")

stats_parser = audit_sub.add_parser("stats", help="Audit statistics")
stats_parser.add_argument("--since")

startups_parser = audit_sub.add_parser("startups", help="Recent hub startups")
startups_parser.add_argument("--limit", type=int, default=10)

cur_parser = audit_sub.add_parser("curation", help="Entity curation history")
cur_parser.add_argument("entity_id")
cur_parser.add_argument("--limit", type=int, default=50)

verify_parser = audit_sub.add_parser("verify", help="Integrity check")
verify_parser.add_argument("--since")

export_parser = audit_sub.add_parser("export", help="Export archive")
export_parser.add_argument("--before", required=True)
export_parser.add_argument("--output", default=os.path.expanduser("~/ha-logs/intelligence/audit-archive"))

tail_parser = audit_sub.add_parser("tail", help="Live tail audit events")
tail_parser.add_argument("--types")
tail_parser.add_argument("--severity-min", default="info")
```

Add handler function `_handle_audit(args)` that opens audit.db read-only and runs the appropriate query, formatting output as table or JSON.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_cli_audit.py tests/hub/test_audit.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/cli.py tests/test_cli_audit.py
git commit -m "feat: add 'aria audit' CLI subcommands — events, requests, timeline, stats, verify, export, tail"
```

---

### Task 7: WebSocket Audit Streaming

**Files:**
- Modify: `aria/hub/api.py` (add `/ws/audit` endpoint)
- Modify: `aria/hub/audit.py` (add subscriber notification on flush)
- Test: `tests/hub/test_api_audit.py` (add WebSocket tests)

**Step 1: Write the failing tests**

Add to `tests/hub/test_api_audit.py`:

```python
class TestAuditWebSocket:
    def test_ws_audit_connects(self, client):
        with client.websocket_connect("/ws/audit") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_api_audit.py::TestAuditWebSocket -v`
Expected: FAIL — WebSocket route not found

**Step 3: Add WebSocket endpoint**

In `aria/hub/api.py`, add `/ws/audit` WebSocket endpoint in `create_api()`:

```python
@app.websocket("/ws/audit")
async def audit_websocket(websocket: WebSocket):
    if _ARIA_API_KEY:
        token = websocket.query_params.get("token")
        if token != _ARIA_API_KEY:
            await websocket.close(code=4003)
            return

    await websocket.accept()
    await websocket.send_json({"type": "connected", "message": "Connected to ARIA Audit stream"})

    type_filter = websocket.query_params.get("types", "").split(",") if websocket.query_params.get("types") else []
    severity_min = websocket.query_params.get("severity_min", "info")

    queue = asyncio.Queue(maxsize=1000)

    if hasattr(hub, "_audit_logger") and hub._audit_logger:
        hub._audit_logger.add_subscriber(queue)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json({"type": "audit_event", "data": event})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if hasattr(hub, "_audit_logger") and hub._audit_logger:
            hub._audit_logger.remove_subscriber(queue)
```

In `aria/hub/audit.py`, add subscriber management:

```python
def __init__(self, ...):
    ...
    self._subscribers: list[asyncio.Queue] = []

def add_subscriber(self, queue: asyncio.Queue):
    self._subscribers.append(queue)

def remove_subscriber(self, queue: asyncio.Queue):
    self._subscribers.discard(queue) if hasattr(self._subscribers, 'discard') else None
    try:
        self._subscribers.remove(queue)
    except ValueError:
        pass
```

In `flush()`, after writing batch, notify subscribers:

```python
for row in batch:
    event_dict = {
        "timestamp": row[0], "event_type": row[1], "source": row[2],
        "action": row[3], "subject": row[4], "severity": row[7],
    }
    for sub in self._subscribers:
        try:
            sub.put_nowait(event_dict)
        except asyncio.QueueFull:
            pass
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api_audit.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/hub/api.py aria/hub/audit.py tests/hub/test_api_audit.py
git commit -m "feat: add /ws/audit WebSocket endpoint for real-time audit streaming"
```

---

### Task 8: Watchdog Alerting Integration

**Files:**
- Modify: `aria/watchdog.py` (add `check_audit_alerts` function)
- Test: `tests/hub/test_watchdog.py` (add audit alert tests)

**Step 1: Write the failing tests**

Add to existing `tests/hub/test_watchdog.py`:

```python
class TestAuditAlerts:
    def test_check_audit_alerts_no_errors(self):
        """No audit errors → OK result."""
        from aria.watchdog import check_audit_alerts
        # Mock audit.db with no recent errors
        result = check_audit_alerts(audit_db_path="/nonexistent/audit.db")
        assert result.level == "OK"

    def test_check_audit_alerts_returns_warning(self, tmp_path):
        """Many recent errors → WARNING result."""
        import aiosqlite, asyncio
        from aria.watchdog import check_audit_alerts
        from aria.hub.audit import AuditLogger

        async def setup():
            al = AuditLogger()
            await al.initialize(str(tmp_path / "audit.db"))
            for i in range(15):
                await al.log(event_type="test.error", source="hub", action="test", severity="error")
            await al.flush()
            await al.shutdown()

        asyncio.get_event_loop().run_until_complete(setup())
        result = check_audit_alerts(
            audit_db_path=str(tmp_path / "audit.db"),
            threshold=10,
            window_minutes=60,
        )
        assert result.level in ("WARNING", "CRITICAL")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_watchdog.py::TestAuditAlerts -v`
Expected: FAIL — `check_audit_alerts` not defined

**Step 3: Add `check_audit_alerts` to `aria/watchdog.py`**

```python
def check_audit_alerts(
    audit_db_path: str,
    threshold: int = 10,
    window_minutes: int = 5,
) -> WatchdogResult:
    """Check audit.db for recent error-severity events."""
    import sqlite3
    from datetime import timedelta

    if not os.path.exists(audit_db_path):
        return WatchdogResult("audit_alerts", "OK", "Audit DB not found — skipping")

    cutoff = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
    try:
        conn = sqlite3.connect(audit_db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM audit_events WHERE severity = 'error' AND timestamp >= ?",
            (cutoff,),
        )
        count = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        return WatchdogResult("audit_alerts", "WARNING", f"Failed to read audit.db: {e}")

    if count >= threshold:
        return WatchdogResult(
            "audit_alerts", "CRITICAL",
            f"{count} audit errors in last {window_minutes}min (threshold: {threshold})",
            details={"error_count": count, "window_minutes": window_minutes},
        )
    elif count > 0:
        return WatchdogResult(
            "audit_alerts", "OK",
            f"{count} audit errors in last {window_minutes}min (below threshold {threshold})",
        )
    return WatchdogResult("audit_alerts", "OK", "No recent audit errors")
```

Wire it into the existing watchdog `run_checks()` function.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_watchdog.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/watchdog.py tests/hub/test_watchdog.py
git commit -m "feat: add audit alerting bridge to watchdog — error threshold checking"
```

---

### Task 9: Integration Tests

**Files:**
- Create: `tests/integration/test_audit_pipeline.py`

**Step 1: Write integration tests**

```python
"""Integration tests — full audit pipeline: event → DB → query → verify."""

import asyncio
import json

import pytest

from aria.hub.audit import AuditLogger
from aria.hub.core import IntelligenceHub, Module


@pytest.fixture
async def full_hub(tmp_path):
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


class TestFullAuditChain:
    @pytest.mark.asyncio
    async def test_cache_write_creates_audit_event(self, full_hub):
        hub, audit = full_hub
        await hub.set_cache("test", {"data": 1})
        await audit.flush()
        events = await audit.query_events(event_type="cache.write")
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_module_register_creates_audit_event(self, full_hub):
        hub, audit = full_hub
        mod = Module("integration_test", hub)
        hub.register_module(mod)
        await asyncio.sleep(0.1)  # let create_task run
        await audit.flush()
        events = await audit.query_events(event_type="module.register")
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_integrity_check_after_writes(self, full_hub):
        hub, audit = full_hub
        for i in range(5):
            await hub.set_cache(f"cat_{i}", {"val": i})
        await audit.flush()
        result = await audit.verify_integrity()
        assert result["total"] >= 5
        assert result["invalid"] == 0

    @pytest.mark.asyncio
    async def test_request_id_correlation(self, full_hub):
        hub, audit = full_hub
        await audit.log_request("req-999", "GET", "/api/test", 200, 5.0, "127.0.0.1")
        await hub.emit_audit(
            event_type="test.correlated",
            source="hub",
            action="test",
            request_id="req-999",
        )
        await audit.flush()
        events = await audit.query_events(request_id="req-999")
        assert len(events) == 1
        reqs = await audit.query_requests()
        assert any(r["id"] == "req-999" for r in reqs)
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/integration/test_audit_pipeline.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/integration/test_audit_pipeline.py
git commit -m "test: add integration tests for full audit pipeline"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `docs/cli-reference.md` (add `aria audit` section)
- Modify: `docs/api-reference.md` (add `/api/audit/*` endpoints)
- Modify: `docs/architecture-detailed.md` (add audit logger to component list)

**Step 1: Add CLI reference section**

Append `aria audit` commands to `docs/cli-reference.md` with usage examples.

**Step 2: Add API reference section**

Append `/api/audit/*` endpoints to `docs/api-reference.md` with curl examples.

**Step 3: Update architecture doc**

Add AuditLogger to the hub component list in `docs/architecture-detailed.md`.

**Step 4: Commit**

```bash
git add docs/cli-reference.md docs/api-reference.md docs/architecture-detailed.md
git commit -m "docs: add audit logger to CLI reference, API reference, and architecture docs"
```

---

### Task 11: Full Test Suite Verification

**Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --timeout=120`
Expected: All tests pass (existing + new audit tests). Note baseline: 1576 passed, 16 skipped.

**Step 2: Verify no regressions**

Check that existing test count hasn't decreased. New tests should add ~60-80 tests.

**Step 3: Run ruff lint**

Run: `.venv/bin/python -m ruff check aria/hub/audit.py tests/hub/test_audit.py tests/hub/test_api_audit.py tests/hub/test_audit_middleware.py tests/integration/test_audit_pipeline.py`

Fix any lint issues.

**Step 4: Final commit if any fixes**

```bash
git add -A
git commit -m "chore: fix lint issues from audit logger implementation"
```

**Step 5: Use superpowers:finishing-a-development-branch**

Follow the finishing skill to present merge/PR options.
