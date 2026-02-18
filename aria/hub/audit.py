"""AuditLogger — write-behind audit event logging with integrity verification.

Provides structured audit logging for ARIA hub events, API requests,
startup snapshots, and entity curation changes. Events are buffered via
an asyncio.Queue and flushed in batches. Direct-write methods exist for
requests, startups, and curation changes.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
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
    checksum TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_source ON audit_events(source);
CREATE INDEX IF NOT EXISTS idx_events_subject ON audit_events(subject);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON audit_events(severity);
CREATE INDEX IF NOT EXISTS idx_events_request_id ON audit_events(request_id);

CREATE TABLE IF NOT EXISTS audit_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    client_ip TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_path ON audit_requests(path);
CREATE INDEX IF NOT EXISTS idx_requests_method ON audit_requests(method);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON audit_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_status ON audit_requests(status_code);

CREATE TABLE IF NOT EXISTS audit_startups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    python_version TEXT NOT NULL,
    modules_loaded TEXT,
    config_snapshot TEXT,
    system_memory_mb INTEGER,
    pid INTEGER,
    duration_ms REAL
);

CREATE TABLE IF NOT EXISTS audit_curation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    old_tier TEXT,
    new_tier TEXT,
    reason TEXT,
    changed_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_curation_entity ON audit_curation_history(entity_id);
CREATE INDEX IF NOT EXISTS idx_curation_timestamp ON audit_curation_history(timestamp);
"""


def _get_system_memory_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def _compute_checksum(
    timestamp: str,
    event_type: str,
    source: str,
    action: str,
    detail: Any = None,
) -> str:
    """SHA-256 of timestamp + event_type + source + action + detail_json_string."""
    detail_str = json.dumps(detail) if detail is not None else ""
    payload = timestamp + event_type + source + action + detail_str
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLogger:
    """Async audit logger with write-behind buffering and integrity verification."""

    def __init__(self, buffer_size: int = 10_000):
        self._buffer_size = buffer_size
        self._queue: asyncio.Queue | None = None
        self._db: aiosqlite.Connection | None = None
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._dropped_events = 0
        self._total_written = 0
        self._subscribers: list[asyncio.Queue] = []

    async def initialize(self, db_path: str) -> None:
        """Open database, create schema, start flush loop."""
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

        self._queue = asyncio.Queue(maxsize=self._buffer_size)
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    async def log(
        self,
        event_type: str,
        source: str,
        action: str,
        subject: str | None = None,
        detail: Any = None,
        request_id: str | None = None,
        severity: str = "info",
    ) -> None:
        """Buffer an audit event for batch insertion."""
        if self._queue is None:
            return
        timestamp = datetime.now(UTC).isoformat()
        checksum = _compute_checksum(timestamp, event_type, source, action, detail)
        detail_json = json.dumps(detail) if detail is not None else None

        row = (timestamp, event_type, source, action, subject, detail_json, request_id, severity, checksum)

        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self._dropped_events += 1
            logger.warning("Audit buffer full — dropped event: %s/%s", event_type, action)

    async def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: str | None = None,
        error: str | None = None,
    ) -> None:
        """Write API request directly (not buffered)."""
        if self._db is None:
            return
        timestamp = datetime.now(UTC).isoformat()
        await self._db.execute(
            "INSERT INTO audit_requests "
            "(timestamp, request_id, method, path, status_code, duration_ms, client_ip, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, request_id, method, path, status_code, duration_ms, client_ip, error),
        )
        await self._db.commit()

    async def log_startup(
        self,
        modules: dict | None = None,
        config_snapshot: dict | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Write startup snapshot directly. Auto-collects python_version, system_memory_mb, pid."""
        if self._db is None:
            return
        timestamp = datetime.now(UTC).isoformat()
        python_version = sys.version
        system_memory_mb = _get_system_memory_mb()
        pid = os.getpid()
        await self._db.execute(
            "INSERT INTO audit_startups "
            "(timestamp, python_version, modules_loaded, config_snapshot, "
            "system_memory_mb, pid, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                python_version,
                json.dumps(modules) if modules is not None else None,
                json.dumps(config_snapshot) if config_snapshot is not None else None,
                system_memory_mb,
                pid,
                duration_ms,
            ),
        )
        await self._db.commit()

    async def log_curation_change(
        self,
        entity_id: str,
        old_status: str | None = None,
        new_status: str | None = None,
        old_tier: str | None = None,
        new_tier: str | None = None,
        reason: str | None = None,
        changed_by: str = "auto",
    ) -> None:
        """Write entity curation change directly."""
        if self._db is None:
            return
        timestamp = datetime.now(UTC).isoformat()
        await self._db.execute(
            "INSERT INTO audit_curation_history "
            "(timestamp, entity_id, old_status, new_status, "
            "old_tier, new_tier, reason, changed_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, entity_id, old_status, new_status, old_tier, new_tier, reason, changed_by),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def add_subscriber(self, queue: asyncio.Queue):
        """Register a WebSocket subscriber queue."""
        self._subscribers.append(queue)

    def remove_subscriber(self, queue: asyncio.Queue):
        """Unregister a WebSocket subscriber queue."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

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
        """Filter and paginate audit_events."""
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

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM audit_events WHERE {where} ORDER BY timestamp ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_dict(row, json_fields=["detail"]) for row in rows]

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
        """Filter API requests."""
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

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM audit_requests WHERE {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def query_timeline(
        self,
        subject: str,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        """All events for a subject, chronological."""
        clauses = ["subject = ?"]
        params: list[Any] = [subject]

        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM audit_events WHERE {where} ORDER BY timestamp ASC"

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_dict(row, json_fields=["detail"]) for row in rows]

    async def get_stats(self, since: str | None = None) -> dict:
        """Aggregate counts by type and severity."""
        params: list[Any] = []
        where = "1=1"
        if since:
            where = "timestamp >= ?"
            params.append(since)

        by_type: dict[str, int] = {}
        async with self._db.execute(
            f"SELECT event_type, COUNT(*) FROM audit_events WHERE {where} GROUP BY event_type",
            params,
        ) as cursor:
            for row in await cursor.fetchall():
                by_type[row[0]] = row[1]

        by_severity: dict[str, int] = {}
        async with self._db.execute(
            f"SELECT severity, COUNT(*) FROM audit_events WHERE {where} GROUP BY severity",
            params,
        ) as cursor:
            for row in await cursor.fetchall():
                by_severity[row[0]] = row[1]

        return {"by_type": by_type, "by_severity": by_severity}

    async def query_startups(self, limit: int = 10) -> list[dict]:
        """Recent startup snapshots."""
        async with self._db.execute("SELECT * FROM audit_startups ORDER BY timestamp DESC LIMIT ?", (limit,)) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_dict(row, json_fields=["modules_loaded", "config_snapshot"]) for row in rows]

    async def query_curation(self, entity_id: str | None = None, limit: int = 50) -> list[dict]:
        """Entity curation history."""
        if entity_id:
            sql = "SELECT * FROM audit_curation_history WHERE entity_id = ? ORDER BY timestamp DESC LIMIT ?"
            params = (entity_id, limit)
        else:
            sql = "SELECT * FROM audit_curation_history ORDER BY timestamp DESC LIMIT ?"
            params = (limit,)

        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def verify_integrity(self, since: str | None = None) -> dict:
        """Recompute and verify checksums. Returns {total, valid, invalid, details}."""
        params: list[Any] = []
        where = "1=1"
        if since:
            where = "timestamp >= ?"
            params.append(since)

        async with self._db.execute(f"SELECT * FROM audit_events WHERE {where}", params) as cursor:
            rows = await cursor.fetchall()

        total = len(rows)
        valid = 0
        invalid = 0
        details: list[dict] = []

        for row in rows:
            d = self._row_to_dict(row, json_fields=["detail"])
            # Recompute checksum from raw fields
            detail_val = d.get("detail")
            expected = _compute_checksum(d["timestamp"], d["event_type"], d["source"], d["action"], detail_val)
            if expected == d["checksum"]:
                valid += 1
            else:
                invalid += 1
                details.append({"id": d["id"], "expected": expected, "actual": d["checksum"]})

        return {"total": total, "valid": valid, "invalid": invalid, "details": details}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def prune(self, retention_days: int) -> int:
        """Delete events and requests older than retention_days. Returns count deleted."""
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        total = 0

        for table in ("audit_events", "audit_requests", "audit_startups", "audit_curation_history"):
            cursor = await self._db.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
            total += cursor.rowcount
        await self._db.commit()
        return total

    async def export_archive(self, before_date: datetime, output_dir: str) -> list[str]:
        """Export events before before_date to JSONL files grouped by YYYY-MM."""
        before_iso = before_date.isoformat()
        async with self._db.execute(
            "SELECT * FROM audit_events WHERE timestamp < ? ORDER BY timestamp ASC",
            (before_iso,),
        ) as cursor:
            rows = await cursor.fetchall()

        # Group by YYYY-MM
        groups: dict[str, list[dict]] = {}
        for row in rows:
            d = self._row_to_dict(row, json_fields=["detail"])
            month_key = d["timestamp"][:7]  # YYYY-MM
            groups.setdefault(month_key, []).append(d)

        out_path = Path(output_dir)
        files: list[str] = []
        for month_key, events in groups.items():
            fpath = out_path / f"{month_key}.jsonl"
            with open(fpath, "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")
            files.append(str(fpath))

        return files

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Force drain buffer to DB."""
        items: list[tuple] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if items:
            await self._batch_insert(items)

    def get_buffer_stats(self) -> dict:
        """Return buffer statistics."""
        return {
            "queue_size": self._queue.qsize() if self._queue else 0,
            "buffer_capacity": self._buffer_size,
            "dropped_events": self._dropped_events,
            "total_written": self._total_written,
        }

    async def shutdown(self) -> None:
        """Stop flush loop, flush remaining, close connection."""
        self._running = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task

        if self._queue:
            await self.flush()

        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Background task: drain every 500ms or when 100 items accumulate."""
        while self._running:
            try:
                # Check every 50ms for the 100-item threshold
                for _ in range(10):
                    if self._queue.qsize() >= 100:
                        break
                    await asyncio.sleep(0.05)
                if self._queue.qsize() > 0:
                    await self.flush()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Audit flush loop error — will retry")

    async def _batch_insert(self, items: list[tuple]) -> None:
        """Batch INSERT in single transaction."""
        await self._db.executemany(
            "INSERT INTO audit_events "
            "(timestamp, event_type, source, action, subject, "
            "detail, request_id, severity, checksum) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            items,
        )
        await self._db.commit()
        self._total_written += len(items)

        # Notify WebSocket subscribers
        for row in items:
            event_dict = {
                "timestamp": row[0],
                "event_type": row[1],
                "source": row[2],
                "action": row[3],
                "subject": row[4],
                "severity": row[7],
            }
            for sub in list(self._subscribers):  # snapshot — safe against concurrent add/remove
                with contextlib.suppress(asyncio.QueueFull):
                    sub.put_nowait(event_dict)

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row, json_fields: list[str] | None = None) -> dict:
        """Convert aiosqlite Row to dict, parse JSON fields."""
        d = dict(row)
        if json_fields:
            for field in json_fields:
                if field in d and d[field] is not None:
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        d[field] = json.loads(d[field])
        return d
