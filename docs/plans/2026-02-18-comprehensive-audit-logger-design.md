# Comprehensive Audit Logger — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full-spectrum audit logging system for ARIA that captures all significant events — module lifecycle, API calls, config changes, cache writes, module decisions — in a queryable, exportable, tamper-evident format with CLI access, real-time streaming, and alerting integration.

**Architecture:** Dedicated SQLite database (`audit.db`) separate from `hub.db`, with an `AuditLogger` hub service providing both automatic middleware capture and explicit module API. Configurable retention (default 90 days) with archival export. FastAPI middleware for request tracing, WebSocket streaming for real-time dashboard, CLI commands for terminal investigation, and watchdog integration for alert escalation.

**Tech Stack:** Python asyncio, aiosqlite, FastAPI middleware, WebSocket, existing CLI framework

---

## 1. Storage Layer

### 1.1 Database Location

`~/ha-logs/intelligence/cache/audit.db` — same directory as `hub.db` for backup consistency.

### 1.2 Schema

```sql
-- Core audit events
CREATE TABLE audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO 8601
    event_type TEXT NOT NULL,         -- e.g., cache.write, config.change, module.lifecycle
    source TEXT NOT NULL,             -- module ID, "hub", "api", "system"
    action TEXT NOT NULL,             -- set, promote, reject, initialize, shutdown, attempt
    subject TEXT,                     -- what was acted on: cache category, config key, entity ID
    detail TEXT,                      -- JSON blob with full context
    request_id TEXT,                  -- correlation ID (nullable, links to requests)
    severity TEXT NOT NULL DEFAULT 'info',  -- info, warning, error
    checksum TEXT                     -- SHA-256 of (timestamp + event_type + source + action + detail)
);

CREATE INDEX idx_audit_events_timestamp ON audit_events(timestamp);
CREATE INDEX idx_audit_events_type ON audit_events(event_type);
CREATE INDEX idx_audit_events_source ON audit_events(source);
CREATE INDEX idx_audit_events_subject ON audit_events(subject);
CREATE INDEX idx_audit_events_request_id ON audit_events(request_id);
CREATE INDEX idx_audit_events_severity ON audit_events(severity);

-- API request log
CREATE TABLE audit_requests (
    id TEXT PRIMARY KEY,              -- UUID request ID
    timestamp TEXT NOT NULL,          -- ISO 8601
    method TEXT NOT NULL,             -- GET, PUT, POST, DELETE
    path TEXT NOT NULL,               -- /api/config/shadow.exploration_rate
    status_code INTEGER,              -- HTTP response code
    duration_ms REAL,                 -- request latency
    client_ip TEXT,                   -- requester IP
    error TEXT                        -- error message if 4xx/5xx
);

CREATE INDEX idx_audit_requests_timestamp ON audit_requests(timestamp);
CREATE INDEX idx_audit_requests_path ON audit_requests(path);
CREATE INDEX idx_audit_requests_status ON audit_requests(status_code);

-- Hub startup snapshots
CREATE TABLE audit_startups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    python_version TEXT NOT NULL,
    modules_loaded TEXT NOT NULL,     -- JSON list of module IDs + status
    config_snapshot TEXT NOT NULL,    -- JSON of all config values at startup
    system_memory_mb INTEGER,
    pid INTEGER,
    duration_ms REAL                  -- startup time
);

-- Entity curation history (supplements hub.db entity_curation single-row table)
CREATE TABLE audit_curation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL,
    old_tier INTEGER,
    new_tier INTEGER NOT NULL,
    reason TEXT,
    changed_by TEXT                   -- "auto", "user", module ID
);

CREATE INDEX idx_audit_curation_entity ON audit_curation_history(entity_id);
CREATE INDEX idx_audit_curation_timestamp ON audit_curation_history(timestamp);
```

### 1.3 Data Integrity

Each `audit_events` row includes a `checksum` column: `SHA-256(timestamp + event_type + source + action + detail)`. This enables detection of after-the-fact modification. Not cryptographically tamper-proof (no chain), but sufficient for accountability auditing.

### 1.4 Retention & Archival

- **Active retention:** Configurable via `audit.retention_days` (default 90)
- **Pruning:** Daily scheduled task alongside existing event pruner
- **Archival:** Before pruning, expired records are exported to JSONL files at `~/ha-logs/intelligence/audit-archive/YYYY-MM.jsonl` (one file per month)
- **Archive rotation:** Archive files are never auto-deleted (disk-managed by user)

---

## 2. AuditLogger Class

### 2.1 Core API

```python
class AuditLogger:
    """Centralized audit logging for ARIA hub."""

    async def initialize(self, db_path: str) -> None:
        """Open audit.db, create tables, start write buffer."""

    async def shutdown(self) -> None:
        """Flush buffer, close connection."""

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
        """Log an API request."""

    async def log_startup(
        self,
        modules: dict[str, str],
        config_snapshot: dict,
        duration_ms: float,
    ) -> None:
        """Log hub startup context."""

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

    # Query methods
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
    ) -> list[dict]: ...

    async def query_requests(
        self,
        path: str | None = None,
        method: str | None = None,
        status_min: int | None = None,
        status_max: int | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict]: ...

    async def query_timeline(
        self,
        subject: str,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]: ...

    async def get_stats(
        self,
        since: str | None = None,
    ) -> dict: ...

    async def prune(self, retention_days: int) -> int: ...

    async def export_archive(self, before_date: str, output_dir: str) -> str: ...

    async def verify_integrity(self, since: str | None = None) -> dict: ...
```

### 2.2 Write-Behind Buffer

To avoid blocking the hot path (cache writes happen frequently), the AuditLogger uses an async write-behind buffer:

1. `log()` appends to an in-memory `asyncio.Queue` (bounded, 10,000 items)
2. A background task drains the queue every 500ms or when 100 items accumulate (whichever is first)
3. Batch INSERT within a single transaction
4. If the queue is full, events are dropped with a WARNING log (counter tracked in stats)

This keeps the audit overhead under 1ms per event for callers.

---

## 3. Middleware Layer (Automatic Capture)

### 3.1 Hub Method Wrappers

Wrap existing hub methods to emit audit events automatically:

| Method | Event Type | Captured Data |
|--------|-----------|---------------|
| `hub.set_cache(category, data)` | `cache.write` | category, data size, version |
| `hub.cache.set_config(key, value, changed_by)` | `config.change` | key, old_value, new_value, changed_by |
| `hub.cache.reset_config(key, changed_by)` | `config.reset` | key, old_value, default_value, changed_by |
| `hub.register_module(module)` | `module.register` | module_id, class name |
| `hub.mark_module_running(name)` | `module.running` | module_id |
| `hub.mark_module_failed(name)` | `module.failed` | module_id, error |
| `hub.shutdown()` | `system.shutdown` | uptime, module count |
| `hub.publish(event_type, data)` | `hub.event` | event_type, subscriber count |

Implementation: Monkey-patch or decorator pattern on hub methods during `AuditLogger.initialize()`. Hub passes `self.audit` reference to AuditLogger.

### 3.2 FastAPI Request Middleware

```python
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    start = time.monotonic()

    response = await call_next(request)

    duration_ms = (time.monotonic() - start) * 1000
    await audit.log_request(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        client_ip=request.client.host,
        error=None if response.status_code < 400 else "see audit_events",
    )
    response.headers["X-Request-ID"] = request_id
    return response
```

The `request_id` is available to route handlers via `request.state.request_id` for correlation with audit events.

### 3.3 Module Registration Collision Detection

When `hub.register_module()` is called with a module_id that already exists:

```python
if module_id in self.modules:
    await self.audit.log(
        event_type="module.collision",
        source="hub",
        action="attempt",
        subject=module_id,
        detail={
            "existing_class": type(self.modules[module_id]).__name__,
            "new_class": type(module).__name__,
        },
        severity="warning",
    )
```

This captures exactly the bug from issue #8.

### 3.4 Failed Attempt Logging

Config validation failures, invalid API parameters, and module init exceptions all emit `*.attempt` events with `severity="warning"` or `severity="error"`:

- `config.attempt` — validation failed (key, attempted_value, error)
- `module.attempt` — init failed (module_id, exception)
- `api.attempt` — bad request (path, status_code, error)

---

## 4. Explicit Module API

### 4.1 Hub Integration

```python
# In IntelligenceHub
async def audit(
    self,
    event_type: str,
    detail: dict,
    subject: str | None = None,
    severity: str = "info",
) -> None:
    """Emit a custom audit event from a module."""
    if self._audit_logger and self._audit_enabled:
        # source is auto-detected from calling module via hub context
        await self._audit_logger.log(
            event_type=event_type,
            source=self._current_module_id or "hub",
            action="decision",
            subject=subject,
            detail=detail,
            severity=severity,
        )
```

### 4.2 Module Usage Examples

```python
# Transfer engine: candidate promoted
await self.hub.audit(
    "transfer.promote",
    subject=candidate.id,
    detail={"from_area": candidate.source_area, "hit_rate": candidate.hit_rate},
)

# Shadow engine: exploration chosen
await self.hub.audit(
    "shadow.explore",
    subject=prediction_id,
    detail={"context": context, "confidence": confidence},
)

# Discovery: entity archived
await self.hub.audit(
    "discovery.archive",
    subject=entity_id,
    detail={"reason": "stale_72h", "last_seen": last_seen},
)
```

### 4.3 Event Chain Correlation

For tracing through module chains (e.g., API request → discovery → transfer generation → promotion):

1. API middleware sets `request_id` on the request
2. Hub publishes events with a `correlation_id` (either the request_id or a generated chain ID)
3. When a module handles an event and triggers further actions, it passes the correlation_id through
4. Module base class stores `self._current_correlation_id` during event handling

Query with: `GET /api/audit/events?request_id=<uuid>` returns the full chain.

---

## 5. Query API

### 5.1 REST Endpoints

```
GET /api/audit/events
    ?type=cache.write
    &source=transfer_engine
    &subject=shadow.exploration_rate
    &severity=error
    &since=2026-02-17T00:00:00
    &until=2026-02-18T00:00:00
    &request_id=<uuid>
    &limit=100
    &offset=0

GET /api/audit/requests
    ?path=/api/config
    &method=PUT
    &status_min=400
    &since=1h
    &limit=50

GET /api/audit/timeline/{subject}
    ?since=7d
    ?until=now

GET /api/audit/stats
    ?since=24h

GET /api/audit/startups
    ?limit=10

GET /api/audit/curation/{entity_id}
    ?limit=50

GET /api/audit/integrity
    ?since=7d

POST /api/audit/export
    {before_date: "2026-01-01", format: "jsonl"}
```

### 5.2 Convenience Parsers

`since` and `until` accept:
- ISO 8601: `2026-02-17T12:00:00`
- Relative: `1h`, `24h`, `7d`, `30d`
- Named: `today`, `yesterday`

### 5.3 WebSocket Real-Time Streaming

New WebSocket channel for live audit events:

```
ws://127.0.0.1:8001/ws/audit
    ?types=config.change,module.lifecycle,*.error
    &severity_min=warning
```

Clients receive audit events as they happen, filtered by type and severity. Supports glob patterns (`cache.*`, `*.error`).

---

## 6. CLI Commands

### 6.1 Command Structure

```bash
# Query events
aria audit events [--type TYPE] [--source SOURCE] [--subject SUBJECT]
                  [--severity SEVERITY] [--since SINCE] [--until UNTIL]
                  [--request-id UUID] [--limit N] [--json]

# Query requests
aria audit requests [--path PATH] [--method METHOD] [--status STATUS]
                    [--since SINCE] [--limit N] [--json]

# Timeline for a subject
aria audit timeline SUBJECT [--since SINCE] [--json]

# Stats summary
aria audit stats [--since SINCE]

# Recent startups
aria audit startups [--limit N]

# Entity curation history
aria audit curation ENTITY_ID [--limit N]

# Integrity check
aria audit verify [--since SINCE]

# Export archive
aria audit export --before DATE [--output DIR]

# Live tail (like journalctl -f)
aria audit tail [--types TYPES] [--severity-min SEVERITY]
```

### 6.2 Output Format

Default: human-readable table format (like `git log --oneline`)

```
$ aria audit events --type config.change --since 2d
2026-02-18 06:50:01 | config.change | user    | set    | shadow.exploration_rate | 0.15 → 0.20
2026-02-18 05:30:00 | config.change | system  | set    | activity.window_minutes | 15 → 30
2026-02-17 22:00:00 | config.reset  | user    | reset  | shadow.exploration_rate | 0.20 → 0.15 (default)
```

With `--json`: newline-delimited JSON objects.

### 6.3 Live Tail

```
$ aria audit tail --severity-min warning
[06:50:01] WARNING module.collision hub: pattern_recognition already registered (PatternRecognition vs PatternRecognitionModule)
[06:50:05] ERROR   module.failed   hub: presence module timeout after 30s
[06:51:00] WARNING cache.write     activity_monitor: activity_summary data size 2.3MB exceeds 1MB threshold
```

Implementation: Connects to `/ws/audit` WebSocket endpoint, renders events in real-time.

---

## 7. Alerting Integration

### 7.1 Audit → Watchdog Bridge

The watchdog already runs every 5 minutes and sends Telegram alerts. Extend it to check audit events:

```python
# In watchdog health check
async def check_audit_alerts(self):
    """Check audit for alertable conditions."""
    recent = await audit.query_events(
        severity="error",
        since="5m",  # since last check
    )
    if len(recent) > threshold:
        await self.send_telegram_alert(
            f"ARIA Audit: {len(recent)} errors in last 5 minutes"
        )
```

### 7.2 Alertable Conditions

| Condition | Severity | Alert Method |
|-----------|----------|-------------|
| Module failed to initialize | error | Telegram (immediate) |
| Module registration collision | warning | Audit log only |
| >10 API 5xx errors in 5min | error | Telegram (immediate) |
| Config changed by unknown source | warning | Watchdog summary |
| Integrity check failure | error | Telegram (immediate) |
| Audit write buffer overflow (dropped events) | error | Telegram + Python logger |
| >100 failed config attempts in 1h | warning | Watchdog summary |

### 7.3 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `audit.alert_on_errors` | true | Send Telegram for error-severity audit events |
| `audit.alert_threshold` | 10 | Number of errors in window before alerting |
| `audit.alert_window_minutes` | 5 | Window for error counting |

---

## 8. Hub Startup Context

### 8.1 Startup Snapshot

On hub startup, after all modules are initialized:

```python
await audit.log_startup(
    modules=hub.module_status,  # {module_id: "running"|"failed"}
    config_snapshot=await hub.cache.get_all_config(),
    duration_ms=startup_duration,
)
```

Captures:
- Python version (`sys.version`)
- All module IDs and their status
- Complete config snapshot (all 76+ parameters)
- System memory (`psutil.virtual_memory()` or `/proc/meminfo`)
- Process PID
- Startup duration

### 8.2 Shutdown Event

```python
await audit.log(
    event_type="system.shutdown",
    source="hub",
    action="shutdown",
    detail={
        "uptime_seconds": time.monotonic() - start_time,
        "modules_running": running_count,
        "reason": "user_request" | "signal" | "error",
    },
)
```

---

## 9. Entity Curation History

### 9.1 Change Capture

Wrap `cache.set_curation()` (or wherever entity curation is updated) to emit curation history:

```python
# Before update, read current state
old = await cache.get_curation(entity_id)

# After update
await audit.log_curation_change(
    entity_id=entity_id,
    old_status=old["status"] if old else None,
    new_status=new_status,
    old_tier=old["tier"] if old else None,
    new_tier=new_tier,
    reason=reason,
    changed_by=changed_by,
)
```

### 9.2 Query

`GET /api/audit/curation/{entity_id}` returns the full change history for an entity.

`aria audit curation sensor.living_room_temperature` from CLI.

---

## 10. Performance Guardrails

### 10.1 Write-Behind Buffer

- **Queue size:** 10,000 events (bounded asyncio.Queue)
- **Flush interval:** 500ms or 100 events (whichever is first)
- **Batch insert:** Single transaction per flush
- **Overflow behavior:** Drop oldest, increment `dropped_events` counter, log WARNING

### 10.2 Expected Volume

| Event Source | Estimated Rate | Daily Volume |
|-------------|---------------|-------------|
| Cache writes | ~100/hour | ~2,400 |
| Config changes | ~5/day | ~5 |
| API requests | ~200/hour | ~4,800 |
| Module lifecycle | ~20/startup | ~20 |
| Module decisions | ~50/hour | ~1,200 |
| **Total** | | **~8,400/day** |

At ~500 bytes/event average: **~4MB/day**, **~360MB over 90 days**. Well within reasonable limits.

### 10.3 DB Maintenance

- WAL mode for concurrent reads during writes
- VACUUM after monthly archive export
- Connection pool size: 1 writer + 3 readers

---

## 11. Config Parameters

| Key | Default | Type | Description |
|-----|---------|------|-------------|
| `audit.enabled` | true | boolean | Master switch for audit logging |
| `audit.retention_days` | 90 | number | Days to retain audit records |
| `audit.log_api_requests` | true | boolean | Log API requests |
| `audit.log_cache_writes` | true | boolean | Log cache writes |
| `audit.buffer_size` | 10000 | number | Write-behind buffer capacity |
| `audit.flush_interval_ms` | 500 | number | Maximum time between DB flushes |
| `audit.alert_on_errors` | true | boolean | Telegram alerts for error-severity events |
| `audit.alert_threshold` | 10 | number | Error count threshold for alerting |
| `audit.alert_window_minutes` | 5 | number | Window for error counting |
| `audit.archive_on_prune` | true | boolean | Export to JSONL before pruning |

---

## 12. Components & Files

| File | Purpose | New/Modify |
|------|---------|-----------|
| `aria/hub/audit.py` | AuditLogger class, DB init, write/query/export | **New** |
| `aria/hub/core.py` | Wire audit into hub, `hub.audit()` method, middleware hooks | Modify |
| `aria/hub/api.py` | Request middleware, `/api/audit/*` routes, WebSocket `/ws/audit` | Modify |
| `aria/hub/cache.py` | Curation change hooks | Modify |
| `aria/hub/config_defaults.py` | Audit config entries (10 parameters) | Modify |
| `aria/cli.py` | `aria audit` subcommands, startup snapshot | Modify |
| `aria/watchdog.py` | Audit alert bridge | Modify |
| `tests/hub/test_audit.py` | Unit tests for AuditLogger | **New** |
| `tests/hub/test_api_audit.py` | API endpoint tests | **New** |
| `tests/hub/test_audit_middleware.py` | Middleware integration tests | **New** |
| `tests/integration/test_audit_pipeline.py` | End-to-end audit flow | **New** |

---

## 13. Testing Strategy

### Unit Tests (test_audit.py)
- DB creation and schema
- Event logging and querying (all filter combinations)
- Request logging and querying
- Startup snapshot logging
- Curation history logging
- Write-behind buffer behavior (flush timing, overflow)
- Integrity checksum generation and verification
- Retention pruning with archive export
- Stats aggregation

### API Tests (test_api_audit.py)
- All `/api/audit/*` endpoints with various query parameters
- Relative time parsing (`1h`, `7d`, `today`)
- Pagination (limit/offset)
- Error responses for invalid parameters

### Middleware Tests (test_audit_middleware.py)
- Request middleware captures method, path, status, duration
- Request ID header propagation
- Cache write interception
- Config change interception
- Module registration collision detection
- Failed attempt logging

### Integration Tests (test_audit_pipeline.py)
- Full chain: API request → cache write → module event → audit query
- Request ID correlation across event chain
- WebSocket audit streaming
- CLI command output parsing
- Startup snapshot accuracy
- Archive export and integrity verification

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Audit writes slow down hot path | Write-behind buffer (async queue) |
| audit.db grows too large | Configurable retention + archive export |
| Module developers forget to audit | Middleware captures 80% automatically |
| Circular audit (auditing the auditor) | AuditLogger never audits its own writes |
| DB corruption | WAL mode + integrity checksums |
| Buffer overflow during burst | Bounded queue, drop + warn, counter in stats |
