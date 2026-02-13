# Deployment Migration Guide: ha-intelligence + ha-intelligence-hub → ARIA

Migration guide for systemd services and timers from the old two-repo setup to the unified ARIA CLI.

## Current Service/Timer Inventory

### Long-running Service

| Unit | ExecStart | Schedule | Memory |
|------|-----------|----------|--------|
| `ha-intelligence-hub.service` | `bin/ha-hub.py --port 8001` | always-on | 1G max |

### Oneshot Timer Services (8 timers)

| Unit | ExecStart | Schedule | Memory |
|------|-----------|----------|--------|
| `ha-intelligence-snapshot` | `ha-intelligence --snapshot` | Daily 23:00 | 2G max |
| `ha-intelligence-full` | `ha-intelligence --full` | Daily 23:30 | 4G max |
| `ha-intelligence-drift` | `ha-intelligence --check-drift` | Daily 02:00 | 2G max |
| `ha-intelligence-intraday` | `ha-intelligence --snapshot-intraday` | Every 4h (00,04,08,12,16,20) | 2G max |
| `ha-intelligence-correlations` | `ha-intelligence --entity-correlations && ha-intelligence --suggest-automations` | Sun 03:15 | 4G max |
| `ha-intelligence-sequences` | `ha-intelligence --train-sequences && ha-intelligence --sequence-anomalies` | Sun 03:45 | 4G max |
| `ha-intelligence-meta-learn` | `ha-intelligence --meta-learn` | Mon 01:30 | 4G max |
| `ha-intelligence-prophet` | `ha-intelligence --train-prophet` | Mon 03:00 | 4G max |

### Key Observation

All timer services call `/home/justin/.local/bin/ha-intelligence` which is a symlink to the **old separate repo** at `/home/justin/Documents/projects/ha-intelligence/bin/ha-intelligence`. The hub service calls `bin/ha-hub.py` directly (now a thin wrapper over `aria serve`). After migration, everything goes through the unified `aria` CLI.

## Command Migration Table

| Old Command | New ARIA Command | Notes |
|-------------|-----------------|-------|
| `bin/ha-hub.py --port 8001` | `aria serve --port 8001` | `bin/ha-hub.py` already wraps this |
| `ha-intelligence --snapshot` | `aria snapshot` | |
| `ha-intelligence --full` | `aria full` | |
| `ha-intelligence --check-drift` | `aria check-drift` | |
| `ha-intelligence --snapshot-intraday` | `aria snapshot-intraday` | |
| `ha-intelligence --entity-correlations` | `aria correlations` | Name shortened |
| `ha-intelligence --suggest-automations` | `aria suggest-automations` | Requires Ollama |
| `ha-intelligence --train-sequences` | `aria sequences train` | Now a subcommand |
| `ha-intelligence --sequence-anomalies` | `aria sequences detect` | Now a subcommand |
| `ha-intelligence --meta-learn` | `aria meta-learn` | Requires Ollama |
| `ha-intelligence --train-prophet` | `aria prophet` | Name shortened |
| `ha-intelligence --retrain` | `aria retrain` | Not in current timers |
| `ha-intelligence --score` | `aria score` | Not in current timers |
| `ha-intelligence --predict` | `aria predict` | Not in current timers |
| `ha-intelligence --occupancy` | `aria occupancy` | Not in current timers |
| `ha-intelligence --power-profiles` | `aria power-profiles` | Not in current timers |

## New Service File Templates

### aria.service (replaces ha-intelligence-hub.service)

```ini
[Unit]
Description=ARIA — Adaptive Residence Intelligence Architecture
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria serve --port 8001'
Restart=on-failure
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=300
TimeoutStopSec=15
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=1G
CPUQuota=50%

[Install]
WantedBy=default.target
```

### aria-snapshot.service + timer (replaces ha-intelligence-snapshot)

```ini
# aria-snapshot.service
[Unit]
Description=ARIA — daily snapshot
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria snapshot'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=1G
MemoryMax=2G
```

```ini
# aria-snapshot.timer
[Unit]
Description=ARIA — daily snapshot at 23:00

[Timer]
OnCalendar=*-*-* 23:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-full.service + timer (replaces ha-intelligence-full)

```ini
# aria-full.service
[Unit]
Description=ARIA — daily full analysis
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria full'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=3G
MemoryMax=4G
```

```ini
# aria-full.timer
[Unit]
Description=ARIA — daily full analysis at 23:30

[Timer]
OnCalendar=*-*-* 23:30:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-drift.service + timer (replaces ha-intelligence-drift)

```ini
# aria-drift.service
[Unit]
Description=ARIA — drift check
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria check-drift'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=1G
MemoryMax=2G
```

```ini
# aria-drift.timer
[Unit]
Description=ARIA — daily drift check at 02:00

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-intraday.service + timer (replaces ha-intelligence-intraday)

```ini
# aria-intraday.service
[Unit]
Description=ARIA — intraday snapshot
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria snapshot-intraday'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=1G
MemoryMax=2G
```

```ini
# aria-intraday.timer
[Unit]
Description=ARIA — intraday snapshot every 4h

[Timer]
OnCalendar=*-*-* 00,04,08,12,16,20:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-correlations.service + timer (replaces ha-intelligence-correlations)

```ini
# aria-correlations.service
[Unit]
Description=ARIA — weekly entity correlations + automation suggestions
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria correlations && /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria suggest-automations'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=3G
MemoryMax=4G
```

```ini
# aria-correlations.timer
[Unit]
Description=ARIA — weekly correlations Sun 03:15

[Timer]
OnCalendar=Sun *-*-* 03:15:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-sequences.service + timer (replaces ha-intelligence-sequences)

```ini
# aria-sequences.service
[Unit]
Description=ARIA — weekly sequence training + anomaly detection
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria sequences train && /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria sequences detect'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=3G
MemoryMax=4G
```

```ini
# aria-sequences.timer
[Unit]
Description=ARIA — weekly sequences Sun 03:45

[Timer]
OnCalendar=Sun *-*-* 03:45:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-meta-learn.service + timer (replaces ha-intelligence-meta-learn)

```ini
# aria-meta-learn.service
[Unit]
Description=ARIA — weekly meta-learning
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria meta-learn'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=3G
MemoryMax=4G
```

```ini
# aria-meta-learn.timer
[Unit]
Description=ARIA — weekly meta-learning Mon 01:30

[Timer]
OnCalendar=Mon *-*-* 01:30:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

### aria-prophet.service + timer (replaces ha-intelligence-prophet)

```ini
# aria-prophet.service
[Unit]
Description=ARIA — weekly Prophet model training
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria prophet'
StandardOutput=append:/home/justin/.local/log/aria.log
StandardError=append:/home/justin/.local/log/aria.log
MemoryHigh=3G
MemoryMax=4G
```

```ini
# aria-prophet.timer
[Unit]
Description=ARIA — weekly Prophet training Mon 03:00

[Timer]
OnCalendar=Mon *-*-* 03:00:00
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
```

## Step-by-Step Migration Procedure

### Prerequisites

```bash
# Verify aria CLI is installed and working
cd /home/justin/Documents/projects/ha-intelligence-hub
.venv/bin/aria --help

# If not installed yet:
.venv/bin/pip install -e .
```

### Phase 1: Install ARIA CLI (no service disruption)

```bash
# Install the package in editable mode
cd /home/justin/Documents/projects/ha-intelligence-hub
.venv/bin/pip install -e .

# Verify entry point works
.venv/bin/aria --help

# Smoke-test a read-only command
. ~/.env && .venv/bin/aria check-drift
```

### Phase 2: Create new service files

```bash
# Create log directory
mkdir -p ~/.local/log

# Copy new service/timer files to systemd user directory
# (Templates above — save each to ~/.config/systemd/user/)

# Reload systemd to pick up new files
systemctl --user daemon-reload
```

### Phase 3: Migrate hub service (the long-running service)

```bash
# Stop old hub
systemctl --user stop ha-intelligence-hub.service
systemctl --user disable ha-intelligence-hub.service

# Enable and start new ARIA service
systemctl --user enable aria.service
systemctl --user start aria.service

# Verify
systemctl --user status aria.service
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```

### Phase 4: Migrate timer services (one at a time)

Migrate in order of risk — start with the least critical timer, verify it runs, then proceed.

**Recommended order:**
1. `ha-intelligence-drift` → `aria-drift` (daily, low-impact)
2. `ha-intelligence-intraday` → `aria-intraday` (every 4h, runs frequently — fast feedback)
3. `ha-intelligence-snapshot` → `aria-snapshot` (daily, data collection)
4. `ha-intelligence-full` → `aria-full` (daily, main pipeline)
5. `ha-intelligence-correlations` → `aria-correlations` (weekly)
6. `ha-intelligence-sequences` → `aria-sequences` (weekly)
7. `ha-intelligence-prophet` → `aria-prophet` (weekly)
8. `ha-intelligence-meta-learn` → `aria-meta-learn` (weekly, Ollama-dependent)

**For each timer:**

```bash
# Example: migrating drift timer
# 1. Stop and disable old timer
systemctl --user stop ha-intelligence-drift.timer
systemctl --user disable ha-intelligence-drift.timer

# 2. Enable and start new timer
systemctl --user enable aria-drift.timer
systemctl --user start aria-drift.timer

# 3. Verify timer is scheduled
systemctl --user list-timers | grep aria-drift

# 4. (Optional) Test the service manually
systemctl --user start aria-drift.service
journalctl --user -u aria-drift.service -n 20
```

### Phase 5: Clean up old units

After all timers are migrated and verified (wait at least one full cycle for each):

```bash
# Remove old service/timer files
rm ~/.config/systemd/user/ha-intelligence-hub.service
rm ~/.config/systemd/user/ha-intelligence-full.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-snapshot.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-drift.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-intraday.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-correlations.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-sequences.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-meta-learn.{service,timer}
rm ~/.config/systemd/user/ha-intelligence-prophet.{service,timer}

# Reload systemd
systemctl --user daemon-reload

# Verify no old units remain
systemctl --user list-units 'ha-intelligence*'
systemctl --user list-timers 'ha-intelligence*'
```

### Phase 6: Update symlink

```bash
# Replace ha-intelligence symlink with aria
ln -sf /home/justin/Documents/projects/ha-intelligence-hub/.venv/bin/aria \
       /home/justin/.local/bin/aria

# Optional: keep ha-intelligence as compat alias
ln -sf /home/justin/.local/bin/aria /home/justin/.local/bin/ha-intelligence
```

## Rollback Plan

### Quick rollback (any phase)

The old `ha-intelligence` symlink and service files remain untouched until Phase 5. To roll back at any point before cleanup:

```bash
# Stop new ARIA service/timers
systemctl --user stop aria.service
systemctl --user stop aria-*.timer

# Re-enable old services
systemctl --user enable ha-intelligence-hub.service
systemctl --user start ha-intelligence-hub.service

# Re-enable old timers (example for one)
systemctl --user enable ha-intelligence-drift.timer
systemctl --user start ha-intelligence-drift.timer
# ... repeat for each timer
```

### Full rollback script

Save as `rollback-to-ha-intelligence.sh`:

```bash
#!/bin/bash
set -e

echo "Rolling back ARIA to ha-intelligence services..."

# Stop all ARIA units
systemctl --user stop aria.service 2>/dev/null || true
for timer in drift intraday snapshot full correlations sequences meta-learn prophet; do
    systemctl --user stop "aria-${timer}.timer" 2>/dev/null || true
    systemctl --user disable "aria-${timer}.timer" 2>/dev/null || true
done
systemctl --user disable aria.service 2>/dev/null || true

# Re-enable old units
systemctl --user enable ha-intelligence-hub.service
systemctl --user start ha-intelligence-hub.service

for timer in drift intraday snapshot full correlations sequences meta-learn prophet; do
    systemctl --user enable "ha-intelligence-${timer}.timer"
    systemctl --user start "ha-intelligence-${timer}.timer"
done

systemctl --user daemon-reload

echo "Rollback complete. Verify:"
echo "  systemctl --user status ha-intelligence-hub.service"
echo "  systemctl --user list-timers 'ha-intelligence*'"
```

### After Phase 5 (cleanup done)

If old files have been deleted, restore from git or recreate manually. The old `ha-intelligence` repo at `~/Documents/projects/ha-intelligence/` is still intact — the symlink just needs re-pointing:

```bash
# Restore symlink to old engine
ln -sf /home/justin/Documents/projects/ha-intelligence/bin/ha-intelligence \
       /home/justin/.local/bin/ha-intelligence
```

## Data Continuity Notes

- **Cache DB** (`~/ha-logs/intelligence/cache/hub.db`): Unchanged. Both old hub and ARIA read/write the same file.
- **Engine output files** (`~/ha-logs/intelligence/`): Unchanged. ARIA engine writes to the same paths.
- **Log files**: New services write to `~/.local/log/aria.log` instead of `~/.local/log/ha-intelligence.log`. Both old and new logs coexist during migration.
- **Snapshot log** (`~/ha-logs/intelligence/snapshot_log.jsonl`): Append-only, no migration needed.
- **ML models** (`~/ha-logs/intelligence/models/`): Same directory, same format.

## Affected Systems (Cross-Project)

These external systems reference `ha-intelligence` and may need updates after migration:

| System | What references it | Action needed |
|--------|--------------------|---------------|
| `~/CLAUDE.md` | `ha-intelligence` in service list, Ollama schedule | Update descriptions |
| `~/Documents/CLAUDE.md` | Ollama task schedule references `ha-intelligence` | Update command names |
| `~/Documents/projects/CLAUDE.md` | Cross-project dependency table | Update project name |
| `telegram-brief` | May reference ha-intelligence in digest generation | Verify no hardcoded names |
| `ha-log-sync` | Separate service, writes to `~/ha-logs/logbook/` — no change needed | None |

## Verification Checklist

After full migration:

- [ ] `aria serve` starts and `/health` returns OK
- [ ] Dashboard loads at `http://127.0.0.1:8001/ui/`
- [ ] WebSocket connects at `ws://127.0.0.1:8001/ws`
- [ ] `systemctl --user list-timers 'aria-*'` shows all 8 timers
- [ ] `aria full` completes manually without error
- [ ] `aria check-drift` completes manually without error
- [ ] `aria correlations` completes manually without error
- [ ] `aria sequences train` completes manually without error
- [ ] No old `ha-intelligence-*` timers remain active
- [ ] Log output appears in `~/.local/log/aria.log`
- [ ] Ollama schedule deconfliction still holds (no overlap within 45 min)
