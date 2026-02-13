# Deployment Guide (Pre-ARIA)

> **SUPERSEDED 2026-02-13.** The hub is now deployed as `aria-hub.service` via `systemd/install.sh`. See `CLAUDE.md` for current deployment instructions.

Original deployment guide for HA Intelligence Hub using systemd and optional Tailscale Serve for remote access.

## Prerequisites

- Ubuntu/Linux system with systemd
- Python 3.12+ installed
- Home Assistant accessible via network
- Environment variables configured in `~/.env`

## Systemd Service Setup

### 1. Create Service File

Create `~/.config/systemd/user/ha-intelligence-hub.service`:

```ini
[Unit]
Description=HA Intelligence Hub - Adaptive intelligence for Home Assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/justin/Documents/projects/ha-intelligence-hub-phase2
ExecStart=/bin/bash -c '. ~/.env && exec /home/justin/Documents/projects/ha-intelligence-hub-phase2/venv/bin/python /home/justin/Documents/projects/ha-intelligence-hub-phase2/bin/ha-hub.py --port 8000 --log-level INFO'
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryLimit=1G
CPUQuota=50%

# Health monitoring
WatchdogSec=60

[Install]
WantedBy=default.target
```

### 2. Enable and Start Service

```bash
# Reload systemd
systemctl --user daemon-reload

# Enable service (start on boot)
systemctl --user enable ha-intelligence-hub.service

# Start service
systemctl --user start ha-intelligence-hub.service

# Check status
systemctl --user status ha-intelligence-hub.service
```

### 3. View Logs

```bash
# Follow logs in real-time
journalctl --user -u ha-intelligence-hub.service -f

# View recent logs
journalctl --user -u ha-intelligence-hub.service -n 100

# Filter by date
journalctl --user -u ha-intelligence-hub.service --since "1 hour ago"
```

## Tailscale Serve (Remote Access)

To access the dashboard remotely via Tailscale:

### 1. Start Tailscale Serve

```bash
# Expose hub on Tailscale network
tailscale serve --bg --https=8000 http://localhost:8000
```

### 2. Access Dashboard

From any device on your Tailscale network:
```
https://justin-linux.tail828051.ts.net:8000/ui
```

### 3. Stop Tailscale Serve

```bash
tailscale serve --https=8000 off
```

## Log Rotation

Systemd journal handles log rotation automatically. To configure:

```bash
# Edit journal configuration (requires sudo)
sudo nano /etc/systemd/journald.conf
```

Add/modify:
```ini
[Journal]
SystemMaxUse=500M
RuntimeMaxUse=100M
MaxRetentionSec=1week
```

Restart journald:
```bash
sudo systemctl restart systemd-journald
```

## Backup and Restore

### Backup

Critical data to backup:
- Cache database: `~/ha-logs/intelligence/cache/hub.db`
- ML models: `~/ha-logs/intelligence/models/`
- Training data: `~/ha-logs/intelligence/daily/`

Backup script:
```bash
#!/bin/bash
BACKUP_DIR=~/backups/ha-intelligence-hub
DATE=$(date +%Y-%m-%d)

mkdir -p "$BACKUP_DIR"

# Backup cache
cp ~/ha-logs/intelligence/cache/hub.db "$BACKUP_DIR/hub-$DATE.db"

# Backup models
tar -czf "$BACKUP_DIR/models-$DATE.tar.gz" ~/ha-logs/intelligence/models/

# Backup training data (last 30 days)
find ~/ha-logs/intelligence/daily/ -name "*.jsonl" -mtime -30 \
  | tar -czf "$BACKUP_DIR/training-data-$DATE.tar.gz" -T -

# Keep only last 7 backups
ls -t "$BACKUP_DIR"/hub-*.db | tail -n +8 | xargs -r rm
ls -t "$BACKUP_DIR"/models-*.tar.gz | tail -n +8 | xargs -r rm
ls -t "$BACKUP_DIR"/training-data-*.tar.gz | tail -n +8 | xargs -r rm
```

Save as `~/bin/backup-ha-hub.sh`, make executable:
```bash
chmod +x ~/bin/backup-ha-hub.sh
```

Add to cron (daily at 2 AM):
```bash
crontab -e
```

Add line:
```
0 2 * * * ~/bin/backup-ha-hub.sh
```

### Restore

```bash
# Stop service
systemctl --user stop ha-intelligence-hub.service

# Restore cache
cp ~/backups/ha-intelligence-hub/hub-2026-02-11.db ~/ha-logs/intelligence/cache/hub.db

# Restore models
tar -xzf ~/backups/ha-intelligence-hub/models-2026-02-11.tar.gz -C ~/

# Restore training data
tar -xzf ~/backups/ha-intelligence-hub/training-data-2026-02-11.tar.gz -C ~/

# Start service
systemctl --user start ha-intelligence-hub.service
```

## Health Monitoring

### Automated Health Checks

Create `~/bin/check-ha-hub-health.sh`:

```bash
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
LOG_FILE=~/logs/ha-hub-health.log

# Check if hub is responding
if curl -s -f "$HEALTH_URL" > /dev/null; then
  echo "$(date): Hub is healthy" >> "$LOG_FILE"
  exit 0
else
  echo "$(date): Hub is DOWN - restarting" >> "$LOG_FILE"
  systemctl --user restart ha-intelligence-hub.service
  exit 1
fi
```

Make executable:
```bash
chmod +x ~/bin/check-ha-hub-health.sh
```

Add to cron (every 5 minutes):
```
*/5 * * * * ~/bin/check-ha-hub-health.sh
```

### Prometheus Metrics (Optional)

Add to `hub/api.py` for Prometheus scraping:

```python
from prometheus_client import generate_latest, Counter, Gauge

# Metrics
requests_total = Counter('hub_requests_total', 'Total HTTP requests')
cache_size = Gauge('hub_cache_size_bytes', 'Cache database size')
modules_active = Gauge('hub_modules_active', 'Number of active modules')

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

## Performance Tuning

### SQLite Optimization

Add to cache initialization:

```python
# In hub/cache.py
await self._conn.execute("PRAGMA journal_mode=WAL")
await self._conn.execute("PRAGMA synchronous=NORMAL")
await self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
await self._conn.execute("PRAGMA temp_store=MEMORY")
```

### Python Optimization

Run with optimizations:
```bash
python -O bin/ha-hub.py  # Removes assert statements
```

### Memory Management

Monitor memory usage:
```bash
# View service memory usage
systemctl --user status ha-intelligence-hub.service | grep Memory

# Detailed memory info
ps aux | grep ha-hub.py
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
journalctl --user -u ha-intelligence-hub.service -n 50

# Verify environment
systemctl --user show ha-intelligence-hub.service | grep Environment

# Test manually
. ~/.env && ./venv/bin/python bin/ha-hub.py
```

### High CPU Usage

```bash
# Check module schedules (avoid overlap)
# Edit bin/ha-hub.py to adjust:
# - discovery_interval_hours (default: 24)
# - training_interval_days (default: 7)
```

### Cache Database Locked

```bash
# Stop service
systemctl --user stop ha-intelligence-hub.service

# Check for stale connections
fuser ~/ha-logs/intelligence/cache/hub.db

# Remove lock file (if exists)
rm ~/ha-logs/intelligence/cache/hub.db-shm
rm ~/ha-logs/intelligence/cache/hub.db-wal

# Start service
systemctl --user start ha-intelligence-hub.service
```

### Dashboard Not Loading

```bash
# Check if service is running
systemctl --user status ha-intelligence-hub.service

# Verify port is listening
netstat -tuln | grep 8000

# Test health endpoint
curl http://localhost:8000/health

# Check firewall (if accessing remotely)
sudo ufw status
```

## Security Considerations

### Local Access Only

If not using Tailscale, ensure hub binds to localhost only:

```bash
# In bin/ha-hub.py, default --host is 127.0.0.1
./venv/bin/python bin/ha-hub.py --host 127.0.0.1 --port 8000
```

### HA Token Security

Protect `~/.env`:
```bash
chmod 600 ~/.env
```

Never commit `~/.env` to git:
```bash
# Add to .gitignore
echo "~/.env" >> .gitignore
```

### Tailscale ACLs

Restrict access to specific devices in Tailscale ACL:

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["justin@github"],
      "dst": ["justin-linux:8000"]
    }
  ]
}
```

## Upgrade Procedure

```bash
# Stop service
systemctl --user stop ha-intelligence-hub.service

# Backup database
cp ~/ha-logs/intelligence/cache/hub.db ~/ha-logs/intelligence/cache/hub.db.backup

# Pull latest code
cd ~/Documents/projects/ha-intelligence-hub-phase2
git pull origin phase2-hub-core

# Update dependencies
. venv/bin/activate
pip install -r requirements.txt --upgrade

# Run migrations (if any)
# ./venv/bin/python scripts/migrate.py

# Start service
systemctl --user start ha-intelligence-hub.service

# Verify upgrade
curl http://localhost:8000/health
```

## Monitoring Dashboard

Consider setting up a monitoring dashboard with:
- Grafana for metrics visualization
- Prometheus for metrics collection
- AlertManager for alerts

Example Grafana dashboard queries:
- Hub uptime: `up{job="ha-intelligence-hub"}`
- Request rate: `rate(hub_requests_total[5m])`
- Module count: `hub_modules_active`
- Cache size: `hub_cache_size_bytes`
