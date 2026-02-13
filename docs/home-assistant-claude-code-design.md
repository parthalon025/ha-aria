# Home Assistant × Claude Code Integration

**Date:** 2026-02-10
**Status:** Design Complete — Ready for Implementation
**Author:** Justin McFarland + Claude Code

---

## Overview

Give Claude Code full access to a large (100+ devices) Home Assistant OS instance running on a Raspberry Pi. Phased approach: monitor and understand first, then build and automate.

## Current State

- **Home Assistant OS** on Raspberry Pi, same LAN as justin-linux
- **100+ devices**, many automations/scripts, multiple integrations
- **Pi is NOT on Tailscale** — LAN-only access via local IP on port 8123
- **Claude Code** on justin-linux (Ubuntu 24.04, Docker, established MCP server pattern)
- **Existing brief pipeline** — `telegram-brief` sends daily sitreps at 7am, 12pm, 9pm

## Goals

1. **Monitor & insights** — surface device health, log patterns, anomalies, optimization opportunities
2. **Sitrep integration** — fold HA health into existing Telegram daily brief
3. **Build automations** — create/edit automations, scripts, dashboards via Claude Code
4. **Log analysis** — learn from historical data, find recurring issues, suggest improvements

---

## Architecture

```
┌─────────────────────────┐          ┌──────────────────────────┐
│   justin-linux          │          │   Raspberry Pi           │
│                         │          │                          │
│  Claude Code            │          │  Home Assistant OS       │
│    ├─ hass-mcp server ──┼─── LAN ─▶│    ├─ REST API :8123     │
│    ├─ HA skills/        │          │    ├─ WebSocket API       │
│    └─ log analyzer      │          │    ├─ MCP Server (built-in)│
│                         │          │    └─ SSH Add-on          │
│  Periodic sync:         │          │                          │
│    ~/ha-logs/           │◀── rsync─┤  /config/                │
│    ~/ha-config-mirror/  │◀── sshfs─┤  home-assistant.log      │
│                         │          │                          │
│  telegram-brief         │          │  100+ devices            │
│    └─ ha-health section │          │  Automations/scripts     │
└─────────────────────────┘          └──────────────────────────┘
```

### Why voska/hass-mcp as primary bridge

- Runs as Docker on the workstation — no Pi resources consumed
- Token-efficient JSON responses (critical for 100+ entity lists)
- Built-in automation health checks and entity auditing
- One-liner setup: `claude mcp add hass-mcp`

### Why NOT the alternatives (for phase 1)

- **HA built-in MCP server** runs on the Pi — already loaded with 100+ devices, don't want to add LLM-facing workload
- **philippb/claude-homeassistant** SSH/SSHFS approach is overkill for monitoring — layered in at phase 2 for config management

---

## Phase 1: Connect & Monitor (MVP)

**Timeline:** 2-3 sessions of work
**Goal:** Claude Code can read all HA state and surface actionable insights

### Step 1: MCP Server Setup

Install voska/hass-mcp on justin-linux:

```bash
claude mcp add hass-mcp \
  -e HA_URL=http://<pi-ip>:8123 \
  -e HA_TOKEN=<long-lived-access-token> \
  -- docker run -i --rm -e HA_URL -e HA_TOKEN voska/hass-mcp
```

Create a **dedicated read-only HA user** for Claude Code. Generate a long-lived access token from that user. Store token in `~/.env`.

This gives Claude Code:
- Entity states (all 100+ devices)
- Domain summaries (lights, sensors, switches, climate, etc.)
- Automation listing and status
- Smart entity search by name, type, or state

### Step 2: Log Sync Pipeline

HAOS doesn't expose logs easily via API alone. Set up periodic pull:

1. Install **Advanced SSH & Web Terminal** add-on on HAOS
2. Generate SSH key on justin-linux, add public key to the add-on
3. Cron job every 15 minutes syncs logs:

```bash
# Add to crontab
*/15 * * * * rsync -az ha-pi:/config/home-assistant.log ~/ha-logs/current.log 2>/dev/null
```

4. Daily rotation script moves `current.log` → `~/ha-logs/YYYY-MM-DD.log`
5. Keep 30 days of history, compress older logs

### Step 3: Claude Code Skills

Three custom skills in `~/.claude/skills/`:

#### `/ha-status` — On-Demand Dashboard

Queries MCP + reads synced logs to display:
- Device availability (online/offline/unavailable counts)
- Battery levels below 20%
- Entities stuck in "unavailable" for >1h
- Error count in last 24h
- Automation success/failure rate

#### `/ha-insights` — Log Analysis

Analyzes synced log files for:
- **Error patterns** — recurring errors, grouped by integration
- **Noisy entities** — sensors updating too frequently, flooding the event bus
- **Flaky devices** — entities toggling between available/unavailable
- **Automation failures** — template errors, unavailable entity references
- **State change frequency** — which entities change most, unexpected patterns

#### `/ha-audit` — Automation Health Check

Uses MCP's built-in `automation_health_check` plus custom analysis:
- Redundant automations (same trigger, similar actions)
- Orphaned entities referenced in automations but no longer exist
- Naming inconsistencies (`entity_naming_consistency` via MCP)
- Automations that never fire (enabled but 0 triggers in 30 days)
- Conflict detection (automations that could fight each other)

### Step 4: Sitrep Integration

New module: `~/.local/bin/ha-health-report`

Queries HA REST API + parses synced logs. Outputs structured text that `telegram-brief` includes as a section.

#### Morning Brief (7am) adds:
- Overnight anomalies (devices that went offline, unexpected state changes)
- Failed automations in last 12h
- Battery levels below 20%
- Sensors stuck in "unavailable"

#### Midday Brief (12pm, weekdays) adds:
- Energy usage today vs. average (if power monitoring exists)
- Automation execution summary (fired/failed count)

#### Evening Brief (9pm) adds:
- Full day health score (0-100): device availability × error rate × automation success
- Devices offline for >1h during the day
- Suggestions queue (automations to review, entities to rename)

---

## Phase 2: Build & Automate

**Timeline:** After phase 1 proves value (2+ weeks)
**Goal:** Claude Code can create and modify HA configurations safely

### Config Management Layer

Add philippb/claude-homeassistant for:
- **SSHFS mount** of Pi's `/config/` → `~/ha-config-mirror/`
- **Rsync push/pull** with pre-deploy validation (syntax check)
- **Entity dependency indexing** — knows which automations reference which entities

### New Skills

#### `/ha-auto` — Create/Edit Automations

Workflow:
1. User describes automation in natural language
2. Claude queries MCP for matching entities (lights, sensors, person trackers)
3. Generates automation YAML
4. Shows YAML for user approval
5. Backs up current config
6. Pushes to Pi via rsync
7. Calls HA REST API to reload automations
8. Verifies it loaded without errors in the log

#### `/ha-script` — Create/Edit Scripts

Same validate-push-reload cycle as automations.

#### `/ha-dashboard` — Generate/Modify Dashboards

Generate Lovelace YAML from natural language descriptions. Preview-safe — dashboards don't affect device behavior.

#### `/ha-rollback` — Emergency Restore

Restores last known-good config from `~/ha-config-backups/` and reloads.

### Safety Rails for Write Operations

- All config changes require explicit user approval (YAML diff shown first)
- Automatic backup before every push (timestamped in `~/ha-config-backups/`)
- Never touch `configuration.yaml` core settings — only automations, scripts, dashboards, scenes
- Rollback is always one command away

---

## Security

### Network Layer
- HA API traffic stays on LAN only (Pi not on Tailscale)
- Long-lived access token stored in `~/.env` (same pattern as existing secrets)
- SSH key auth for config sync (no passwords)

### Access Scoping
- **Dedicated HA user** for Claude Code with limited permissions
- **Phase 1:** read-only token (entity states, logs, device info)
- **Phase 2:** separate read-write token, enabled only when actively building
- Admin token is never exposed to Claude Code

### Operational Safety
- **Exclusion list** for safety-critical devices: locks, garage doors, alarms, security cameras
  - These entities are excluded from MCP responses and automation templates
- **Human approval required** for all config changes — Claude proposes, user confirms
- **Config backup before every write** — automatic, timestamped, 30-day retention
- **No autonomous execution** — Claude Code only acts when invoked by user
- **Rollback always available** — `/ha-rollback` restores last known-good

### What This Does NOT Protect Against
- Bad automations that the user approves — syntax is validated but logic edge cases aren't
- LAN-level attacks — existing risk, not introduced by this integration
- Pi hardware failure — out of scope (standard HA backup practices apply)

---

## Lean Gate

| Gate | Answer |
|------|--------|
| **Hypothesis** | Claude Code + MCP + log analysis will reduce HA maintenance time and surface issues currently being missed in a 100+ device setup |
| **MVP** | Phase 1 only: MCP bridge + log sync + `/ha-status` skill + morning brief integration |
| **First 5 users** | Single user (Justin) — personal infrastructure |
| **Success metric** | Within 2 weeks: Claude surfaces at least 3 actionable insights not found manually (dead devices, redundant automations, error patterns) |
| **Pivot trigger** | If after 2 weeks of phase 1, all surfaced insights are already known — monitoring layer isn't adding value, stop and reassess |

---

## Implementation Order

### Phase 1 (MVP — 2-3 sessions)
1. Create dedicated HA user + read-only long-lived token
2. Install Advanced SSH & Web Terminal add-on on HAOS
3. Set up SSH key auth from justin-linux → Pi
4. Install voska/hass-mcp Docker container + `claude mcp add`
5. Set up log sync cron (15-min rsync)
6. Build `/ha-status` skill
7. Build `/ha-insights` skill
8. Build `/ha-audit` skill
9. Build `ha-health-report` module for telegram-brief integration
10. Validate: run for 2 weeks, measure against success metric

### Phase 2 (if phase 1 proves value)
11. Set up SSHFS mount + rsync push/pull (philippb approach)
12. Create read-write HA token (separate from read-only)
13. Build config backup system (`~/ha-config-backups/`)
14. Build `/ha-auto` skill (automations)
15. Build `/ha-script` skill
16. Build `/ha-dashboard` skill
17. Build `/ha-rollback` skill
18. Define exclusion list for safety-critical devices

### Future Considerations
- Put Pi on Tailscale for remote HA management
- Energy dashboard with historical trends (if power monitoring exists)
- Integration with Vector platform for job-context-aware home automations
- Voice control via Telegram listener ("turn off the lights" → HA API)

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `~/.claude/skills/ha-status/SKILL.md` | On-demand HA dashboard skill |
| `~/.claude/skills/ha-insights/SKILL.md` | Log analysis and pattern detection |
| `~/.claude/skills/ha-audit/SKILL.md` | Automation health check and audit |
| `~/.local/bin/ha-health-report` | Sitrep module for telegram-brief |
| `~/.env` | HA_URL, HA_TOKEN additions |
| `~/ha-logs/` | Synced HA log directory |
| `~/ha-config-mirror/` | SSHFS mount point (phase 2) |
| `~/ha-config-backups/` | Config snapshots (phase 2) |
