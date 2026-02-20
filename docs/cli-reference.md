# ARIA CLI Reference

Reference doc for CLAUDE.md. All commands route through the unified `aria` entry point (`aria/cli.py`).

| Command | What it does |
|---------|-------------|
| `aria serve` | Start real-time hub + dashboard (replaces `bin/ha-hub.py`) |
| `aria full` | Full daily pipeline: snapshot → predict → report |
| `aria snapshot` | Collect current HA state snapshot |
| `aria predict` | Generate predictions from latest snapshot |
| `aria score` | Score yesterday's predictions against actuals |
| `aria retrain` | Retrain ML models from accumulated data |
| `aria meta-learn` | LLM meta-learning to tune feature config |
| `aria check-drift` | Detect concept drift in predictions |
| `aria correlations` | Compute entity co-occurrence correlations |
| `aria suggest-automations` | Generate HA automation YAML via LLM |
| `aria prophet` | Train Prophet seasonal forecasters |
| `aria occupancy` | Bayesian occupancy estimation |
| `aria power-profiles` | Analyze per-outlet power consumption |
| `aria sequences train` | Train Markov chain model from logbook sequences |
| `aria sequences detect` | Detect anomalous event sequences |
| `aria snapshot-intraday` | Collect intraday snapshot (used internally by hub) |
| `aria sync-logs` | Sync HA logbook to local JSON |
| `aria watchdog` | Run health checks and alert on failures |
| `aria status` | Show ARIA hub status |
| `aria demo` | Generate synthetic demo data for visual testing |
| `aria capabilities list` | List all registered capabilities (--layer, --status, --verbose) |
| `aria capabilities verify` | Validate all capabilities against tests/config/deps |
| `aria capabilities export` | Export capability registry as JSON |

## aria audit

Query, stream, and export the audit log (stored in `audit.db`, separate from `hub.db`).

| Subcommand | What it does |
|------------|-------------|
| `aria audit events` | Query audit events with optional filters |
| `aria audit requests` | Query HTTP request log |
| `aria audit timeline SUBJECT` | Chronological event history for a single subject |
| `aria audit stats` | Aggregate counts and error rates |
| `aria audit startups` | List recent hub startup records |
| `aria audit curation ENTITY_ID` | Curation change history for one entity |
| `aria audit verify` | Verify tamper-evident checksums across stored events |
| `aria audit export` | Archive events to disk before a cutoff date |
| `aria audit tail` | Live-stream new audit events (WebSocket-backed) |

### Flags

```
aria audit events [--type TYPE] [--source SOURCE] [--subject SUBJECT]
                  [--severity SEVERITY] [--since ISO] [--until ISO]
                  [--request-id ID] [--limit N] [--json]

aria audit requests [--path PATH] [--method METHOD] [--status CODE]
                    [--since ISO] [--limit N] [--json]

aria audit timeline SUBJECT [--since ISO] [--json]

aria audit stats [--since ISO]

aria audit startups [--limit N]

aria audit curation ENTITY_ID [--limit N]

aria audit verify [--since ISO]

aria audit export --before ISO [--output DIR]

aria audit tail [--types TYPES] [--severity-min LEVEL]
```

`--json` on any query subcommand outputs newline-delimited JSON for piping.

## Support Scripts

| Script | What it does |
|--------|-------------|
| `bin/check-ha-health.sh` | Validates HA connectivity + core stats before batch timers run (used by all snapshot/training systemd timers) |

Engine commands delegate to `aria.engine.cli` with old-style flags internally.
